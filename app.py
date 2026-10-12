"""
Campus AI Security - integrated detection server.

Combines FIVE models in one webcam loop:
  - best.pt                      YOLOv11 detector -> fallen / sitting / standing (real per-person bbox)
  - yolov5s_best.pt              YOLOv5 detector -> fire (single class, needs the yolov5 repo to load)
  - smoking_detection_trained2.pt   YOLO detector -> smoking (single class)
  - mobilenetv2_tsm_violence_best.pth   MobileNetV2+TSM -> whole-frame Violence/Non-Violence
  - yolov8n.pt                   Stock COCO YOLOv8n detector -> person (class 0), reused for crowd-
                                  gathering: people are clustered by on-screen proximity and a
                                  cluster at or above a minimum size is flagged as a gathering
                                  (adapted from "Crowd-Gathering-Detection", MIT licensed)

NOTE: the old pose+MLP (FallMLP) fall detector has been removed and replaced
by best.pt, per request. This means fall detection is no longer skeleton-only
-- it now runs directly on RGB frames like the other three original detectors.

Runs ONE webcam capture loop, feeds all five models, draws CCTV-style
overlays, and streams annotated frames + structured events to a dashboard
over Socket.IO. Must run on your local machine (needs real webcam access).

Setup:
    pip install torch torchvision ultralytics opencv-python flask flask-socketio

    # yolov5s_best.pt needs the separate yolov5 repo (NOT the ultralytics
    # package -- it will refuse to load this checkpoint format):
    git clone https://github.com/ultralytics/yolov5 yolov5_repo
    pip install -r yolov5_repo/requirements.txt

Usage:
    python app.py --fight_checkpoint mobilenetv2_tsm_violence_best.pth \
                   --fall_checkpoint best.pt \
                   --fire_checkpoint yolov5s_best.pt \
                   --smoking_checkpoint smoking_detection_trained2.pt \
                   --crowd_checkpoint yolov8n.pt \
                   --yolov5_repo yolov5_repo

Then open http://localhost:5000 in a browser.

NOTICE (MIT attribution):
    Crowd-gathering detection is adapted from the "Crowd-Gathering-Detection"
    project (MIT licensed). The upstream script flags ANY two people closer
    than a pixel threshold as a "gathering", which fires constantly for
    ordinary foot traffic (e.g. two friends walking together). Here it's
    reworked into simple proximity clustering (union-find over the same
    pairwise distance test) so an alert only fires when a cluster reaches
    --crowd_min_group people -- closer to what "gathering" should mean for
    campus security.
"""

import argparse
import re
import time
import uuid
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from ultralytics import YOLO
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------------------------------------------------------------
# 1. Fall model -- best.pt, YOLOv11 detector, classes: fallen/sitting/standing
# ----------------------------------------------------------------------
def run_person_state_detection(frame, fall_model, conf=0.5):
    """Returns a list of dicts, one per detected person:
       {bbox: (x1,y1,x2,y2), cls_name: 'fallen'|'sitting'|'standing', confidence: float}
    """
    results = fall_model(frame, verbose=False, conf=conf)[0]
    people = []
    if results.boxes is None or len(results.boxes) == 0:
        return people

    names = results.names
    for box in results.boxes:
        cls_id = int(box.cls[0])
        people.append({
            "bbox": tuple(box.xyxy[0].cpu().numpy().astype(int)),
            "cls_name": names[cls_id],
            "confidence": float(box.conf[0]),
        })
    return people


# ----------------------------------------------------------------------
# 1b. Fire model -- yolov5s_best.pt, YOLOv5 architecture (separate repo)
# ----------------------------------------------------------------------
def run_fire_detection(frame, fire_model, conf=0.4):
    """yolov5 AutoShape models take BGR/RGB numpy frames directly and
       return a pandas-like results object; results.xyxy[0] is a tensor of
       [x1, y1, x2, y2, confidence, class]. Only class 0 ('fire') exists.
    """
    fire_model.conf = conf
    results = fire_model(frame)
    detections = []
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, score, cls_id = det
        detections.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": float(score),
        })
    return detections


# ----------------------------------------------------------------------
# 1c. Smoking model -- smoking_detection_trained2.pt, single class 'smoking'
# ----------------------------------------------------------------------
def run_smoking_detection(frame, smoking_model, conf=0.4):
    results = smoking_model(frame, verbose=False, conf=conf)[0]
    detections = []
    if results.boxes is None or len(results.boxes) == 0:
        return detections
    for box in results.boxes:
        detections.append({
            "bbox": tuple(box.xyxy[0].cpu().numpy().astype(int)),
            "confidence": float(box.conf[0]),
        })
    return detections


# ----------------------------------------------------------------------
# 1d. Crowd-gathering -- yolov8n.pt, stock COCO detector, person = class 0.
#     Adapted from "Crowd-Gathering-Detection" (MIT): people are clustered by
#     proximity (union-find over a pairwise center-distance threshold), and a
#     cluster is only flagged once it reaches --crowd_min_group people, so
#     ordinary foot traffic doesn't trip it the way a raw pairwise check would.
# ----------------------------------------------------------------------
def run_crowd_gathering_detection(frame, crowd_model, conf=0.4, distance_threshold=120, min_group=4):
    """Returns a list of gathering clusters, each:
       {bboxes: [(x1,y1,x2,y2), ...], confidences: [float, ...]}
       Only clusters with >= min_group people are returned as gatherings.
    """
    results = crowd_model(frame, verbose=False, conf=conf)[0]
    boxes, confidences = [], []
    if results.boxes is not None:
        for box in results.boxes:
            if int(box.cls[0]) != 0:  # COCO class 0 = person
                continue
            boxes.append(tuple(box.xyxy[0].cpu().numpy().astype(int)))
            confidences.append(float(box.conf[0]))

    n = len(boxes)
    total_people = n
    if n < min_group:
        return [], total_people

    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]

    # Union-find: group anyone within distance_threshold of anyone else in
    # the same group (chains transitively, unlike a single pairwise check).
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if (dx * dx + dy * dy) ** 0.5 < distance_threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    gatherings = []
    for members in groups.values():
        if len(members) >= min_group:
            gatherings.append({
                "bboxes": [boxes[i] for i in members],
                "confidences": [confidences[i] for i in members],
            })
    return gatherings, total_people


# ----------------------------------------------------------------------
# 2. Violence model -- copied from test_violence_webcam_2.py, unchanged
# ----------------------------------------------------------------------
def temporal_shift(x, n_segment, fold_div=8):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    return out.view(nt, c, h, w)


class ShiftedBlock(nn.Module):
    def __init__(self, block, n_segment, fold_div):
        super().__init__()
        self.block = block
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        x = temporal_shift(x, self.n_segment, self.fold_div)
        return self.block(x)


def build_violence_model(n_segment=8, fold_div=8, num_classes=2):
    model = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
    for i in range(1, 18):
        model.features[i] = ShiftedBlock(model.features[i], n_segment, fold_div)
    return model


def load_violence_checkpoint(model, checkpoint_path):
    raw_sd = torch.load(checkpoint_path, map_location="cpu")
    new_sd = {}
    for k, v in raw_sd.items():
        k2 = k[len("base_model."):] if k.startswith("base_model.") else k
        m = re.match(r"^features\.(\d+)\.1\.(.*)$", k2)
        if m and 1 <= int(m.group(1)) <= 17:
            k2 = f"features.{m.group(1)}.block.{m.group(2)}"
        new_sd[k2] = v
    model.load_state_dict(new_sd, strict=True)
    return model


VIOLENCE_CLASS_NAMES = ["Non-Violence", "Violence"]

VIOLENCE_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_violence_detection(frame_buffer, violence_model, device, n_segment, threshold=0.3):
    """frame_buffer: deque of preprocessed tensors, length n_segment.
       Returns (is_violence, confidence) or None if buffer isn't full yet.
    """
    if len(frame_buffer) < n_segment:
        return None

    clip = torch.stack(list(frame_buffer), dim=0).to(device)
    with torch.no_grad():
        logits = violence_model(clip)
        probs = F.softmax(logits, dim=1).mean(dim=0)
        violence_prob = probs[1].item()

    return (violence_prob >= threshold, violence_prob)


# ----------------------------------------------------------------------
# 3. CCTV-style overlay drawing
# ----------------------------------------------------------------------
def draw_alert_overlay(frame, bbox, label, brand_name="Campus AI Security"):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    text_y = max(y2 + 30, 30)
    cv2.putText(frame, "!", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    cv2.putText(frame, label, (x1 + 30, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (20, h - 60), (340, h - 20), (255, 255, 255), -1)
    cv2.putText(frame, label, (35, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(frame, brand_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def enclosing_bbox(boxes):
    """Given a list of (x1,y1,x2,y2), return the box that contains them all."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (int(x1), int(y1), int(x2), int(y2))


# ----------------------------------------------------------------------
# 4. Main detection loop
# ----------------------------------------------------------------------
FALL_COOLDOWN_SEC = 5
FIGHT_COOLDOWN_SEC = 5
FIRE_COOLDOWN_SEC = 8
SMOKING_COOLDOWN_SEC = 8
CROWD_GATHERING_COOLDOWN_SEC = 6


def video_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fall_model = YOLO(args.fall_checkpoint)
    print("Fall detector (best.pt) loaded. Classes:", fall_model.names)

    smoking_model = YOLO(args.smoking_checkpoint)
    print("Smoking detector loaded. Classes:", smoking_model.names)

    crowd_model = YOLO(args.crowd_checkpoint)
    print("Crowd-gathering detector (stock COCO) loaded. Classes:", crowd_model.names)

    # yolov5s_best.pt needs the actual yolov5 repo code to unpickle correctly --
    # the `ultralytics` package alone will refuse this checkpoint format.
    fire_model = torch.hub.load(args.yolov5_repo, "custom", path=args.fire_checkpoint, source="local")
    print("Fire detector (yolov5s) loaded. Classes:", fire_model.names)

    violence_model = build_violence_model(n_segment=args.n_segment, fold_div=args.fold_div)
    violence_model = load_violence_checkpoint(violence_model, args.fight_checkpoint)
    violence_model.to(device).eval()
    print("Violence classifier loaded.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    frame_buffer = deque(maxlen=args.n_segment)
    last_alert = {"fall": 0.0, "fight": 0.0, "fire": 0.0, "smoking": 0.0, "crowd_gathering": 0.0}
    cooldowns = {"fall": FALL_COOLDOWN_SEC, "fight": FIGHT_COOLDOWN_SEC,
                 "fire": FIRE_COOLDOWN_SEC, "smoking": SMOKING_COOLDOWN_SEC,
                 "crowd_gathering": CROWD_GATHERING_COOLDOWN_SEC}

    def maybe_emit(event_type, confidence, now):
        if (now - last_alert[event_type]) > cooldowns[event_type]:
            last_alert[event_type] = now
            socketio.emit("new_event", {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "confidence": confidence,
                "timestamp": now,
                "status": "open",
            })

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()

        # ---- Fall / sitting / standing detection (per-person, every frame) ----
        people = run_person_state_detection(frame, fall_model, conf=args.fall_conf)
        fallen = [p for p in people if p["cls_name"] == "fallen"]
        for p in fallen:
            frame = draw_alert_overlay(frame, p["bbox"], "Fall Detected")
        if fallen:
            maybe_emit("fall", max(p["confidence"] for p in fallen), now)

        # ---- Fire detection ----
        fires = run_fire_detection(frame, fire_model, conf=args.fire_conf)
        for f in fires:
            frame = draw_alert_overlay(frame, f["bbox"], "Fire Detected")
        if fires:
            maybe_emit("fire", max(f["confidence"] for f in fires), now)

        # ---- Smoking detection ----
        smokers = run_smoking_detection(frame, smoking_model, conf=args.smoking_conf)
        for s in smokers:
            frame = draw_alert_overlay(frame, s["bbox"], "Smoking Detected")
        if smokers:
            maybe_emit("smoking", max(s["confidence"] for s in smokers), now)

        # ---- Crowd-gathering detection (proximity clustering of stock person detections) ----
        gatherings, total_people = run_crowd_gathering_detection(
            frame, crowd_model, conf=args.crowd_conf,
            distance_threshold=args.crowd_distance_threshold, min_group=args.crowd_min_group,
        )
        h, w = frame.shape[:2]
        cv2.putText(frame, f"People: {total_people}", (w - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        for g in gatherings:
            box = enclosing_bbox(g["bboxes"])
            frame = draw_alert_overlay(frame, box, f"Crowd Gathering ({len(g['bboxes'])})")
        if gatherings:
            best = max(gatherings, key=lambda g: len(g["bboxes"]))
            maybe_emit("crowd_gathering", max(best["confidences"]), now)

        # ---- Violence detection (rolling 8-frame buffer, whole-frame classification) ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(VIOLENCE_PREPROCESS(rgb))
        violence_result = run_violence_detection(
            frame_buffer, violence_model, device, args.n_segment, threshold=args.fight_threshold
        )

        if violence_result and violence_result[0]:
            is_violence, conf = violence_result
            # Proxy box: enclose all detected people (from the fall model's boxes)
            # so the alert still LOOKS boxed like the reference CCTV screenshots.
            # This is NOT per-person attribution -- violence is a whole-frame classification.
            if people:
                box = enclosing_bbox([p["bbox"] for p in people])
                frame = draw_alert_overlay(frame, box, "Fight Detected")
            else:
                h, w = frame.shape[:2]
                frame = draw_alert_overlay(frame, (10, 10, w - 10, h - 10), "Fight Detected")
            maybe_emit("fight", conf, now)

        # ---- Stream annotated frame to dashboard ----
        _, buf = cv2.imencode(".jpg", frame)
        jpg_b64 = base64.b64encode(buf).decode("utf-8")
        socketio.emit("frame_update", {"image": jpg_b64})

        time.sleep(0.01)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fight_checkpoint", default="mobilenetv2_tsm_violence_best.pth")
    parser.add_argument("--fall_checkpoint", default="best.pt")
    parser.add_argument("--fire_checkpoint", default="yolov5s_best.pt")
    parser.add_argument("--smoking_checkpoint", default="smoking_detection_trained2.pt")
    parser.add_argument("--crowd_checkpoint", default="yolov8n.pt")
    parser.add_argument("--yolov5_repo", default="yolov5_repo",
                         help="Path to a local clone of github.com/ultralytics/yolov5 (needed to load yolov5s_best.pt)")
    parser.add_argument("--n_segment", type=int, default=8)
    parser.add_argument("--fold_div", type=int, default=8)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fall_conf", type=float, default=0.5)
    parser.add_argument("--fire_conf", type=float, default=0.4)
    parser.add_argument("--smoking_conf", type=float, default=0.4)
    parser.add_argument("--crowd_conf", type=float, default=0.4)
    parser.add_argument("--crowd_distance_threshold", type=float, default=120,
                         help="Max pixel distance between person centers to count as 'close' (frame-resolution dependent).")
    parser.add_argument("--crowd_min_group", type=int, default=4,
                         help="Minimum people in a proximity cluster before it's flagged as a gathering.")
    parser.add_argument("--fight_threshold", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    threading.Thread(target=video_loop, args=(args,), daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000)
