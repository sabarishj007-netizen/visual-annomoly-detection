# Crowd-Gathering Detection — Attribution

Crowd-gathering detection in `app.py` (`run_crowd_gathering_detection`) is
adapted from **Crowd-Gathering-Detection**, MIT licensed:
https://github.com/ (original repo as uploaded, "Crowd-Gathering-Detection-main")

## What was reused
- The core idea: run a stock YOLO person detector, get bounding-box centers,
  and flag people who are close together on-screen.
- The stock `yolov8n.pt` COCO weights (already present in this project).

## What was changed
The upstream script (`crowd_gathering.py`) flags **any pair** of people
closer than a pixel threshold as a "gathering" — including two people just
walking side by side, which fires on ordinary foot traffic. This project's
version (`run_crowd_gathering_detection`) instead:
- Clusters people transitively with a union-find over the same pairwise
  distance test (so a chain of nearby people forms one group, not several
  overlapping pairs).
- Only flags a cluster once it reaches `--crowd_min_group` people (default 4),
  which is closer to what "gathering" should mean for campus security.
- Reports a live "People: N" count and integrates with the existing
  Socket.IO event/cooldown/dashboard pipeline instead of a standalone
  `cv2.imshow` window.

## License
MIT — see the original repository for the full license text.
