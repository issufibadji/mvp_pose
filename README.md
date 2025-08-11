# MVP Pose

Minimal gesture recognition demo using MediaPipe Pose (33 keypoints) and
simple geometric rules. The pipeline is kept modular so other pose
providers (MoveNet, RTMPose) can be added later without changing the
rest of the code.

## Installation

Python 3.10+

```bash
pip install -r requirements.txt
```

### Colab

```python
!pip install -r requirements.txt
```

## Running

Webcam:

```bash
python main.py --source 0
```

Video file:

```bash
python main.py --source path/to/video.mp4
```

Outputs are written to `out/`:

* `result_demo.mp4` – overlayed video
* `events.csv` – detected gesture events (t_sec, event)
* `keypoints.csv` – raw keypoints per frame

## Metrics

Given a manually labelled `events_gt.csv`, compute precision/recall/F1:

```bash
python -m metrics.evaluate_events out/events.csv events_gt.csv
```

## Thresholds

| gesture    | rule summary                      |
|------------|-----------------------------------|
| arm_raise  | wrist above shoulder for 3 frames |
| squat      | knee <100° & hip drop >0.06       |
| sit_down   | knees 70–110°, low hip variance   |

`--min-vis` controls the visibility gate (default 0.5).

## Acceptance

Process a short video: `out/result_demo.mp4`, `out/events.csv` and
`out/keypoints.csv` are produced and `metrics/evaluate_events.py` runs
without errors.
