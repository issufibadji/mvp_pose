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

.venv/Scripts/Activate.ps1
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

## Acceptance

Process a short video: `out/result_demo.mp4`, `out/events.csv` and
`out/keypoints.csv` are produced and `metrics/evaluate_events.py` runs
without errors.


Como rodar (cartoon vs real)

python main.py --source "video/armraise_04.mp4" --mode cartoon

Pessoa real (webcam):

python main.py --source 0 --mode real

Como usar

python main.py --source video.mp4 --mode cartoon --preview overlay

Avatar apenas na janela e grava os dois:
python main.py --source 0 --preview avatar --record 1 --record-avatar 1

Mostrar IDs nos pontos do avatar: --avatar-ids 1 (padrão já é 1)
