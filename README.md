# MVP – Reconhecimento de Gestos (MediaPipe + OpenCV)

**Gestos incluídos (prontos):**
- Levantar braço (D/E, conta o maior)
- Agachamento (joelho, conta menores ângulos)
- Sentar (queda de quadril + dobra de joelho) – *experimental*

**Como rodar (webcam):**
```bash
pip install -r requirements.txt
python main.py
```

**Como rodar em vídeo:**
Edite `VIDEO_SOURCE` em `main.py` para o caminho de arquivo (ex.: `"data/video.mp4"`).

**Saídas:**
- Overlay com esqueleto/contadores e FPS
- `out/events.csv` – log de eventos (tempo e gesto)
- `out/keypoints.csv` – (opcional) keypoints normalizados e ângulos por frame

**Observações:**
- Este MVP usa **MediaPipe Pose** (BlazePose), sem treinar nada.
- O código já é modular para futuramente plugar **RTMPose/MMPose** em `pose_providers/`.
