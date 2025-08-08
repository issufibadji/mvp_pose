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


# OUTRAS 

**1) Ativar o venv no PowerShell**
- Opção mais segura (só vale para essa sessão): 

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

```
- Se quiser liberar de vez para o seu usuário:

```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# feche e reabra o PowerShell, depois:
.\.venv\Scripts\Activate.ps1
Alternativa (sem mexer na policy): usar o CMD e o .bat

```
Alternativa (sem mexer na policy): usar o CMD e o .bat
```bash
.\.venv\Scripts\activate.bat
```

**2) Instalar as dependências (faltou o OpenCV)**
-- Depois de ativar o venv:

```bash
python -m pip install -U pip
pip install opencv-python numpy
# (e o que mais seu projeto precisar, ex.: mediapipe, etc.)
```

**3) Rodar**

```bash
python main.py

```
- Dica: confira a policy atual com:
```bash
Get-ExecutionPolicy -List
```