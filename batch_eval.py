import cv2, csv, time, numpy as np
from pathlib import Path
from pose_providers.mediapipe_pose import MediaPipePose, IDX
from features import angle, torso_len, ema
from gestures import GestureFSM

DATA_DIR = Path("data")
OUT_DIR  = Path("out/batch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# thresholds (mesmos do MVP)
ARM   = GestureFSM(start_thr=20,  peak_thr=45, end_thr=12,  name="raise_arm")
SQUAT = GestureFSM(start_thr=130, peak_thr=90, end_thr=120, name="squat")
SIT   = GestureFSM(start_thr=0.05, peak_thr=0.10, end_thr=0.03, name="sit_down")
MIN_VIS = 0.6  # visibilidade mínima

def get_xy(pts, name, W, H):
    x,y,v = pts[IDX[name]]
    return np.array([x*W, y*H, v], dtype=float)

def auto_baseline_hip(hip_y_series, knee_series):
    """Baseline = 5º percentil do quadril, considerando frames 'em pé'
       (joelho esticado, p. ex. >150°). Usa primeiros ~1.5s."""
    if len(hip_y_series) == 0:
        return None
    hip = np.array(hip_y_series)
    knee = np.array(knee_series)
    mask_stand = knee > 150
    if mask_stand.any():
        hip = hip[mask_stand]
    return float(np.percentile(hip, 5))

def process_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERRO] Não abriu: {path}")
        return None

    pose = MediaPipePose()
    t0 = time.time()
    count_arm = count_squat = count_sit = 0

    ev_path = OUT_DIR / f"{path.stem}_events.csv"
    events = open(ev_path, "w", newline="", encoding="utf-8")
    ew = csv.writer(events); ew.writerow(["t","gesture"])

    # 1ª passada curta (auto baseline hip)
    hip_buf, knee_buf = [], []
    frames_for_baseline = int(1.5 * cap.get(cv2.CAP_PROP_FPS) or 45)
    for i in range(frames_for_baseline):
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts, _ = pose(rgb)
        if pts is None: continue
        H, W = frame.shape[:2]
        ls = get_xy(pts, "l_shoulder", W, H); rs = get_xy(pts, "r_shoulder", W, H)
        lh = get_xy(pts, "l_hip", W, H);      rh = get_xy(pts, "r_hip", W, H)
        lk = get_xy(pts, "l_knee", W, H);     rk = get_xy(pts, "r_knee", W, H)
        la = get_xy(pts, "l_ankle", W, H);    ra = get_xy(pts, "r_ankle", W, H)
        need = [ls, rs, lh, rh, lk, rk, la, ra]
        if any(p[2] < MIN_VIS for p in need): 
            continue
        shoulder_mid = (ls[:2] + rs[:2]) / 2.0
        hip_mid = (lh[:2] + rh[:2]) / 2.0
        knee_l = angle(lh[:2], lk[:2], la[:2])
        knee_r = angle(rh[:2], rk[:2], ra[:2])
        knee_val = min(knee_l, knee_r)
        hip_buf.append(hip_mid[1])
        knee_buf.append(knee_val)

    hip_baseline = auto_baseline_hip(hip_buf, knee_buf)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # volta ao início

    # estados
    hip_smoothed = None

    # 2ª passada: contagem
    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts, _ = pose(rgb)
        if pts is None: continue

        H, W = frame.shape[:2]
        ls = get_xy(pts, "l_shoulder", W, H); rs = get_xy(pts, "r_shoulder", W, H)
        lh = get_xy(pts, "l_hip", W, H);      rh = get_xy(pts, "r_hip", W, H)
        le = get_xy(pts, "l_elbow", W, H);    re = get_xy(pts, "r_elbow", W, H)
        lw = get_xy(pts, "l_wrist", W, H);    rw = get_xy(pts, "r_wrist", W, H)
        lk = get_xy(pts, "l_knee", W, H);     rk = get_xy(pts, "r_knee", W, H)
        la = get_xy(pts, "l_ankle", W, H);    ra = get_xy(pts, "r_ankle", W, H)

        need = [ls, rs, lh, rh, le, re, lw, rw, lk, rk, la, ra]
        if any(p[2] < MIN_VIS for p in need): 
            continue

        shoulder_mid = (ls[:2] + rs[:2]) / 2.0
        hip_mid      = (lh[:2] + rh[:2]) / 2.0
        torso        = torso_len(shoulder_mid, hip_mid)

        ang_arm_l = angle(shoulder_mid, ls[:2], le[:2])
        ang_arm_r = angle(shoulder_mid, rs[:2], re[:2])
        arm_val   = max(ang_arm_l, ang_arm_r)

        ang_knee_l = angle(lh[:2], lk[:2], la[:2])
        ang_knee_r = angle(rh[:2], rk[:2], ra[:2])
        knee_val   = min(ang_knee_l, ang_knee_r)

        if hip_baseline is None:  # fallback
            hip_baseline = hip_mid[1]
        hip_drop_norm = max(0.0, (hip_mid[1] - hip_baseline) / (torso if torso>0 else 1.0))
        hip_smoothed  = ema(hip_smoothed, hip_drop_norm, alpha=0.5)

        # gates
        allow_squat = (knee_val < 110) and (hip_drop_norm > 0.02)
        allow_sit   = (hip_drop_norm > 0.06) and (knee_val < 120)

        t = time.time() - t0
        e1 = ARM.step(arm_val, t)
        e2 = SQUAT.step(knee_val, t) if allow_squat else None
        e3 = SIT.step(hip_smoothed, t) if allow_sit else None

        if e1: count_arm   += 1; ew.writerow([e1["t"], e1["gesture"]])
        if e2: count_squat += 1; ew.writerow([e2["t"], e2["gesture"]])
        if e3: count_sit   += 1; ew.writerow([e3["t"], e3["gesture"]])

    events.close(); cap.release()
    return dict(video=path.name, arm=count_arm, squat=count_squat, sit=count_sit)

def main():
    vids = sorted([p for p in DATA_DIR.glob("*.mp4")])
    if not vids:
        print("Nenhum .mp4 em data/")
        return
    summary = []
    for v in vids:
        print(f"[PROCESSANDO] {v.name}")
        res = process_video(v)
        if res: 
            print(f"  → arm={res['arm']}  squat={res['squat']}  sit={res['sit']}")
            summary.append(res)
    # salva resumo
    if summary:
        with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video","arm","squat","sit"])
            w.writeheader(); w.writerows(summary)
        print(f"\nResumo salvo em {OUT_DIR/'summary.csv'}")

if __name__ == "__main__":
    main()
