import cv2, csv, time, numpy as np
from pathlib import Path
from pose_providers.mediapipe_pose import MediaPipePose, IDX
from features import angle, torso_len, ema
from gestures import GestureFSM
from overlay import draw_text, draw_skeleton

# ---- Paths / IO ----
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

# ---- Configurações ----
VIDEO_SOURCE = 'data/sitdown_01_fps.mp4'   # 0 = webcam; ou caminho 'data/video.mp4'
SAVE_KEYPOINTS = True
VIS_THR = 0.5

# Visibilidade mínima exigida para pontos-chave (corta falsos positivos)
MIN_VIS = 0.6

# Gestos (thresholds de exemplo; ajuste nos seus vídeos)
ARM = GestureFSM(start_thr=20, peak_thr=45, end_thr=12, name="raise_arm")  # ângulo no ombro
SQUAT = GestureFSM(start_thr=130, peak_thr=90, end_thr=120, name="squat")   # ângulo joelho
# Sentar: usa queda do quadril normalizada + dobra do joelho (experimental)
SIT = GestureFSM(start_thr=0.05, peak_thr=0.10, end_thr=0.03, name="sit_down")  # delta_y_hip_norm

def get_xy(pts, name, W, H):
    x, y, v = pts[IDX[name]]
    return np.array([x*W, y*H, v], dtype=float)

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Erro ao abrir vídeo/webcam."); 
        return

    pose = MediaPipePose()
    t0 = time.time()
    count_arm = count_squat = count_sit = 0

    # Logs
    events = open(OUT_DIR / "events.csv", "w", newline="", encoding="utf-8")
    ew = csv.writer(events); ew.writerow(["t","gesture"])

    kpf = open(OUT_DIR / "keypoints.csv", "w", newline="", encoding="utf-8")
    kw = csv.writer(kpf)
    kw.writerow([
        "t","ls.x","ls.y","rs.x","rs.y","lh.x","lh.y","rh.x","rh.y","le.x","le.y","re.x","re.y",
        "lw.x","lw.y","rw.x","rw.y","lk.x","lk.y","rk.x","rk.y","la.x","la.y","ra.x","ra.y",
        "ang_arm_l","ang_arm_r","ang_knee_l","ang_knee_r","hip_drop_norm"
    ])

    # Estados locais
    hip_baseline = None
    hip_smoothed = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pts, _ = pose(rgb)

            if pts is not None:
                H, W = frame.shape[:2]
                # Keypoints essenciais
                ls = get_xy(pts, "l_shoulder", W, H); rs = get_xy(pts, "r_shoulder", W, H)
                lh = get_xy(pts, "l_hip", W, H);      rh = get_xy(pts, "r_hip", W, H)
                le = get_xy(pts, "l_elbow", W, H);    re = get_xy(pts, "r_elbow", W, H)
                lw = get_xy(pts, "l_wrist", W, H);    rw = get_xy(pts, "r_wrist", W, H)
                lk = get_xy(pts, "l_knee", W, H);     rk = get_xy(pts, "r_knee", W, H)
                la = get_xy(pts, "l_ankle", W, H);    ra = get_xy(pts, "r_ankle", W, H)

                # Gate por visibilidade mínima (evita falsos positivos)
                need = [ls, rs, lh, rh, le, re, lw, rw, lk, rk, la, ra]
                if any(p[2] < MIN_VIS for p in need):
                    draw_text(frame, f"Arm raises: {count_arm}", 10, 30)
                    draw_text(frame, f"Squats: {count_squat}", 10, 60)
                    draw_text(frame, f"Sit downs: {count_sit}", 10, 90)
                    cv2.imshow("MVP Pose – MediaPipe", frame)
                    if cv2.waitKey(1) in (27, ord('q')):
                        break
                    continue

                shoulder_mid = (ls[:2] + rs[:2]) / 2.0
                hip_mid = (lh[:2] + rh[:2]) / 2.0
                torso = torso_len(shoulder_mid, hip_mid)

                # --- FEATURES ---
                ang_arm_l = angle(shoulder_mid, ls[:2], le[:2])
                ang_arm_r = angle(shoulder_mid, rs[:2], re[:2])
                arm_val   = max(ang_arm_l, ang_arm_r)  # usa o maior ângulo

                ang_knee_l = angle(lh[:2], lk[:2], la[:2])
                ang_knee_r = angle(rh[:2], rk[:2], ra[:2])
                knee_val   = min(ang_knee_l, ang_knee_r)  # agacho reduz ângulo

                # Sentar: queda do quadril em relação à baseline, normalizada pelo torso
                hip_y = hip_mid[1]
                if hip_baseline is None:
                    hip_baseline = hip_y  # inicializa na postura inicial
                hip_drop_norm = max(0.0, (hip_y - hip_baseline) / (torso if torso > 0 else 1.0))
                hip_smoothed  = ema(hip_smoothed, hip_drop_norm, alpha=0.5)

                # Portas de entrada extras (gating) para reduzir falsos positivos
                allow_squat = (knee_val < 110) and (hip_drop_norm > 0.02)
                allow_sit   = (hip_drop_norm > 0.06) and (knee_val < 120)

                # FSMs
                t = time.time() - t0
                evt1 = ARM.step(arm_val, t)
                evt2 = SQUAT.step(knee_val, t) if allow_squat else None
                evt3 = SIT.step(hip_smoothed, t) if allow_sit else None

                if evt1: count_arm   += 1; ew.writerow([evt1["t"], evt1["gesture"]])
                if evt2: count_squat += 1; ew.writerow([evt2["t"], evt2["gesture"]])
                if evt3: count_sit   += 1; ew.writerow([evt3["t"], evt3["gesture"]])

                # salvar keypoints/angles (opcional)
                if SAVE_KEYPOINTS:
                    kw.writerow([
                        t,
                        ls[0],ls[1], rs[0],rs[1], lh[0],lh[1], rh[0],rh[1],
                        le[0],le[1], re[0],re[1], lw[0],lw[1], rw[0],rw[1],
                        lk[0],lk[1], rk[0],rk[1], la[0],la[1], ra[0],ra[1],
                        ang_arm_l, ang_arm_r, ang_knee_l, ang_knee_r, hip_smoothed
                    ])

                # Overlay
                frame = draw_skeleton(frame, pts, IDX, visibility_thr=VIS_THR)

            # HUD
            draw_text(frame, f"Arm raises: {count_arm}", 10, 30)
            draw_text(frame, f"Squats: {count_squat}", 10, 60)
            draw_text(frame, f"Sit downs: {count_sit}", 10, 90)

            cv2.imshow("MVP Pose – MediaPipe", frame)
            key = cv2.waitKey(1)
            if key == ord('b') and pts is not None:  # 'b' = baseline
                hip_baseline = hip_mid[1]
            if key in (27, ord('q')):  # ESC ou q
                break

    finally:
        events.close(); kpf.close()
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
