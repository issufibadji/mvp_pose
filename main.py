import cv2, csv, time, numpy as np
from pathlib import Path
from pose_providers.mediapipe_pose import MediaPipePose, IDX
from features import angle, torso_len, ema
from gestures import GestureFSM
from overlay import draw_text, draw_skeleton

# -------------------- Paths / IO --------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

# -------------------- Visual/Janela -----------------
WIN_NAME = "MVP Pose – MediaPipe"
TARGET_W, TARGET_H = 960, 540   # tamanho máximo da janela

def resize_keep_aspect(img, max_w, max_h, allow_upscale=False):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if not allow_upscale:
        scale = min(scale, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# -------------------- Configurações -----------------
#VIDEO_SOURCE = "video/squat_02_fps.mp4"   # 0 = webcam; ou caminho de arquivo
#https://github.com/tensorflow/tfjs-models/raw/master/pose-detection/assets/dance_input.gif
VIDEO_SOURCE = "https://github.com/tensorflow/tfjs-models/raw/master/pose-detection/assets/dance_input.gif"
SAVE_KEYPOINTS = True
VIS_THR = 0.5

# Debug
DEBUG = True
SAVE_FAIL_FRAMES = 10      # quantos frames problemáticos salvar

# Visibilidade mínima exigida (mais tolerante p/ cartoon/silhueta)
MIN_VIS = 0.35

# Gestos (thresholds mais permissivos)
ARM   = GestureFSM(start_thr=20,  peak_thr=45,  end_thr=12,  name="raise_arm")
SQUAT = GestureFSM(start_thr=140, peak_thr=100, end_thr=130, name="squat")
SIT   = GestureFSM(start_thr=0.04, peak_thr=0.08, end_thr=0.02, name="sit_down")

# --- Gravação de vídeo (overlay leve) ---
RECORD = True                               # habilita por padrão
OUT_VIDEO = OUT_DIR / "result_demo.mp4"
OUT_FPS = 24                                # 0 = usa FPS de entrada
TRY_CODECS = ["avc1", "H264", "mp4v", "XVID"]  # tenta nessa ordem

# --- Modo cartoon: agacho com pouca deformação/queda de quadril ---
CARTOON_MODE = True #CARTOON_MODE = False

if CARTOON_MODE:
    # Braço mais sensível (cartoon costuma manter braço alto)
    ARM   = GestureFSM(start_thr=10,  peak_thr=25, end_thr=8,  name="raise_arm")
    # Squat baseado SÓ NO JOELHO (sem hip drop)
    SQUAT = GestureFSM(start_thr=155, peak_thr=145, end_thr=150, name="squat")
    # Sit desativado em cartoon
    SIT   = GestureFSM(start_thr=1.0, peak_thr=1.0, end_thr=0.9,  name="sit_down")
else:
    ARM   = GestureFSM(start_thr=20,  peak_thr=45,  end_thr=12,  name="raise_arm")
    SQUAT = GestureFSM(start_thr=140, peak_thr=100, end_thr=130, name="squat")
    SIT   = GestureFSM(start_thr=0.04, peak_thr=0.08, end_thr=0.02, name="sit_down")

def get_xy(pts, name, W, H):
    x, y, v = pts[IDX[name]]
    return np.array([x*W, y*H, v], dtype=float)

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Erro ao abrir vídeo/webcam.")
        return

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

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

    # Debug counters
    total_frames = 0
    pose_ok_frames = 0
    fail_saved = 0

    # Writer de vídeo
    writer = None
    used_codec = None
    record_enabled = RECORD  # copia local; não reatribui RECORD

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pts, _ = pose(rgb)

            def show_and_maybe_record(curr_frame):
                """Mostra e grava o frame redimensionado."""
                nonlocal writer, used_codec, record_enabled
                disp = resize_keep_aspect(curr_frame, TARGET_W, TARGET_H)
                if record_enabled and writer is None:
                    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
                    fps_out = (OUT_FPS if OUT_FPS > 0 else fps_in)
                    for cc in TRY_CODECS:
                        fourcc = cv2.VideoWriter_fourcc(*cc)
                        w = cv2.VideoWriter(str(OUT_VIDEO), fourcc, fps_out,
                                            (disp.shape[1], disp.shape[0]))
                        if w.isOpened():
                            writer, used_codec = w, cc
                            print(f"[REC] Gravando {OUT_VIDEO.name} | codec={cc} | "
                                  f"{disp.shape[1]}x{disp.shape[0]} @ {fps_out:.0f}fps")
                            break
                    if writer is None:
                        print("[REC] Falha ao abrir writer; gravação desativada.")
                        record_enabled = False
                if record_enabled and writer is not None:
                    writer.write(disp)
                cv2.imshow(WIN_NAME, disp)
                return cv2.waitKey(1)

            # Falha: sem pose
            if pts is None:
                if DEBUG and fail_saved < SAVE_FAIL_FRAMES:
                    cv2.imwrite(str(DEBUG_DIR / f"fail_no_pose_{fail_saved:02d}.jpg"), frame)
                    fail_saved += 1
                draw_text(frame, f"Arm raises: {count_arm}", 10, 30)
                draw_text(frame, f"Squats: {count_squat}", 10, 60)
                draw_text(frame, f"Sit downs: {count_sit}", 10, 90)
                if DEBUG and total_frames > 0:
                    ok_pct = 100.0 * pose_ok_frames / total_frames
                    draw_text(frame, f"Pose OK: {ok_pct:.1f}% ({pose_ok_frames}/{total_frames})", 10, 120)
                key = show_and_maybe_record(frame)
                if key in (27, ord('q')):
                    break
                continue

            H, W = frame.shape[:2]
            # Keypoints essenciais
            ls = get_xy(pts, "l_shoulder", W, H); rs = get_xy(pts, "r_shoulder", W, H)
            lh = get_xy(pts, "l_hip", W, H);      rh = get_xy(pts, "r_hip", W, H)
            le = get_xy(pts, "l_elbow", W, H);    re = get_xy(pts, "r_elbow", W, H)
            lw = get_xy(pts, "l_wrist", W, H);    rw = get_xy(pts, "r_wrist", W, H)
            lk = get_xy(pts, "l_knee", W, H);     rk = get_xy(pts, "r_knee", W, H)
            la = get_xy(pts, "l_ankle", W, H);    ra = get_xy(pts, "r_ankle", W, H)

            # Gate por visibilidade mínima
            need = [ls, rs, lh, rh, le, re, lw, rw, lk, rk, la, ra]
            if any(p[2] < MIN_VIS for p in need):
                if DEBUG and fail_saved < SAVE_FAIL_FRAMES:
                    cv2.imwrite(str(DEBUG_DIR / f"fail_low_vis_{fail_saved:02d}.jpg"), frame)
                    fail_saved += 1
                draw_text(frame, f"Arm raises: {count_arm}", 10, 30)
                draw_text(frame, f"Squats: {count_squat}", 10, 60)
                draw_text(frame, f"Sit downs: {count_sit}", 10, 90)
                if DEBUG and total_frames > 0:
                    ok_pct = 100.0 * pose_ok_frames / total_frames
                    draw_text(frame, f"Pose OK: {ok_pct:.1f}% ({pose_ok_frames}/{total_frames})", 10, 120)
                key = show_and_maybe_record(frame)
                if key in (27, ord('q')):
                    break
                continue

            # Pose válida
            pose_ok_frames += 1

            shoulder_mid = (ls[:2] + rs[:2]) / 2.0
            hip_mid = (lh[:2] + rh[:2]) / 2.0
            torso = torso_len(shoulder_mid, hip_mid)

            # --- FEATURES ---
            ang_arm_l = angle(shoulder_mid, ls[:2], le[:2])
            ang_arm_r = angle(shoulder_mid, rs[:2], re[:2])
            arm_val   = max(ang_arm_l, ang_arm_r)

            ang_knee_l = angle(lh[:2], lk[:2], la[:2])
            ang_knee_r = angle(rh[:2], rk[:2], ra[:2])
            knee_val   = min(ang_knee_l, ang_knee_r)  # menor = joelho mais flexionado

            # Baseline do quadril com suavização inicial (primeiros ~30 frames em pé)
            hip_y = hip_mid[1]
            if hip_baseline is None:
                hip_baseline = hip_y
            if total_frames < 30 and knee_val > 150:
                hip_baseline = hip_baseline*0.9 + hip_y*0.1

            # Queda do quadril normalizada pelo torso
            hip_drop_norm = max(0.0, (hip_y - hip_baseline) / (torso if torso > 0 else 1.0))
            hip_smoothed  = ema(hip_smoothed, hip_drop_norm, alpha=0.3)

            # Portas de entrada extras (mais permissivas)
            #allow_squat = (knee_val < 115) and (hip_drop_norm > 0.01)

           # Portas de entrada (gates)
            if CARTOON_MODE:
                # Squat só pelo joelho (cartoon/silhueta quase não tem queda de quadril)
                allow_squat = (knee_val < 155)
                allow_sit   = False  # desliga sit em cartoon
            else:
                allow_squat = (knee_val < 115) and (hip_drop_norm > 0.01)
                allow_sit   = (hip_drop_norm > 0.05) and (knee_val < 125)

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

            # Overlay esqueleto
            frame = draw_skeleton(frame, pts, IDX, visibility_thr=VIS_THR)

            # HUD (inclui debug ao vivo)
            draw_text(frame, f"Arm raises(Braco levantado): {count_arm}", 10, 30)
            draw_text(frame, f"Squats(Agachamento): {count_squat}", 10, 60)
            draw_text(frame, f"Sit downs(Sente-se): {count_sit}", 10, 90)
            if DEBUG and total_frames > 0:
                ok_pct = 100.0 * pose_ok_frames / total_frames
                draw_text(frame, f"Pose OK: {ok_pct:.1f}% ({pose_ok_frames}/{total_frames})", 10, 120)
                draw_text(frame, f"knee_min: {knee_val:.1f}°", 10, 150)
                draw_text(frame, f"hip_drop: {hip_drop_norm:.3f}", 10, 180)

            # Mostrar & gravar
            key = show_and_maybe_record(frame)
            if key == ord('b'):          # calibra baseline manualmente
                hip_baseline = hip_mid[1]
            if key == ord('f'):          # toggle fullscreen
                is_full = cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN) == 1
                cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN)
            if key == ord('r'):          # alterna gravação ON/OFF
                record_enabled = not record_enabled
                print(f"[REC] recording={'ON' if record_enabled else 'OFF'}")
            if key in (27, ord('q')):    # ESC ou 'q'
                break

    finally:
        events.close(); kpf.close()
        cap.release(); cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
        if DEBUG and total_frames > 0:
            ok_pct = 100.0 * pose_ok_frames / total_frames
            print(f"[DEBUG] Frames com pose: {pose_ok_frames}/{total_frames} = {ok_pct:.1f}%")
            print(f"[DEBUG] Frames-problema (até {SAVE_FAIL_FRAMES}) salvos em: {DEBUG_DIR}")

if __name__ == "__main__":
    main()
