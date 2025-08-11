from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from features import angles, hip_drop_norm, hip_y, torso_len, ema
from gestures import ArmRaiseFSM, SquatFSM, SitDownFSM
from overlay import draw_hud, draw_skeleton
from pose_providers.mediapipe_pose import KP_INDEX, MediaPipePose
from utils_io import event_writer, keypoint_writer, open_video_capture, open_video_writer, read_rgb

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'out'
DEBUG_DIR = OUT_DIR / 'debug'
OUT_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description='Simple pose gesture demo.')
    ap.add_argument('--source', default=0, help='0 for webcam or path to video')
    ap.add_argument('--record', type=int, default=1, help='save result_demo.mp4')
    ap.add_argument('--mode', choices=['real', 'cartoon'], default='real')
    ap.add_argument('--min-vis', type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()

    cap = open_video_capture(0 if str(args.source) == '0' else args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    provider = MediaPipePose()
    provider.start()

    # Gesture FSMs
    arm = ArmRaiseFSM()
    squat = SquatFSM()
    sit = SitDownFSM()

    counts = {'arm_raise': 0, 'squat': 0, 'sit_down': 0}

    # CSV writers
    ev_f, ev_w = event_writer(OUT_DIR / 'events.csv')
    kp_f, kp_w = keypoint_writer(OUT_DIR / 'keypoints.csv')

    # Video writer (lazy)
    writer = None
    used_codec = None

    hip_baseline = None
    hip_ys = []
    hip_drop_s = None
    ang_s = {'arm_l': None, 'arm_r': None, 'knee_l': None, 'knee_r': None}

    pose_frames = 0
    total_frames = 0
    last_fail_save = 0.0

    t0 = time.time()
    try:
        while True:
            ok, rgb = read_rgb(cap)
            if not ok:
                break
            total_frames += 1
            (xy, vis), pose_found = provider.infer(rgb)
            pose_ok = False
            if pose_found:
                needed = [
                    KP_INDEX['l_shoulder'], KP_INDEX['r_shoulder'],
                    KP_INDEX['l_wrist'], KP_INDEX['r_wrist'],
                    KP_INDEX['l_hip'], KP_INDEX['r_hip'],
                    KP_INDEX['l_knee'], KP_INDEX['r_knee'],
                    KP_INDEX['l_ankle'], KP_INDEX['r_ankle'],
                ]
                vis_ok = vis[needed] >= args.min_vis
                pose_ok = vis_ok.sum() >= 12
            t = time.time() - t0
            if pose_ok:
                pose_frames += 1
                ang = angles(xy)
                for k in ang:
                    ang_s[k] = ema(ang_s[k], ang[k])
                torso = torso_len(xy)
                h_y = hip_y(xy)
                hip_baseline = h_y if hip_baseline is None else hip_baseline
                hip_drop = hip_drop_norm(h_y, hip_baseline, torso)
                hip_drop_s = ema(hip_drop_s, hip_drop)
                hip_ys.append(h_y)
                hip_ys = hip_ys[-12:]
                knee_min = min(ang_s['knee_l'], ang_s['knee_r'])
                evt1 = arm.step(xy, t)
                evt2 = squat.step(knee_min, hip_drop_s, t)
                evt3 = sit.step(ang_s['knee_l'], ang_s['knee_r'], hip_ys, hip_drop_s, t)
                for evt in (evt1, evt2, evt3):
                    if evt:
                        counts[evt.name] += 1
                        ev_w.writerow([evt.t, evt.name])
            else:
                if t - last_fail_save > 1.0:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(DEBUG_DIR / f'fail_{total_frames:06d}.jpg'), bgr)
                    last_fail_save = t
                hip_drop_s = hip_drop_s  # keep last value
                knee_min = ang_s['knee_l'] if ang_s['knee_l'] is not None else 180.0

            # log keypoints
            row = [t, int(pose_ok)]
            if pose_found:
                for p, v in zip(xy, vis):
                    row.extend([f'{p[0]:.5f}', f'{p[1]:.5f}', f'{v:.5f}'])
            else:
                row.extend(['0.0', '0.0', '0.0'] * len(KP_INDEX))
            kp_w.writerow(row)

            # prepare BGR frame for overlay
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if pose_found:
                draw_skeleton(bgr, xy, vis)
            ok_pct = 100.0 * pose_frames / total_frames
            draw_hud(bgr, counts, ok_pct, knee_min, hip_drop_s or 0.0)

            if args.record and writer is None:
                h, w = bgr.shape[:2]
                writer, used_codec = open_video_writer(OUT_DIR / 'result_demo.mp4', (w, h), fps_in)
            if writer:
                writer.write(bgr)

            cv2.imshow('mvp_pose', bgr)
            key = cv2.waitKey(1)
            if key in (27, ord('q')):
                break
            if key == ord('b'):
                hip_baseline = h_y if pose_ok else hip_baseline
    finally:
        cap.release()
        provider.stop()
        cv2.destroyAllWindows()
        ev_f.close()
        kp_f.close()
        if writer:
            writer.release()
        if total_frames:
            print(f'Pose OK: {pose_frames/total_frames*100:.1f}% ({pose_frames}/{total_frames})')
            if used_codec:
                print(f'Video saved with codec {used_codec}')


if __name__ == '__main__':
    main()
