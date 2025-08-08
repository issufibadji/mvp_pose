import cv2
import numpy as np

def draw_text(img, txt, x=10, y=30):
    cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def draw_skeleton(frame, pts, idx_map, visibility_thr=0.5, color=(0,255,0)):
    if pts is None: 
        return frame
    H, W = frame.shape[:2]
    def XY(i):
        x,y,v = pts[i]
        return int(x*W), int(y*H), v
    # Conexões principais (MediaPipe style reduzido)
    edges = [
        ("l_shoulder","r_shoulder"), ("l_shoulder","l_elbow"), ("l_elbow","l_wrist"),
        ("r_shoulder","r_elbow"), ("r_elbow","r_wrist"),
        ("l_shoulder","l_hip"), ("r_shoulder","r_hip"),
        ("l_hip","r_hip"),
        ("l_hip","l_knee"), ("l_knee","l_ankle"),
        ("r_hip","r_knee"), ("r_knee","r_ankle")
    ]
    for a,b in edges:
        ia, ib = idx_map[a], idx_map[b]
        xa,ya,va = XY(ia); xb,yb,vb = XY(ib)
        if va>visibility_thr and vb>visibility_thr:
            cv2.line(frame, (xa,ya), (xb,yb), color, 2)
    return frame
