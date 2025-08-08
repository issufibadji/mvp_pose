import numpy as np

def angle(a,b,c):
    a,b,c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a-b, c-b
    den = (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
    cos = np.dot(v1,v2) / den
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def torso_len(shoulder_mid, hip_mid):
    return float(np.linalg.norm(np.array(shoulder_mid)-np.array(hip_mid)) + 1e-9)

def ema(prev, x, alpha=0.6):
    return alpha*x + (1.0-alpha)*prev if prev is not None else x
