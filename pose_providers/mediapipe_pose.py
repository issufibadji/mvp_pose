import mediapipe as mp

IDX = {
  "nose":0, "l_shoulder":11, "r_shoulder":12, "l_elbow":13, "r_elbow":14,
  "l_wrist":15, "r_wrist":16, "l_hip":23, "r_hip":24, "l_knee":25, "r_knee":26,
  "l_ankle":27, "r_ankle":28, "left_eye":2, "right_eye":5, "left_ear":7, "right_ear":8
}

class MediaPipePose:
    def __init__(self, static=False, model_complexity=1, det_conf=0.5, track_conf=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static,
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )

    def __call__(self, frame_rgb):
        res = self.pose.process(frame_rgb)
        if not res.pose_landmarks: 
            return None, 0.0
        pts = [(lmk.x, lmk.y, lmk.visibility) for lmk in res.pose_landmarks.landmark]
        return pts, 1.0
