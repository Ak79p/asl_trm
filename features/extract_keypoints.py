import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

BODY_IDXS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_hip": 23,
    "right_hip": 24,
}

def extract_frame_keypoints(frame, holistic):
    h, w, _ = frame.shape
    result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    keypoints = []

    # Left hand
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            keypoints.append([lm.x, lm.y])
    else:
        keypoints.extend([[0.0, 0.0]] * 21)

    # Right hand
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            keypoints.append([lm.x, lm.y])
    else:
        keypoints.extend([[0.0, 0.0]] * 21)

    # Upper body
    if result.pose_landmarks:
        for idx in BODY_IDXS.values():
            lm = result.pose_landmarks.landmark[idx]
            keypoints.append([lm.x, lm.y])
    else:
        keypoints.extend([[0.0, 0.0]] * 6)

    return np.array(keypoints, dtype=np.float32)  # (48, 2)
