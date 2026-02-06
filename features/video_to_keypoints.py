import cv2
import numpy as np
from tqdm import tqdm
from features.extract_keypoints import extract_frame_keypoints

import mediapipe as mp

def extract_video_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kp = extract_frame_keypoints(frame, holistic)
            frames.append(kp)

    cap.release()

    return np.stack(frames)  # (T_raw, 48, 2)
