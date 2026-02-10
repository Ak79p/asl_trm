import cv2
import numpy as np
# from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks.python import vision

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ASLTRMKeypointProcessor:
    def __init__(self, pose_model_path='pose_landmarker.task', hand_model_path='hand_landmarker.task'):
        # 1. Setup Pose Landmarker
        pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            output_segmentation_masks=False)
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # 2. Setup Hand Landmarker
        hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2)
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        # Standard Connections for Plotting
        self.HAND_CONNECTIONS = [
            (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
            (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20)
        ]
        self.BODY_IDXS = {
                        "left_shoulder": 11,
                        "right_shoulder": 12,
                        "left_elbow": 13,
                        "right_elbow": 14,
                        "left_hip": 23,
                        "right_hip": 24,
                    }

    def process_frame(self, frame, frame_size = None):
        if frame_size is None:
            # frame = cv2.resize(frame, frame_size)
            h, w, _ = frame.shape
            # frame_size = (w, h)
        else:
            h, w = frame_size

        # Tasks API expects MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Inference
        pose_result = self.pose_detector.detect(mp_image)
        hand_result = self.hand_detector.detect(mp_image)

        op_pose = np.zeros((6, 2))
        op_lhand = np.zeros((21, 2))
        op_rhand = np.zeros((21, 2))

        if hand_result.hand_landmarks:
            for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                # Identify if hand is Left or Right
                lbl = hand_result.handedness[idx][0].category_name # "Left" or "Right"
                target = op_lhand if lbl == "Left" else op_rhand
                for i in range(21):
                    target[i] = [hand_lms[i].x, hand_lms[i].y]

        # --- Pose Mapping ---
        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks[0]
            # Tasks API returns a list of landmarks; we take the first person detected
            for i,idx in  enumerate(self.BODY_IDXS.values()):
                op_pose[i] = [lms[idx].x, lms[idx].y]
        

        annotated_image = self.draw_skeleton(frame.copy(), op_pose, op_lhand, op_rhand)

        keypoints = list()
        keypoints.extend(op_lhand)
        keypoints.extend(op_rhand)
        keypoints.extend(op_pose)

        keypoints = np.array(keypoints, dtype=np.float32)  # (48, 2)
        # Draw for Visualization
        annotated_image = self.draw_skeleton(frame.copy(), op_pose, op_lhand, op_rhand)
        
        return annotated_image, keypoints

    

    def draw_skeleton(self, img, pose, lhand, rhand):
      # 1. Draw Hands (Green circles, White lines)
        for hand in [lhand, rhand]:
            for pts in hand:
            #   if pts[2] > 0: 
                cv2.circle(img, (int(pts[0]), int(pts[1])), 3, (0, 255, 0), -1)
            for connection in self.HAND_CONNECTIONS:
                p1, p2 = hand[connection[0]], hand[connection[1]]
                # if p1[2] > 0 and p2[2] > 0:
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255), 1)
        
        # 2. Define Pose Edges for 6 keypoints
        # Indices: 0=left_shoulder, 1=right_shoulder, 2=left_elbow, 3=right_elbow, 4=left_hip, 5=right_hip
        upper_body_edges = [
            (0, 2),  # left_shoulder to left_elbow
            (2, 4),  # left_elbow to left_hip
            (0, 4),  # left_shoulder to left_hip
            (1, 3),  # right_shoulder to right_elbow
            (3, 5),  # right_elbow to right_hip
            (1, 5),  # right_shoulder to right_hip
            (0, 1),  # left_shoulder to right_shoulder (connects shoulders)
        ]
        
        all_pose_edges = upper_body_edges
        
        # 3. Draw Pose (Red lines and Blue joints for the head)
        for edge in all_pose_edges:
            p1, p2 = pose[edge[0]], pose[edge[1]]
            
            # Check confidence/presence threshold (0.3 is a good balance)
            #   if p1[2] > 0.3 and p2[2] > 0.3:
            # Draw the connection line
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
            
            # Optional: Draw specific joints for the face in a different color
            if edge[1] >= 15: # If it's an eye or ear
                cv2.circle(img, (int(p2[0]), int(p2[1])), 3, (255, 0, 0), -1)

        return img



class ASLTRMKeypointProcessorOLD:
    def __init__(self):
        self.keypoints_extractor = vision.holistic.Holistic(
                                    static_image_mode=False,
                                    model_complexity=1,
                                    enable_segmentation=False,
                                    refine_face_landmarks=False,
                                )   
        
        self.BODY_IDXS = {
                        "left_shoulder": 11,
                        "right_shoulder": 12,
                        "left_elbow": 13,
                        "right_elbow": 14,
                        "left_hip": 23,
                        "right_hip": 24,
                    }
        
    def process_frame(self, frame):
        # h, w, _ = frame.shape
        result = self.keypoints_extractor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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
            for idx in self.BODY_IDXS.values():
                lm = result.pose_landmarks.landmark[idx]
                keypoints.append([lm.x, lm.y])
        else:
            keypoints.extend([[0.0, 0.0]] * 6)

        return np.array(keypoints, dtype=np.float32)  # (48, 2)


