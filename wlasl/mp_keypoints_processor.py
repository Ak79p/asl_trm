import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ASLKeypointProcessor:
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
        
        # Map to OpenPose Format
        op_pose, op_lhand, op_rhand = self._map_to_openpose(pose_result, hand_result, w, h)
        
        # Draw for Visualization
        annotated_image = self.draw_skeleton(frame.copy(), op_pose, op_lhand, op_rhand)
        
        return annotated_image, {"pose": op_pose, "left_hand": op_lhand, "right_hand": op_rhand}

    def _map_to_openpose(self, pose_res, hand_res, w, h):
        op_pose = np.zeros((25, 3))
        op_lhand = np.zeros((21, 3))
        op_rhand = np.zeros((21, 3))

        # --- Pose Mapping ---
        if pose_res.pose_landmarks:
            # Tasks API returns a list of landmarks; we take the first person detected
            lms = pose_res.pose_landmarks[0]
            mapping = {0:0, 12:2, 14:3, 16:4, 11:5, 13:6, 15:7, 24:9, 26:10, 28:11, 23:12, 25:13, 27:14, 5:15, 2:16, 8:17, 7:18}
            for mp_idx, op_idx in mapping.items():
                op_pose[op_idx] = [lms[mp_idx].x * w, lms[mp_idx].y * h, lms[mp_idx].presence]
            
            # Virtual Joints (Neck and Mid-Hip)
            op_pose[1] = (op_pose[2] + op_pose[5]) / 2 
            op_pose[8] = (op_pose[9] + op_pose[12]) / 2

        # --- Hand Mapping ---
        if hand_res.hand_landmarks:
            for idx, hand_lms in enumerate(hand_res.hand_landmarks):
                # Identify if hand is Left or Right
                lbl = hand_res.handedness[idx][0].category_name # "Left" or "Right"
                target = op_lhand if lbl == "Left" else op_rhand
                for i in range(21):
                    target[i] = [hand_lms[i].x * w, hand_lms[i].y * h, 1.0]

        return op_pose, op_lhand, op_rhand

    def draw_skeleton(self, img, pose, lhand, rhand):
      # 1. Draw Hands (Green circles, White lines)
      for hand in [lhand, rhand]:
          for pts in hand:
              if pts[2] > 0: 
                  cv2.circle(img, (int(pts[0]), int(pts[1])), 3, (0, 255, 0), -1)
          for connection in self.HAND_CONNECTIONS:
              p1, p2 = hand[connection[0]], hand[connection[1]]
              if p1[2] > 0 and p2[2] > 0:
                  cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255), 1)
      
      # 2. Define Pose Edges
      # Upper Body: Neck to Shoulders/Arms and Neck to Nose
      upper_body_edges = [(1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,0)]
      
      # Face: Nose to Eyes, and Eyes to Ears
      # OpenPose BODY_25 indices: Nose(0), REye(15), LEye(16), REar(17), LEar(18)
      face_edges = [
          (0, 15), # Nose to Right Eye
          (0, 16), # Nose to Left Eye
          (15, 17), # Right Eye to Right Ear
          (16, 18)  # Left Eye to Left Ear
      ]
      
      all_pose_edges = upper_body_edges + face_edges
      
      # 3. Draw Pose (Red lines and Blue joints for the head)
      for edge in all_pose_edges:
          p1, p2 = pose[edge[0]], pose[edge[1]]
          
          # Check confidence/presence threshold (0.3 is a good balance)
          if p1[2] > 0.3 and p2[2] > 0.3:
              # Draw the connection line
              cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
              
              # Optional: Draw specific joints for the face in a different color
              if edge[1] >= 15: # If it's an eye or ear
                  cv2.circle(img, (int(p2[0]), int(p2[1])), 3, (255, 0, 0), -1)

      return img