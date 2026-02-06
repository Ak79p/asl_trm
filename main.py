from wlasl import PoseTGCNInference
from wlasl import ASLKeypointProcessor
# from .keypoint_storage import KeypointStorage
import cv2

def main():
    # Initialization
    NUM_SAMPLES = 50
    # Configuration
    WINDOW_SIZE = NUM_SAMPLES
    OVERLAP_PERCENT = 0.20
    STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENT)) # 40 frames

    POSE_CHECKPOINTS_PATH = "wlasl/checkpoints/pose_landmarker.task"
    HAND_CHECKPOINTS_PATH = "wlasl/checkpoints/hand_landmarker.task"

    RECOGNIZER_MODEL_PATH = "wlasl/checkpoints/pose_tgcn_asl100.pth"
    LABELS_PATH = "wlasl/configs/label_map_100.json"
    VIDEO_PATH = r"D:\self\projects\masters\capstone\data\wlasl\sample\69345_preprocessed.mp4"  # Replace with your video path

    # Initialize Processor and Inference Engine
    processor = ASLKeypointProcessor(POSE_CHECKPOINTS_PATH, HAND_CHECKPOINTS_PATH)
    engine = PoseTGCNInference(RECOGNIZER_MODEL_PATH, LABELS_PATH, NUM_SAMPLES)
    buffer = []
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame_count += 1
        
        # 1. Process frame (ensure target_size is consistent with model training)
        # Using the 256 scale required by your normalization logic (x / 256.0)
        _, keypoints = processor.process_frame(frame, frame_size = (256,256)) 
        
        buffer.append(keypoints)
        
        # 2. Check if the window is full
        if len(buffer) == WINDOW_SIZE:
            # Perform Inference
            prediction = engine.predict(buffer)
            
            # Display timestamp or frame range for clarity
            start_f = frame_count - WINDOW_SIZE
            print(f"Frames {start_f}-{frame_count} | Detected Sign: {prediction}")
            
            # 3. Slide the window: Remove 'STRIDE' number of frames
            # This leaves 'OVERLAP' frames (10) in the buffer for the next window
            buffer = buffer[STRIDE:]

    cap.release()

if __name__ == "__main__":
    main()