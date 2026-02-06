# ASL Sign Language Recognition

This project provides a complete pipeline for American Sign Language (ASL) sign recognition from video, including keypoint extraction, Pose-TGCN and TRN-ASL inference, and an interactive Streamlit web app for visualization and analysis.

## Demo Video

You can preview a sample ASL video used for testing in the app:

[![Sample ASL Video](assets/sample_video.gif)](assets/sample_video.mp4)

> The video is located in the `assets/` folder as `sample_video.mp4`. You can use it to test the pipeline or as an example upload in the Streamlit app.


## Features
- **Video Upload & Processing**: Upload ASL videos and process them in-browser.
- **Keypoint Extraction**: Uses MediaPipe to extract pose and hand keypoints.
- **TRM-ASL / Pose-TGCN Inference**: Recognizes 100 ASL signs using a Pose-TGCN / TRM-ASL model.
- **Sliding Window**: Configurable window size and overlap for robust recognition.
- **Frame Skipping**: Option to process every Nth frame for speed.
- **Frame Cropping**: Crop left/right edges by X% before processing.
- **Keypoint Visualization**: See extracted keypoints overlaid on sample frames.
- **Confidence Scores**: View model confidence for each prediction.
- **Export Results**: Download recognition results as JSON.

## Quick Start

### 1. Install dependencies
```powershell
uv pip install .
```

### 2. Run the Streamlit app
```powershell
uv run streamlit run app.py
```

### 3. Using the App
- Upload a video file (mp4, avi, mov, mkv)
- Adjust window size, overlap, frame skip, and crop in the sidebar
- Click **Start Processing**
- View sample frames with keypoints, recognition results, and download options

## Project Structure
- `app.py` — Streamlit web app
- `main.py` — CLI entry point
- `wlasl/` — Core modules:
  - `mp_keypoints_processor.py` — MediaPipe keypoint extraction
  - `pose_tgcn_model.py` — Pose-TGCN inference
  - `configs/label_map_100.json` — ASL label mapping
  - `checkpoints/` — Model and MediaPipe checkpoints

## Model
- **Architecture**: Pose-TGCN (Graph Convolutional Network with attention)
- **Classes**: 100 ASL signs
- **Input**: Pose (25 keypoints) + hands (21 each)
- **Output**: Predicted sign + confidence

## Configuration Options
- **Window Size**: Number of frames per prediction window
- **Overlap**: Percentage overlap between windows (0–90%)
- **Frame Skip**: Process every Nth frame (0 = all)
- **Crop**: Crop X% from left/right before keypoint extraction

## Requirements
- Python 3.8+
- See `pyproject.toml` for dependencies (mediapipe, torch, streamlit, opencv-python, numpy, etc.)

## License
MIT



---
For questions or issues, please contact the project maintainer.

