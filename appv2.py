"""
Streamlit App for ASL Sign Language Recognition
Visualizes the complete pipeline: video upload → keypoint extraction → word prediction
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import mediapipe as mp
from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig
from features.extract_keypoints import extract_frame_keypoints

# Page configuration
st.set_page_config(
    page_title="ASL Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    }
    .prediction-word {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 10px 0;
    }
    .confidence {
        font-size: 1.5rem;
        color: #666;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def load_models(dataset_name="asl100", checkpoint_path="checkpoints/asl100/best_model.pt"):
    """Load the ASL keypoint processor and TRM inference model."""
    with st.spinner("Loading TRM models..."):
        try:
            dataset_cfg = DatasetConfig(dataset_name)
            idx_to_gloss = {}
            for gloss, idx in dataset_cfg.label_map.items():
                idx_to_gloss[idx] = gloss
                try:
                    idx_to_gloss[int(idx)] = gloss
                except (TypeError, ValueError):
                    pass

            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )

            model = TRMMicro(num_classes=dataset_cfg.num_classes).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()

            return model, idx_to_gloss, device
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, None


def build_trm_feature_tensor_from_window(kps_window, target_t=48):
    """Build TRM-ready feature tensor (T,48,8) from raw keypoints (T_raw,48,2)."""
    if kps_window.ndim != 3 or kps_window.shape[1:] != (48, 2):
        raise ValueError(f"Invalid keypoint window shape: {kps_window.shape}")

    indices = np.linspace(0, len(kps_window) - 1, target_t).astype(int)
    kps = kps_window[indices]

    vel = np.zeros_like(kps)
    vel[1:] = kps[1:] - kps[:-1]

    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]

    lh_center = kps[:, 0:21].mean(axis=1, keepdims=True)
    rh_center = kps[:, 21:42].mean(axis=1, keepdims=True)

    rel = kps.copy()
    rel[:, 0:21] -= lh_center
    rel[:, 21:42] -= rh_center

    return np.concatenate([kps, vel, acc, rel], axis=-1).astype(np.float32)


def decode_gloss(idx_to_gloss, class_idx):
    """Decode class index to gloss with int/string fallback."""
    if class_idx in idx_to_gloss:
        return idx_to_gloss[class_idx]

    class_idx_str = str(class_idx)
    if class_idx_str in idx_to_gloss:
        return idx_to_gloss[class_idx_str]

    return f"CLASS_{class_idx}"


def process_video(video_path, model, idx_to_gloss, device, window_size=50, overlap_percent=0.20, num_samples=50, frame_skip=1, crop_lr_percent=0):
    """Process video and extract predictions with visualization data.
    
    Zero-pads the buffer if total frames are less than num_samples required.
    """
    stride = int(window_size * (1 - overlap_percent))
    cap = cv2.VideoCapture(str(video_path))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    buffer = []
    predictions = []
    frame_count = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Sample frames for visualization
    sample_frames = []
    sample_interval = max(1, total_frames // 10)
    
    processed_frame_count = 0
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Update progress
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)
            skip_display = "none" if frame_skip == 0 else str(frame_skip)
            status_text.text(f"Processing frame {frame_count}/{total_frames} (skip: {skip_display})")

            # Skip frames if frame_skip > 0
            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                continue

            processed_frame_count += 1

            # Crop frame from left/right if requested
            if crop_lr_percent > 0:
                h, w = frame.shape[:2]
                crop_px = int(w * (crop_lr_percent / 100.0))
                if w - (2 * crop_px) > 0:
                    frame = frame[:, crop_px:w - crop_px]

            keypoints = extract_frame_keypoints(frame, holistic)
            buffer.append(keypoints)

            # Save sample frames for visualization
            if processed_frame_count % sample_interval == 0 and len(sample_frames) < 10:
                sample_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Check if window is full
            if len(buffer) == window_size:
                valid_window = [frame_kp for frame_kp in buffer if frame_kp is not None]

                try:
                    if len(valid_window) == 0:
                        raise ValueError("No valid frames in window")

                    raw_window = np.stack(valid_window, axis=0)
                    feature_window = build_trm_feature_tensor_from_window(raw_window, target_t=48)
                    input_tensor = torch.from_numpy(feature_window).float().unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(input_tensor).squeeze(0)
                        probs = F.softmax(logits, dim=0)
                        top_values, top_indices = probs.topk(min(5, probs.shape[0]))

                    top5_predictions = [
                        {
                            "word": decode_gloss(idx_to_gloss, int(idx.item())),
                            "confidence": float(score.item() * 100)
                        }
                        for idx, score in zip(top_indices, top_values)
                    ]

                    prediction = top5_predictions[0]["word"]
                    confidence = top5_predictions[0]["confidence"]
                except Exception:
                    prediction = "UNKNOWN"
                    confidence = None
                    top5_predictions = []

                start_frame = frame_count - window_size
                timestamp = start_frame / fps if fps > 0 else 0
                predictions.append({
                    'frame_range': (start_frame, frame_count),
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'confidence': confidence,
                    'top5': top5_predictions,
                    'frame_count': frame_count
                })
                # Slide window
                buffer = buffer[stride:]
    
    cap.release()
    
    # Handle case where video has fewer frames than num_samples
    if buffer and len(buffer) < num_samples:
        status_text.text(f"Final window has {len(buffer)} frames. Zero-padding to {num_samples} frames...")

        valid_buffer = [frame_kp for frame_kp in buffer if frame_kp is not None]
        if valid_buffer:
            try:
                raw_window = np.stack(valid_buffer, axis=0)
                feature_window = build_trm_feature_tensor_from_window(raw_window, target_t=48)
                input_tensor = torch.from_numpy(feature_window).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor).squeeze(0)
                    probs = F.softmax(logits, dim=0)
                    top_values, top_indices = probs.topk(min(5, probs.shape[0]))

                top5_predictions = [
                    {
                        "word": decode_gloss(idx_to_gloss, int(idx.item())),
                        "confidence": float(score.item() * 100)
                    }
                    for idx, score in zip(top_indices, top_values)
                ]

                prediction = top5_predictions[0]["word"]
                confidence = top5_predictions[0]["confidence"]
            except Exception:
                prediction = "UNKNOWN"
                confidence = None
                top5_predictions = []

            num_padding = max(0, num_samples - len(buffer))
            predictions.append({
                'frame_range': (frame_count - len(buffer), frame_count),
                'timestamp': frame_count / fps if fps > 0 else 0,
                'prediction': prediction,
                'confidence': confidence,
                'top5': top5_predictions,
                'frame_count': frame_count,
                'padded': True,
                'padding_frames': num_padding
            })

            actual_frames = len(valid_buffer)
            st.info(f"ℹ️ Processed {processed_frame_count} frames total. Final window: {actual_frames} frames, padded with {num_padding} zero frames to reach {num_samples} frames for prediction.")
    
    progress_bar.empty()
    status_text.empty()
    
    return predictions, sample_frames, total_frames, fps


def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Sign Language Recognition</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        dataset_name = st.selectbox(
            "Vocabulary",
            ["app", "asl100", "asl300", "asl1000", "asl2000"],
            index=1
        )

        checkpoint_path = st.text_input(
            "Checkpoint Path",
            f"checkpoints/{dataset_name}/best_model.pt"
        )
        
        st.markdown("---")
        
        window_size = st.slider("Window Size (frames)", 20, 100, 50, 5,
                                help="Number of frames to analyze for each prediction")
        
        overlap_percent = st.slider("Overlap Percentage", 0.0, 0.90, 0.2, 0.05,
                                    help="Percentage of frames to overlap between windows")
        
        frame_skip = st.slider("Frame Skip (every Nth frame)", 0, 10, 0, 1,
                      help="Process every Nth frame for faster processing. 0 = process all frames, 1 = skip 1 frame, 2 = skip 2 frames")
        crop_lr_percent = st.slider("Crop Left/Right (%)", 0, 40, 0, 1,
                        help="Crop X%% from left and right edges before keypoint processing")
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("""
        **Model:** TRM (Transformer)  
        **Architecture:** Transformer-based architecture  
        **Input:** Body pose + hand keypoints  
        **Window:** Sliding window approach
        """)
    
    # Load models (cached with model type)
    model_key = f"trm_engine::{dataset_name}::{checkpoint_path}"
    labels_key = f"trm_labels::{dataset_name}"
    device_key = f"trm_device::{dataset_name}::{checkpoint_path}"
    
    if model_key not in st.session_state:
        model, labels, device = load_models(dataset_name, checkpoint_path)
        if model:
            st.session_state[model_key] = model
            st.session_state[labels_key] = labels
            st.session_state[device_key] = device
            st.success("✅ TRM models loaded successfully!")
        else:
            st.error("❌ Failed to load models. Please check configuration.")
            return
    else:
        model = st.session_state[model_key]
        labels = st.session_state.get(labels_key, {})
        device = st.session_state.get(device_key, "cpu")
    
    # Video upload
    st.markdown('<h2 class="step-header">📹 Step 1: Upload Video</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video containing ASL sign language gestures"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display video
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(uploaded_file)
        
        with col2:
            st.markdown("### 📝 Video Info")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            st.metric("Duration", f"{duration:.2f}s")
            st.metric("Total Frames", f"{total_frames}")
            st.metric("FPS", f"{fps:.1f}")
            st.metric("Resolution", f"{width}x{height}")
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            st.markdown('<h2 class="step-header">⚡ Step 2: Processing Video</h2>', unsafe_allow_html=True)
            
            # Process video with NUM_SAMPLES for zero-padding
            predictions, sample_frames, total_frames, fps = process_video(
                video_path,
                model,
                labels,
                device,
                window_size,
                overlap_percent,
                num_samples=50,
                frame_skip=frame_skip,
                crop_lr_percent=crop_lr_percent
            )
            
            if predictions:
                st.success(f"✅ Processing complete! Found {len(predictions)} predictions.")
                
                # Display sample frames
                st.markdown('<h2 class="step-header">🖼️ Step 3: Keypoint Extraction</h2>', unsafe_allow_html=True)
                st.markdown("**Sample frames from the video:**")
                
                cols = st.columns(5)
                for idx, frame in enumerate(sample_frames[:10]):
                    with cols[idx % 5]:
                        st.image(frame, use_container_width=True, caption=f"Frame {idx+1}")
                
                st.markdown("---")
                
                # Display predictions
                st.markdown('<h2 class="step-header">🎯 Step 4: Sign Recognition Results</h2>', unsafe_allow_html=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Predictions", len(predictions))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    unique_signs = len(set(p['prediction'] for p in predictions))
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Unique Signs Detected", unique_signs)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    # Most common prediction
                    from collections import Counter
                    pred_counts = Counter(p['prediction'] for p in predictions)
                    most_common = pred_counts.most_common(1)[0]
                    # Find average confidence for most common sign (handle None values for TRM)
                    confidences = [p['confidence'] for p in predictions if p['prediction'] == most_common[0] and p['confidence'] is not None]
                    if confidences:
                        avg_conf = np.mean(confidences)
                        conf_text = f"Avg conf: {avg_conf:.2f}"
                    else:
                        conf_text = "No confidence available"
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Most Frequent Sign", f"{most_common[0]} ({most_common[1]}x)", conf_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Main prediction display
                # st.markdown("### 🏆 Primary Detected Sign")
                # conf_display = f"Average confidence: {avg_conf:.2f}" if confidences else "No confidence scores available"
                # st.markdown(f"""
                # <div class="prediction-box">
                #     <div class="prediction-word">{most_common[0].upper()}</div>
                #     <div class="confidence">Detected {most_common[1]} times across {len(predictions)} windows</div>
                #     <div class="confidence">{conf_display}</div>
                # </div>
                # """, unsafe_allow_html=True)
                
                # Detailed predictions timeline
                st.markdown("### 📊 Prediction Timeline")
                
                # Create a chart-friendly format
                timeline_data = []
                for pred in predictions:
                    top1_text = pred['prediction']
                    top5_text = " | ".join(
                        [f"{item['word']} ({item['confidence']:.2f}%)" for item in pred.get('top5', [])]
                    ) if pred.get('top5') else "-"
                    timeline_data.append({
                        'Timestamp (s)': f"{pred['timestamp']:.2f}",
                        'Frame Range': f"{pred['frame_range'][0]}-{pred['frame_range'][1]}",
                        'Top-1 Prediction': top1_text,
                        'Top-5 Predictions': top5_text
                    })
                st.dataframe(timeline_data, use_container_width=True, height=300)
                
                # Sign frequency chart
                st.markdown("### 📈 Sign Frequency Distribution")
                
                sign_counts = {}
                for pred in predictions:
                    sign = pred['prediction']
                    sign_counts[sign] = sign_counts.get(sign, 0) + 1
                
                st.bar_chart(sign_counts)
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Export Results")
                
                results_json = json.dumps({
                    'video_info': {
                        'total_frames': total_frames,
                        'fps': fps,
                        'duration': total_frames / fps if fps > 0 else 0
                    },
                    'configuration': {
                        'window_size': window_size,
                        'overlap_percent': overlap_percent
                    },
                    'predictions': predictions,
                    'summary': {
                        'total_predictions': len(predictions),
                        'unique_signs': unique_signs,
                        'most_common_sign': most_common[0],
                        'sign_frequency': sign_counts
                    }
                }, indent=2)
                
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=results_json,
                    file_name="asl_recognition_results.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            else:
                st.warning("⚠️ No predictions generated. Video might be too short or missing keypoints.")
            
            # Cleanup
            Path(video_path).unlink(missing_ok=True)
    
    else:
        st.info("👆 Please upload a video file to begin analysis.")


if __name__ == "__main__":
    main()