import streamlit as st
import torch
import torch.nn.functional as F
import tempfile
import cv2
import numpy as np

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig
from features.video_to_keypoints import extract_video_keypoints
from features.build_tensor import build_feature_tensor_continuous


st.set_page_config(page_title="ASL Fixed Duration Inference", layout="wide")

st.title("🤟 ASL Fixed Duration Word Inference")
st.markdown("Upload video → Split by seconds → Convert each chunk to 48 frames → Predict.")


# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------
st.sidebar.header("⚙ Configuration")

dataset_name = st.sidebar.selectbox(
    "Vocabulary",
    ["app", "asl100", "asl300", "asl1000", "asl2000"]
)

chunk_duration = st.sidebar.number_input(
    "Chunk Duration (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5
)

topk = st.sidebar.slider("Top-K Predictions", 1, 10, 5)

checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path",
    f"checkpoints/{dataset_name}/best_model.pt"
)


# --------------------------------------------------
# Upload Video
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(uploaded_file)

    # --------------------------------------------------
    # Get Original Video Info
    # --------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_duration = total_video_frames / fps

    st.info(f"Original FPS: {fps:.2f}")
    st.info(f"Video Duration: {video_duration:.2f} seconds")
    st.info(f"Total Raw Frames: {total_video_frames}")

    # --------------------------------------------------
    # Extract Keypoints
    # --------------------------------------------------
    with st.spinner("Extracting keypoints..."):
        kps = extract_video_keypoints(video_path)
        full_tensor = build_feature_tensor_continuous(kps)

    T_full = full_tensor.shape[0]

    st.success("Feature extraction complete.")
    st.info(f"Processed Tensor Frames: {T_full}")

    # Sanity check
    if T_full != total_video_frames:
        st.warning("Processed frames differ from raw frames. FPS alignment adjusted.")

    # --------------------------------------------------
    # Compute Frames Per Chunk (Real Seconds)
    # --------------------------------------------------
    effective_fps = T_full / video_duration
    frames_per_chunk = max(1, round(chunk_duration * effective_fps))

    st.info(f"Frames per chunk (before resampling): {frames_per_chunk}")

    # --------------------------------------------------
    # Create 48-Frame Chunks
    # --------------------------------------------------
    chunks = []

    for start in range(0, T_full, frames_per_chunk):
        end = min(start + frames_per_chunk, T_full)
        chunk = full_tensor[start:end]

        if len(chunk) < 5:
            continue

        # Resample to exactly 48 frames
        indices = np.linspace(0, len(chunk) - 1, 48).astype(int)
        chunk_48 = chunk[indices]

        chunks.append((start, end, chunk_48))

    st.success(f"Total chunks created: {len(chunks)}")


    # --------------------------------------------------
    # Run Inference
    # --------------------------------------------------
    if st.button("🚀 Run Inference"):

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        st.info(f"Using device: {device}")

        cfg = DatasetConfig(dataset_name)
        idx_to_gloss = {v: k for k, v in cfg.label_map.items()}

        model = TRMMicro(num_classes=cfg.num_classes).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        results = []

        with torch.no_grad():

            for start, end, chunk_48 in chunks:

                input_tensor = (
                    torch.from_numpy(chunk_48)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )

                logits = model(input_tensor).squeeze(0)
                probs = F.softmax(logits, dim=0)

                values, indices = probs.topk(topk)

                predictions = []

                for idx, score in zip(indices, values):
                    word = idx_to_gloss.get(idx.item(), "UNKNOWN")
                    conf = float(score.item() * 100)

                    predictions.append({
                        "word": word,
                        "confidence": conf
                    })

                results.append({
                    "segment": f"{start}-{end}",
                    "predictions": predictions
                })

        st.success("Inference complete.")

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------
        st.markdown("## 📊 Chunk Predictions")

        for r in results:
            with st.expander(f"Segment {r['segment']}"):
                for rank, pred in enumerate(r["predictions"], 1):
                    st.write(
                        f"{rank}. {pred['word']} — {pred['confidence']:.2f}%"
                    )