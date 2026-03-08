import streamlit as st
import torch
import torch.nn.functional as F
import tempfile
import cv2
import numpy as np
import os

import google.generativeai as genai

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig
from features.video_to_keypoints import extract_video_keypoints
from features.build_tensor import build_feature_tensor_continuous


# --------------------------------------------------
# Gemini API Setup
# --------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel("gemini-2.5-flash")


def translate_asl_segments(segments):

    prompt = """
You are an ASL interpreter.

The following are possible ASL gloss predictions from a gesture recognition model.

Rules:
- Choose the most likely gloss from each segment outputs.
- Ignore gloss that do not make sense with surrounding signs.
- Do NOT invent new words.
- Produce a SHORT and SIMPLE English sentence.
- Keep the sentence close to ASL structure by filling in prepositions that bridges the sentence.

Examples:

Predictions: Vacation , Clock, Where, You
Output: "Where are you going on vacation?"

Segments:
"""

    for s in segments:
        prompt += f"\n{s['segment']} : {', '.join(s['candidates'])}"

    prompt += "\nReturn ONLY the simple English sentence."

    response = llm_model.generate_content(prompt)

    return response.text.strip()


# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="ASL Overlapping Window Inference", layout="wide")

st.title("🤟 ASL Continuous Inference Explorer")
st.markdown("Upload video → Overlapping windows → 48-frame inference → Sentence generation")


# --------------------------------------------------
# Sidebar Config
# --------------------------------------------------
st.sidebar.header("⚙ Configuration")

dataset_name = st.sidebar.selectbox(
    "Vocabulary",
    ["app", "asl-citizen"]
)

chunk_duration = st.sidebar.number_input(
    "Window Duration (seconds)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.5
)

stride_seconds = st.sidebar.number_input(
    "Stride (seconds)",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.25
)

topk = st.sidebar.slider("Top-K Predictions", 1, 10, 5)

checkpoint_path = st.sidebar.text_input(
    "Checkpoint Path",
    f"checkpoints/{dataset_name}/best_model.pt"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=5
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
    # Video Info
    # --------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_duration = total_video_frames / fps

    st.info(f"FPS: {fps:.2f}")
    st.info(f"Video Duration: {video_duration:.2f}s")
    st.info(f"Total Frames: {total_video_frames}")

    # --------------------------------------------------
    # Extract Keypoints
    # --------------------------------------------------
    with st.spinner("Extracting keypoints..."):
        kps = extract_video_keypoints(video_path)
        full_tensor = build_feature_tensor_continuous(kps)

    T_full = full_tensor.shape[0]

    st.success("Keypoints extracted")
    st.info(f"Tensor frames: {T_full}")

    # --------------------------------------------------
    # Window + Stride Calculation
    # --------------------------------------------------
    effective_fps = T_full / video_duration

    frames_per_chunk = max(1, round(chunk_duration * effective_fps))
    stride_frames = max(1, round(stride_seconds * effective_fps))

    st.info(f"Frames per window: {frames_per_chunk}")
    st.info(f"Stride frames: {stride_frames}")

    # --------------------------------------------------
    # Create Overlapping Windows
    # --------------------------------------------------
    chunks = []

    for start in range(0, T_full - frames_per_chunk + 1, stride_frames):

        end = start + frames_per_chunk
        chunk = full_tensor[start:end]

        if len(chunk) < 5:
            continue

        # resample to 48 frames
        indices = np.linspace(0, len(chunk) - 1, 48).astype(int)
        chunk_48 = chunk[indices]

        chunks.append((start, end, chunk_48))

    st.success(f"Total windows created: {len(chunks)}")


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

                    if conf >= confidence_threshold:

                        predictions.append({
                            "word": word,
                            "confidence": conf
                        })
                
                if len(predictions) > 0:
                    results.append({
                        "segment": f"{start}-{end}",
                        "predictions": predictions
                    })

        st.success("Inference complete")

        # --------------------------------------------------
        # Prepare Data for LLM
        # --------------------------------------------------
        segments_for_llm = []

        for r in results:

            candidates = []

            for p in r["predictions"]:
                candidates.append(f"{p['word']} ({p['confidence']:.1f}%)")

            segments_for_llm.append({
                "segment": r["segment"],
                "candidates": candidates
            })

        st.session_state["results"] = results
        st.session_state["segments_for_llm"] = segments_for_llm


# --------------------------------------------------
# Prediction Display
# --------------------------------------------------
if "results" in st.session_state:

    st.markdown("## 📊 Window Predictions")

    for r in st.session_state["results"]:

        with st.expander(f"Segment {r['segment']}"):

            for rank, pred in enumerate(r["predictions"], 1):

                st.write(
                    f"{rank}. {pred['word']} — {pred['confidence']:.2f}%"
                )


# --------------------------------------------------
# Gemini Sentence Generation
# --------------------------------------------------
if "segments_for_llm" in st.session_state:

    st.markdown("## 🧠 Sentence Interpretation")

    if st.button("🧠 Generate Sentence with Gemini"):

        with st.spinner("Generating sentence..."):

            sentence = translate_asl_segments(
                st.session_state["segments_for_llm"]
            )

        st.session_state["generated_sentence"] = sentence

    if "generated_sentence" in st.session_state:

        st.success(st.session_state["generated_sentence"])