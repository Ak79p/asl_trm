import streamlit as st
import torch
import torch.nn.functional as F
import tempfile

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig
from features.video_to_keypoints import extract_video_keypoints
from features.build_tensor import build_feature_tensor_continuous
from utils.llm_sentence_builder import build_sentence_from_windows


st.set_page_config(page_title="ASL Sliding Window Analyzer", layout="wide")

st.title("🤟 ASL Sliding Window Inference Explorer")
st.markdown("Upload → Inspect frames → Configure window/stride → Run inference.")


# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "full_tensor" not in st.session_state:
    st.session_state.full_tensor = None

if "results" not in st.session_state:
    st.session_state.results = None

if "structured_for_llm" not in st.session_state:
    st.session_state.structured_for_llm = None


# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙ Configuration")

dataset_name = st.sidebar.selectbox(
    "Vocabulary",
    ["app", "asl100", "asl300", "asl1000", "asl2000"]
)

topk = st.sidebar.slider("Top-K", 1, 10, 5)

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
    # STEP 1 — Extract Features Immediately
    # --------------------------------------------------
    if st.session_state.full_tensor is None:

        with st.spinner("Extracting keypoints and building feature tensor..."):
            kps = extract_video_keypoints(video_path)
            full_tensor = build_feature_tensor_continuous(kps)

        st.session_state.full_tensor = full_tensor
        st.success("Video processed successfully.")

    # --------------------------------------------------
    # Frame Count Display
    # --------------------------------------------------
    if st.session_state.full_tensor is not None:

        full_tensor = st.session_state.full_tensor
        T_full = full_tensor.shape[0]

        st.markdown("## 📊 Video Statistics")
        st.info(f"Total Frames After Processing: {T_full}")

        # --------------------------------------------------
        # Window & Stride Configuration
        # --------------------------------------------------
        st.markdown("## ⚙ Sliding Window Configuration")

        # --- Slider Controls ---
        window_slider = st.slider(
            "Window Size (Slider)",
            min_value=1,
            max_value=T_full,
            value=min(30, T_full)
        )

        stride_slider = st.slider(
            "Stride (Slider)",
            min_value=1,
            max_value=min(50, window_slider),
            value=min(4, window_slider)
        )

        st.markdown("### Manual Override")

        window_manual = st.number_input(
            "Window Size (Manual)",
            min_value=1,
            max_value=T_full,
            value=window_slider
        )

        stride_manual = st.number_input(
            "Stride (Manual)",
            min_value=1,
            max_value=window_manual,
            value=stride_slider
        )

        # Final Values
        window_size = int(window_manual)
        stride = int(stride_manual)

        if stride > window_size:
            stride = window_size
            st.warning("Stride adjusted to match window size.")

        # Estimate windows
        num_windows = max(0, (T_full - window_size) // stride + 1)
        st.info(f"Estimated number of windows: {num_windows}")

        # --------------------------------------------------
        # RUN INFERENCE
        # --------------------------------------------------
        if st.button("🚀 Run Sliding Window Inference"):

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
            structured_for_llm = []

            with torch.no_grad():
                for start in range(0, T_full - window_size + 1, stride):

                    end = start + window_size
                    window = full_tensor[start:end]

                    window_t = (
                        torch.from_numpy(window)
                        .float()
                        .unsqueeze(0)
                        .to(device)
                    )

                    logits = model(window_t).squeeze(0)
                    probs = F.softmax(logits, dim=0)

                    values, indices = probs.topk(topk)

                    top_predictions = []
                    word_conf_pairs = []

                    for idx, score in zip(indices, values):
                        word = idx_to_gloss.get(idx.item(), "UNKNOWN")
                        conf = float(score.item() * 100)

                        top_predictions.append({
                            "word": word,
                            "confidence": conf
                        })

                        word_conf_pairs.append((word, conf))

                    results.append({
                        "window": f"{start}-{end}",
                        "topk": top_predictions
                    })

                    structured_for_llm.append({
                        "window": f"{start}-{end}",
                        "words": word_conf_pairs
                    })

            st.session_state.results = results
            st.session_state.structured_for_llm = structured_for_llm

            st.success("Inference complete.")


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
if st.session_state.results is not None:

    st.markdown("## 📊 Sliding Window Results")

    for r in st.session_state.results:
        with st.expander(f"Window {r['window']}"):
            for rank, pred in enumerate(r["topk"], 1):
                st.write(
                    f"{rank}. {pred['word']} — {pred['confidence']:.2f}%"
                )


# --------------------------------------------------
# SENTENCE RECONSTRUCTION
# --------------------------------------------------
if st.session_state.structured_for_llm is not None:

    st.markdown("---")
    st.markdown("## 🧠 Sentence Reconstruction")

    hf_token = st.text_input("Enter Hugging Face API Token", type="password")

    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 5)

    if st.button("✨ Generate Sentence"):

        if not hf_token:
            st.error("Please enter Hugging Face token.")
        else:

            filtered_structured = []

            for item in st.session_state.structured_for_llm:
                filtered_words = [
                    (w, c)
                    for (w, c) in item["words"]
                    if c >= confidence_threshold
                ]

                if filtered_words:
                    filtered_structured.append({
                        "window": item["window"],
                        "words": filtered_words
                    })

            if not filtered_structured:
                st.warning("No words passed confidence threshold.")
            else:
                with st.spinner("Generating sentence..."):
                    sentence = build_sentence_from_windows(
                        window_predictions=filtered_structured,
                        hf_token=hf_token
                    )

                st.markdown("### 📝 Reconstructed Sentence")
                st.success(sentence)