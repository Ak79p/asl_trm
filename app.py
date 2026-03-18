"""
Streamlit App for ASL Sign Language Recognition
Visualizes the complete pipeline: video upload → keypoint extraction → word prediction

Modes:
  1. Sliding Window              – original fixed-window approach
  2. Hand Boundary (Sentence)    – accumulate hand-present frames; gaps = word boundaries
  3. Sliding Window in Boundary  – hand-boundary segmentation + sliding-window aggregation
                                   per word (avg softmax across windows → stable prediction)
"""
import os
import google.generativeai as genai
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

def generate_sentence_from_words(words, api_key):
    if not api_key:
        raise ValueError("Missing Gemini API key")

    # Configure dynamically
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = """
You are an expert ASL interpreter.

Task:
Given noisy word predictions, generate multiple possible corrected sentences.

Rules:
- Prefer high confidence words
- Ignore incorrect ones
- Use context to fix sentence
- You MAY replace words using top-5 suggestions
- You MAY add helper words (is, are, the, etc.)

Return output STRICTLY in JSON format:

{
  "best": {"sentence": "...", "confidence": 0-100},
  "alternatives": [
    {"sentence": "...", "confidence": 0-100},
    {"sentence": "...", "confidence": 0-100}
  ]
}

Confidence is your estimated certainty.

Predictions:
"""     
    for w in words:
        top5 = " | ".join(
            f"{t['word']} ({t['confidence']:.1f}%)"
            for t in w["top5"]
        )
        prompt += f"\nWord {w['word_index']}: {top5}"

    response = llm_model.generate_content(prompt)

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except:
        return {
            "best": {"sentence": text, "confidence": 50},
            "alternatives": []
        }

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:3rem; color:#1f77b4; text-align:center; margin-bottom:2rem; }
    .step-header  { font-size:1.8rem; color:#2ca02c; margin-top:2rem; margin-bottom:1rem; }
    .sentence-box {
        background-color:#f0f8ff; border:2px solid #1f77b4;
        border-radius:10px; padding:20px; margin:20px 0; text-align:center;
    }
    .sentence-text { font-size:2.4rem; font-weight:bold; color:#1f77b4; margin:10px 0; }
    .word-chip {
        display:inline-block; background:#1f77b4; color:white;
        border-radius:20px; padding:6px 16px; margin:4px;
        font-size:1.1rem; font-weight:bold;
    }
    .metric-card { background-color:#f8f9fa; border-radius:8px; padding:15px; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models(dataset_name="app", checkpoint_path="checkpoints/app/best_model.pt"):
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


# ── Feature helpers ───────────────────────────────────────────────────────────
def has_hands(kp: np.ndarray) -> bool:
    """Return True if either hand has at least one non-zero keypoint."""
    return bool(np.any(kp[0:21] != 0) or np.any(kp[21:42] != 0))


def build_trm_feature_tensor(kps_window: np.ndarray, target_t: int = 48) -> np.ndarray:
    """Uniformly sample to target_t frames and build (T,48,8) feature tensor."""
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
    if class_idx in idx_to_gloss:
        return idx_to_gloss[class_idx]
    class_idx_str = str(class_idx)
    if class_idx_str in idx_to_gloss:
        return idx_to_gloss[class_idx_str]
    return f"CLASS_{class_idx}"


def predict_from_buffer(buffer: list, model, idx_to_gloss, device) -> dict:
    """Run TRM on a list of keypoint arrays; return top-5 prediction dict."""
    try:
        raw = np.stack(buffer, axis=0)
        feat = build_trm_feature_tensor(raw, target_t=48)
        inp = torch.from_numpy(feat).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp).squeeze(0)
            probs = F.softmax(logits, dim=0)
            top_values, top_indices = probs.topk(min(5, probs.shape[0]))

        top5 = [
            {"word": decode_gloss(idx_to_gloss, int(i.item())), "confidence": float(s.item() * 100)}
            for i, s in zip(top_indices, top_values)
        ]
        return {"prediction": top5[0]["word"], "confidence": top5[0]["confidence"], "top5": top5}
    except Exception:
        return {"prediction": "UNKNOWN", "confidence": 0.0, "top5": []}


def predict_from_buffer_sliding(buffer: list, model, idx_to_gloss, device,
                                 window_size: int = 30, overlap_percent: float = 0.5) -> dict:
    """
    Slide a window over the word buffer, collect softmax distributions from each
    window, average them, and decode the top-5 from the averaged distribution.
    Falls back to a single full-buffer prediction when the buffer is shorter than
    one window.
    """
    if len(buffer) < window_size:
        # Not enough frames for even one full window – predict directly
        result = predict_from_buffer(buffer, model, idx_to_gloss, device)
        result["num_windows"] = 1
        return result

    stride = max(1, int(window_size * (1 - overlap_percent)))
    all_probs: list[np.ndarray] = []
    window_records: list[dict] = []   # per-window top-5 for display

    start = 0
    win_idx = 1
    while start + window_size <= len(buffer):
        window = buffer[start: start + window_size]
        try:
            raw = np.stack(window, axis=0)
            feat = build_trm_feature_tensor(raw, target_t=48)
            inp = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(inp).squeeze(0)
                probs = F.softmax(logits, dim=0).cpu().numpy()
            all_probs.append(probs)

            # Record top-5 for this window
            top_idx_w = np.argsort(probs)[::-1][:5]
            window_records.append({
                "window": win_idx,
                "frames": f"{start}–{start + window_size - 1}",
                "top1": decode_gloss(idx_to_gloss, int(top_idx_w[0])),
                "top1_conf": float(probs[top_idx_w[0]] * 100),
                "top5": " | ".join(
                    f"{decode_gloss(idx_to_gloss, int(i))} ({probs[i]*100:.1f}%)"
                    for i in top_idx_w
                ),
            })
        except Exception:
            pass
        start += stride
        win_idx += 1

    if not all_probs:
        result = predict_from_buffer(buffer, model, idx_to_gloss, device)
        result["num_windows"] = 0
        result["windows"] = []
        return result

    avg_probs = np.mean(all_probs, axis=0)
    top_idx = np.argsort(avg_probs)[::-1][:5]
    top5 = [
        {"word": decode_gloss(idx_to_gloss, int(i)), "confidence": float(avg_probs[i] * 100)}
        for i in top_idx
    ]
    return {
        "prediction": top5[0]["word"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
        "num_windows": len(all_probs),
        "windows": window_records,
    }


def predict_from_buffer_auto3(buffer: list, model, idx_to_gloss, device,
                               num_windows: int = 3,
                               window_size: int = None) -> dict:
    """
    Fit exactly `num_windows` windows over the word buffer and average softmax scores.

    window_size:
        If provided (user-configured), use that fixed size and auto-compute stride so
        all `num_windows` windows fit within the buffer.
        If None, auto-compute as N // num_windows (each window gets an equal share).

    stride = (N - window_size) // (num_windows - 1)

    Falls back to a single full-buffer prediction when the buffer is too short.
    """
    N = len(buffer)

    # Resolve window size
    if window_size is None:
        window_size = max(4, N // num_windows)
    else:
        window_size = max(4, min(window_size, N))   # clamp to [4, N]

    if N < window_size or num_windows < 2:
        result = predict_from_buffer(buffer, model, idx_to_gloss, device)
        result["num_windows"] = 1
        result["windows"] = []
        result["auto_window_size"] = window_size
        result["auto_stride"] = 0
        return result

    stride = max(1, (N - window_size) // (num_windows - 1))

    all_probs: list[np.ndarray] = []
    window_records: list[dict] = []

    for win_idx in range(num_windows):
        start = win_idx * stride
        end   = start + window_size
        if end > N:
            end = N
        window = buffer[start:end]
        if len(window) < 2:
            continue
        try:
            raw  = np.stack(window, axis=0)
            feat = build_trm_feature_tensor(raw, target_t=48)
            inp  = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(inp).squeeze(0)
                probs  = F.softmax(logits, dim=0).cpu().numpy()
            all_probs.append(probs)

            top_idx_w = np.argsort(probs)[::-1][:5]
            window_records.append({
                "window":      win_idx + 1,
                "frames":      f"{start}–{end - 1}",
                "top1":        decode_gloss(idx_to_gloss, int(top_idx_w[0])),
                "top1_conf":   float(probs[top_idx_w[0]] * 100),
                "top5": " | ".join(
                    f"{decode_gloss(idx_to_gloss, int(i))} ({probs[i]*100:.1f}%)"
                    for i in top_idx_w
                ),
            })
        except Exception:
            pass

    if not all_probs:
        result = predict_from_buffer(buffer, model, idx_to_gloss, device)
        result["num_windows"] = 0
        result["windows"] = []
        return result

    avg_probs = np.mean(all_probs, axis=0)
    top_idx   = np.argsort(avg_probs)[::-1][:5]
    top5 = [
        {"word": decode_gloss(idx_to_gloss, int(i)), "confidence": float(avg_probs[i] * 100)}
        for i in top_idx
    ]
    return {
        "prediction":  top5[0]["word"],
        "confidence":  top5[0]["confidence"],
        "top5":        top5,
        "num_windows": len(all_probs),
        "windows":     window_records,
        "auto_window_size": window_size,
        "auto_stride":      stride,
    }


# ── Mode 1: Sliding Window ────────────────────────────────────────────────────
def process_video_sliding(video_path, model, idx_to_gloss, device,
                          window_size=50, overlap_percent=0.20,
                          frame_skip=0, crop_lr_percent=0):
    """Original sliding-window approach."""
    stride = int(window_size * (1 - overlap_percent))
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    buffer = []
    predictions = []
    frame_count = 0
    processed = 0
    sample_frames = []
    sample_interval = max(1, total_frames // 10)

    progress_bar = st.progress(0)
    status_text = st.empty()

    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, refine_face_landmarks=False,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")

            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                continue

            processed += 1

            if crop_lr_percent > 0:
                h, w = frame.shape[:2]
                crop_px = int(w * crop_lr_percent / 100.0)
                if w - 2 * crop_px > 0:
                    frame = frame[:, crop_px:w - crop_px]

            kp = extract_frame_keypoints(frame, holistic)
            buffer.append(kp)

            if processed % sample_interval == 0 and len(sample_frames) < 10:
                sample_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if len(buffer) == window_size:
                valid = [k for k in buffer if k is not None]
                result = predict_from_buffer(valid, model, idx_to_gloss, device) if valid else \
                         {"prediction": "UNKNOWN", "confidence": None, "top5": []}

                start_frame = frame_count - window_size
                predictions.append({
                    "frame_range": (start_frame, frame_count),
                    "timestamp": start_frame / fps if fps > 0 else 0,
                    **result,
                })
                buffer = buffer[stride:]

    # Handle tail
    if buffer:
        valid = [k for k in buffer if k is not None]
        if valid:
            result = predict_from_buffer(valid, model, idx_to_gloss, device)
            predictions.append({
                "frame_range": (frame_count - len(buffer), frame_count),
                "timestamp": frame_count / fps if fps > 0 else 0,
                "padded": True,
                **result,
            })

    cap.release()
    progress_bar.empty()
    status_text.empty()
    return predictions, sample_frames, total_frames, fps


# ── Mode 2 & 3: Hand-Boundary Word Segmentation ──────────────────────────────
def process_video_sentence(video_path, model, idx_to_gloss, device,
                           min_word_frames=8, gap_frames=5,
                           frame_skip=0, crop_lr_percent=0,
                           word_predict_fn=None):
    """
    Accumulate frames that contain hand keypoints into a word buffer.
    When `gap_frames` consecutive frames have no hand keypoints, treat the
    accumulated buffer as one word, predict it via `word_predict_fn`, then reset.

    word_predict_fn(buffer) -> dict  – defaults to predict_from_buffer (mode 2).
    Pass predict_from_buffer_sliding (partial) for mode 3.
    Returns a list of word-level dicts with timestamps and top-5 predictions.
    """
    if word_predict_fn is None:
        word_predict_fn = lambda buf: predict_from_buffer(buf, model, idx_to_gloss, device)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    word_buffer: list[np.ndarray] = []   # keypoints for current word
    word_start_frame: int = 0
    words: list[dict] = []
    consecutive_no_hand: int = 0
    frame_count: int = 0
    sample_frames: list = []
    sample_interval = max(1, total_frames // 10)
    processed: int = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    live_sentence = st.empty()          # live sentence display

    def flush_buffer(end_frame: int):
        """Predict word from buffer and append to words list."""
        if len(word_buffer) < min_word_frames:
            return
        result = word_predict_fn(word_buffer)
        words.append({
            "word_index": len(words) + 1,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "top5": result["top5"],
            "num_windows": result.get("num_windows"),
            "windows": result.get("windows", []),
            "auto_window_size": result.get("auto_window_size"),
            "auto_stride": result.get("auto_stride"),
            "frame_range": (word_start_frame, end_frame),
            "timestamp_start": word_start_frame / fps if fps > 0 else 0,
            "timestamp_end": end_frame / fps if fps > 0 else 0,
            "num_frames": len(word_buffer),
        })
        sentence = " ".join(w["prediction"] for w in words)
        live_sentence.markdown(
            f'<div class="sentence-box"><div class="sentence-text">{sentence}</div></div>',
            unsafe_allow_html=True,
        )

    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, refine_face_landmarks=False,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)
            status_text.text(
                f"Frame {frame_count}/{total_frames} | "
                f"Word buffer: {len(word_buffer)} frames | "
                f"Words so far: {len(words)}"
            )

            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                continue

            processed += 1

            if crop_lr_percent > 0:
                w = frame.shape[1]
                crop_px = int(w * crop_lr_percent / 100.0)
                if w - 2 * crop_px > 0:
                    frame = frame[:, crop_px:w - crop_px]

            kp = extract_frame_keypoints(frame, holistic)

            if has_hands(kp):
                if not word_buffer:
                    word_start_frame = frame_count   # start of new word
                word_buffer.append(kp)
                consecutive_no_hand = 0

                if processed % sample_interval == 0 and len(sample_frames) < 10:
                    sample_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                consecutive_no_hand += 1
                if consecutive_no_hand >= gap_frames and word_buffer:
                    flush_buffer(frame_count)
                    word_buffer = []
                    consecutive_no_hand = 0

    # Flush any remaining frames at end of video
    if word_buffer:
        flush_buffer(frame_count)

    cap.release()
    progress_bar.empty()
    status_text.empty()
    return words, sample_frames, total_frames, fps


# ── UI helpers ────────────────────────────────────────────────────────────────
def show_sentence_results(words: list[dict]):

    sentence = " ".join(w["prediction"] for w in words)

    st.markdown("## 🗣️ Raw Prediction")

    st.markdown(
        f'<div class="sentence-box"><div class="sentence-text">{sentence}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Words", len(words))
    col2.metric("Unique Words", len(set(w["prediction"] for w in words)))
    avg_conf = np.mean([w["confidence"] for w in words]) if words else 0
    col3.metric("Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    # Word details
    st.markdown("### 📊 Word Details")

    table = []
    for w in words:
        table.append({
            "Word #": w["word_index"],
            "Prediction": w["prediction"],
            "Confidence": f"{w['confidence']:.1f}%",
            "Top-5": " | ".join(
                f"{t['word']} ({t['confidence']:.1f}%)"
                for t in w["top5"]
            )
        })

    st.dataframe(table, use_container_width=True)


def show_sliding_results(predictions: list[dict]):
    """Display sliding-window results (original style)."""
    from collections import Counter

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(predictions))
    col2.metric("Unique Signs", len(set(p["prediction"] for p in predictions)))

    pred_counts = Counter(p["prediction"] for p in predictions)
    most_common = pred_counts.most_common(1)[0]
    confs = [p["confidence"] for p in predictions if p["prediction"] == most_common[0] and p["confidence"]]
    col3.metric(
        "Most Frequent Sign",
        f'{most_common[0]} ({most_common[1]}x)',
        f'Avg {np.mean(confs):.1f}%' if confs else ""
    )

    st.markdown("---")
    st.markdown("### 📊 Prediction Timeline")
    timeline = []
    for pred in predictions:
        top5_text = " | ".join(
            f'{item["word"]} ({item["confidence"]:.1f}%)' for item in pred.get("top5", [])
        ) if pred.get("top5") else "-"
        timeline.append({
            "Timestamp (s)": f'{pred["timestamp"]:.2f}',
            "Frame Range": f'{pred["frame_range"][0]}-{pred["frame_range"][1]}',
            "Top-1": pred["prediction"],
            "Top-5": top5_text,
        })
    st.dataframe(timeline, use_container_width=True, height=300)

    st.markdown("### 📈 Sign Frequency")
    sign_counts = {}
    for p in predictions:
        sign_counts[p["prediction"]] = sign_counts.get(p["prediction"], 0) + 1
    st.bar_chart(sign_counts)


# ── Main ──────────────────────────────────────────────────────────────────────
# ONLY showing CHANGED MAIN PART (rest of your code remains SAME)

def main():
    st.markdown('<h1 class="main-header">🤟 ASL Sign Language Recognition</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar (ONLY DEFAULT MODE)
    with st.sidebar:
        st.header("⚙️ Configuration")

        dataset_name = st.selectbox("Vocabulary", ["app", "asl100"], index=0)
        checkpoint_path = st.text_input("Checkpoint Path", f"checkpoints/{dataset_name}/best_model.pt")
        
        st.markdown("---")
        st.markdown("#### 🔑 Gemini API Key (Optional)")

        api_key = st.text_input(
            "Enter Gemini API Key",
            type="password",   # ✅ masked
            placeholder="AIza...",
        )

        # Store securely in session
        if api_key:
            st.session_state["gemini_api_key"] = api_key

        st.markdown("---")
        st.markdown("#### Hand Boundary Settings")

        min_word_frames = st.slider("Min frames per word", 4, 48, 8, 1)
        gap_frames = st.slider("Gap frames", 1, 30, 5, 1)

        st.markdown("---")
        frame_skip = st.slider("Frame Skip", 0, 10, 0, 1)
        crop_lr_percent = st.slider("Crop (%)", 0, 40, 0, 1)

    # Load model
    model_key = f"{dataset_name}_{checkpoint_path}"
    if model_key not in st.session_state:
        model, labels, device = load_models(dataset_name, checkpoint_path)
        st.session_state["model"] = model
        st.session_state["labels"] = labels
        st.session_state["device"] = device

    model = st.session_state["model"]
    labels = st.session_state["labels"]
    device = st.session_state["device"]

    # Upload
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded_file is None:
        return

    # Reset when new video uploaded
    if "video_name" not in st.session_state or st.session_state["video_name"] != uploaded_file.name:
        st.session_state.clear()
        st.session_state["video_name"] = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(uploaded_file)

    # START PROCESSING BUTTON
    if st.button("🚀 Start Processing"):

        words, sample_frames, total_frames, fps = process_video_sentence(
            video_path,
            model,
            labels,
            device,
            min_word_frames=min_word_frames,
            gap_frames=gap_frames,
            frame_skip=frame_skip,
            crop_lr_percent=crop_lr_percent,
        )

        st.session_state["words"] = words

    # ── SHOW RESULTS (PERSISTENT) ─────────────────────
    if "words" in st.session_state:

        words = st.session_state["words"]

        if not words:
            st.warning("No words detected")
        else:
            st.success(f"Detected {len(words)} words")

            show_sentence_results(words)

            # ── LLM SECTION ─────────────────────
            st.markdown("---")
            st.markdown("### 🧠 AI Corrected Sentence")

            if st.button("✨ Generate Sentence"):
                api_key = st.session_state.get("gemini_api_key")

                if not api_key:
                    st.warning("Please enter Gemini API key in sidebar to enable AI sentence generation.")
                else:
                    with st.spinner("Generating..."):
                        try:
                            filtered = [w for w in words if w["confidence"] > 10] or words
                            st.session_state["llm"] = generate_sentence_from_words(filtered, api_key)
                        except Exception as e:
                            st.error(f"LLM Error: {e}")
                    
            if "llm" in st.session_state:
                result = st.session_state["llm"]

                st.markdown("### 🧠 AI Refined Sentence")

                st.markdown(
                    f'<div class="sentence-box"><div class="sentence-text">{result["best"]["sentence"]}</div></div>',
                    unsafe_allow_html=True,
                )
                
                st.caption(f'Confidence: {result["best"]["confidence"]}%')

                if result.get("alternatives"):
                    st.markdown("#### 🔁 Other Possibilities")
                    st.dataframe(result["alternatives"], use_container_width=True)

                

    Path(video_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
