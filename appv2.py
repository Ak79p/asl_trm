"""
Streamlit App for ASL Sign Language Recognition
Visualizes the complete pipeline: video upload → keypoint extraction → word prediction

Modes:
  1. Sliding Window              – original fixed-window approach
  2. Hand Boundary (Sentence)    – accumulate hand-present frames; gaps = word boundaries
  3. Sliding Window in Boundary  – hand-boundary segmentation + sliding-window aggregation
                                   per word (avg softmax across windows → stable prediction)
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
def load_models(dataset_name="asl100", checkpoint_path="checkpoints/asl100/best_model.pt"):
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
    """Display sentence-level results."""
    sentence = " ".join(w["prediction"] for w in words)

    st.markdown("### 🗣️ Predicted Sentence")
    st.markdown(
        f'<div class="sentence-box"><div class="sentence-text">{sentence}</div></div>',
        unsafe_allow_html=True,
    )

    # Word chips
    chips_html = "".join(
        f'<span class="word-chip">#{w["word_index"]} {w["prediction"]} '
        f'({w["confidence"]:.1f}%)</span>'
        for w in words
    )
    st.markdown(chips_html, unsafe_allow_html=True)
    st.markdown("---")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Words", len(words))
    col2.metric("Unique Words", len(set(w["prediction"] for w in words)))
    avg_conf = np.mean([w["confidence"] for w in words]) if words else 0
    col3.metric("Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    # Per-word table — expands to per-window rows for Sliding Window in Boundary mode
    st.markdown("### 📊 Word Details")
    has_window_detail = any(w.get("windows") for w in words)
    table = []
    for w in words:
        top5_avg_text = " | ".join(
            f'{item["word"]} ({item["confidence"]:.1f}%)' for item in w.get("top5", [])
        )
        # ── Aggregate / summary row ──
        summary_row = {
            "Word #": w["word_index"],
            "Row type": "▶ WORD (avg)",
            "Prediction": w["prediction"],
            "Confidence": f'{w["confidence"]:.1f}%',
            "Frames": w["num_frames"],
            "Time (s)": f'{w["timestamp_start"]:.2f} – {w["timestamp_end"]:.2f}',
            "Top-5": top5_avg_text,
        }
        if has_window_detail:
            summary_row["Window"] = f'— ({w.get("num_windows", 1)} windows)'
            summary_row["Frame range (in segment)"] = "—"
        # Auto-window columns — only populated for "Auto Window" mode
        if w.get("auto_window_size") is not None:
            summary_row["Auto win size"] = w["auto_window_size"]
            summary_row["Auto stride"] = w["auto_stride"]
        table.append(summary_row)

        # ── One row per sliding window ──
        for r in w.get("windows", []):
            table.append({
                "Word #": w["word_index"],
                "Row type": f'  └ win {r["window"]}',
                "Prediction": r["top1"],
                "Confidence": f'{r["top1_conf"]:.1f}%',
                "Frames": "—",
                "Time (s)": "—",
                "Top-5": r["top5"],
                "Window": r["window"],
                "Frame range (in segment)": r["frames"],
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
def main():
    st.markdown('<h1 class="main-header">🤟 ASL Sign Language Recognition</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        dataset_name = st.selectbox(
            "Vocabulary",
            ["app", "asl100", "asl300", "asl1000", "asl2000"],
            index=1,
        )
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            f"checkpoints/{dataset_name}/best_model.pt",
        )

        st.markdown("---")
        mode = st.radio(
            "Processing Mode",
            [
                "Hand Boundary (Sentence)",
                "Auto Window (3 per word)",
                "Sliding Window in Boundary",
                "Sliding Window",
            ],
            index=0,
            help=(
                "**Hand Boundary:** one prediction per hand-present word segment.\n\n"
                "**Auto Window (3 per word):** auto-sizes window so exactly 3 windows "
                "cover each word segment; averages their softmax → balanced coverage.\n\n"
                "**Sliding Window in Boundary:** manual window size + overlap inside "
                "each word segment; averages softmax.\n\n"
                "**Sliding Window:** original fixed-window approach (no hand filtering)."
            ),
        )

        st.markdown("---")

        # Hand-boundary controls (shared by boundary-based modes)
        BOUNDARY_MODES = ("Hand Boundary (Sentence)", "Auto Window (3 per word)", "Sliding Window in Boundary")
        if mode in BOUNDARY_MODES:
            st.markdown("#### Hand Boundary Settings")
            min_word_frames = st.slider(
                "Min frames per word", 4, 48, 8, 1,
                help="Buffers shorter than this are discarded as noise",
            )
            gap_frames = st.slider(
                "Gap frames (word boundary)", 1, 30, 5, 1,
                help="Consecutive no-hand frames before treating gap as word end",
            )

        if mode == "Auto Window (3 per word)":
            st.markdown("#### Auto Window Settings")
            auto_num_windows = st.slider(
                "Windows per word", 2, 6, 3, 1,
                help="Stride is auto-computed so exactly this many windows cover each word segment",
            )
            auto_window_size = st.slider(
                "Window Size (frames)", 4, 48, 20, 1,
                help="Fixed window size per window; stride adapts per word to fit the requested number of windows",
            )

        if mode == "Sliding Window in Boundary":
            st.markdown("#### Intra-word Sliding Window")
            sw_window_size = st.slider("Window Size (frames)", 10, 80, 30, 5,
                                       help="Sliding window size applied within each word segment")
            sw_overlap = st.slider("Overlap Percentage", 0.0, 0.90, 0.50, 0.05,
                                   help="Overlap between consecutive windows inside one word")

        if mode == "Sliding Window":
            st.markdown("#### Sliding Window Settings")
            window_size = st.slider("Window Size (frames)", 20, 100, 50, 5)
            overlap_percent = st.slider("Overlap Percentage", 0.0, 0.90, 0.20, 0.05)

        st.markdown("---")
        frame_skip = st.slider(
            "Frame Skip", 0, 10, 0, 1,
            help="0 = all frames; N = keep every (N+1)th frame",
        )
        crop_lr_percent = st.slider(
            "Crop Left/Right (%)", 0, 40, 0, 1,
            help="Crop X% from each side before processing",
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(
            "**Model:** TRM (Transformer)\n\n"
            "**Input:** Body pose + hand keypoints (48 pts)\n\n"
            "**Training:** 48 uniformly sampled frames per word"
        )

    # ── Model loading ─────────────────────────────────────────────────────────
    model_key  = f"trm_engine::{dataset_name}::{checkpoint_path}"
    labels_key = f"trm_labels::{dataset_name}"
    device_key = f"trm_device::{dataset_name}::{checkpoint_path}"

    if model_key not in st.session_state:
        model, labels, device = load_models(dataset_name, checkpoint_path)
        if model:
            st.session_state[model_key]  = model
            st.session_state[labels_key] = labels
            st.session_state[device_key] = device
            st.success("✅ TRM models loaded successfully!")
        else:
            st.error("❌ Failed to load models. Please check configuration.")
            return
    else:
        model  = st.session_state[model_key]
        labels = st.session_state.get(labels_key, {})
        device = st.session_state.get(device_key, "cpu")

    # ── Video upload ──────────────────────────────────────────────────────────
    st.markdown('<h2 class="step-header">📹 Step 1: Upload Video</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video containing ASL sign language gestures",
    )

    if uploaded_file is None:
        st.info("👆 Please upload a video file to begin analysis.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(uploaded_file)
    with col2:
        st.markdown("### 📝 Video Info")
        cap = cv2.VideoCapture(video_path)
        fps_info   = cap.get(cv2.CAP_PROP_FPS)
        n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration   = n_frames / fps_info if fps_info > 0 else 0
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        st.metric("Duration",     f"{duration:.2f} s")
        st.metric("Total Frames", str(n_frames))
        st.metric("FPS",          f"{fps_info:.1f}")
        st.metric("Resolution",   f"{width}x{height}")

    st.markdown("---")

    if not st.button("🚀 Start Processing", type="primary", use_container_width=True):
        return

    st.markdown('<h2 class="step-header">⚡ Step 2: Processing Video</h2>', unsafe_allow_html=True)

    # ── Run chosen mode ───────────────────────────────────────────────────────
    BOUNDARY_MODES = ("Hand Boundary (Sentence)", "Auto Window (3 per word)", "Sliding Window in Boundary")
    if mode in BOUNDARY_MODES:

        if mode == "Hand Boundary (Sentence)":
            predict_fn = None   # defaults to predict_from_buffer inside the function
            export_mode_key = "hand_boundary"
            export_extra = {}
            st.info(
                f"**Hand Boundary** – one prediction per hand-present word segment. "
                f"Gap ≥ {gap_frames} frames → word boundary; min {min_word_frames} frames/word."
            )

        elif mode == "Auto Window (3 per word)":
            _n, _ws = auto_num_windows, auto_window_size
            predict_fn = lambda buf: predict_from_buffer_auto3(
                buf, model, labels, device, num_windows=_n, window_size=_ws,
            )
            export_mode_key = "auto_window"
            export_extra = {"num_windows": auto_num_windows, "window_size": auto_window_size}
            st.info(
                f"**Auto Window ({auto_num_windows} per word)** – window size and stride are "
                f"auto-computed per word so exactly {auto_num_windows} windows cover the segment; "
                f"softmax scores are averaged. "
                f"Gap ≥ {gap_frames} frames → word boundary; min {min_word_frames} frames/word."
            )

        else:  # Sliding Window in Boundary
            predict_fn = lambda buf: predict_from_buffer_sliding(
                buf, model, labels, device,
                window_size=sw_window_size,
                overlap_percent=sw_overlap,
            )
            export_mode_key = "sliding_window_in_boundary"
            export_extra = {"sw_window_size": sw_window_size, "sw_overlap": sw_overlap}
            st.info(
                f"**Sliding Window in Boundary** – within each word a {sw_window_size}-frame "
                f"window (overlap {sw_overlap:.0%}) slides and softmax scores are averaged. "
                f"Gap ≥ {gap_frames} frames → word boundary; min {min_word_frames} frames/word."
            )

        words, sample_frames, total_frames, fps = process_video_sentence(
            video_path, model, labels, device,
            min_word_frames=min_word_frames,
            gap_frames=gap_frames,
            frame_skip=frame_skip,
            crop_lr_percent=crop_lr_percent,
            word_predict_fn=predict_fn,
        )

        if not words:
            st.warning("⚠️ No words detected. Try lowering 'Min frames per word' or 'Gap frames'.")
        else:
            st.success(f"✅ Done! Detected {len(words)} word(s).")

            st.markdown('<h2 class="step-header">🖼️ Step 3: Sample Frames</h2>', unsafe_allow_html=True)
            if sample_frames:
                cols = st.columns(min(5, len(sample_frames)))
                for idx, frame in enumerate(sample_frames[:10]):
                    with cols[idx % 5]:
                        st.image(frame, use_container_width=True, caption=f"Frame {idx+1}")
            st.markdown("---")

            st.markdown('<h2 class="step-header">🎯 Step 4: Sentence Results</h2>', unsafe_allow_html=True)
            show_sentence_results(words)

            # Export
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            export_settings = {"min_word_frames": min_word_frames, "gap_frames": gap_frames}
            export_settings.update(export_extra)
            export = {
                "mode": export_mode_key,
                "video_info": {"total_frames": total_frames, "fps": fps, "duration": total_frames / fps if fps > 0 else 0},
                "settings": export_settings,
                "sentence": " ".join(w["prediction"] for w in words),
                "words": words,
            }
            st.download_button(
                "📥 Download Results (JSON)",
                data=json.dumps(export, indent=2),
                file_name="asl_sentence_results.json",
                mime="application/json",
                use_container_width=True,
            )

    else:  # Sliding Window
        predictions, sample_frames, total_frames, fps = process_video_sliding(
            video_path, model, labels, device,
            window_size=window_size,
            overlap_percent=overlap_percent,
            frame_skip=frame_skip,
            crop_lr_percent=crop_lr_percent,
        )

        if not predictions:
            st.warning("⚠️ No predictions generated. Video might be too short or missing keypoints.")
        else:
            st.success(f"✅ Processing complete! Found {len(predictions)} predictions.")

            st.markdown('<h2 class="step-header">🖼️ Step 3: Sample Frames</h2>', unsafe_allow_html=True)
            if sample_frames:
                cols = st.columns(min(5, len(sample_frames)))
                for idx, frame in enumerate(sample_frames[:10]):
                    with cols[idx % 5]:
                        st.image(frame, use_container_width=True, caption=f"Frame {idx+1}")
            st.markdown("---")

            st.markdown('<h2 class="step-header">🎯 Step 4: Sign Recognition Results</h2>', unsafe_allow_html=True)
            show_sliding_results(predictions)

            # Export
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            from collections import Counter
            sign_counts = Counter(p["prediction"] for p in predictions)
            export = {
                "mode": "sliding_window",
                "video_info": {"total_frames": total_frames, "fps": fps, "duration": total_frames / fps if fps > 0 else 0},
                "settings": {"window_size": window_size, "overlap_percent": overlap_percent},
                "predictions": predictions,
                "summary": {"sign_frequency": dict(sign_counts)},
            }
            st.download_button(
                "📥 Download Results (JSON)",
                data=json.dumps(export, indent=2),
                file_name="asl_recognition_results.json",
                mime="application/json",
                use_container_width=True,
            )

    Path(video_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
