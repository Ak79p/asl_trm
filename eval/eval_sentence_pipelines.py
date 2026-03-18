"""
eval/eval_sentence_pipelines.py

Evaluate appv2.py inference pipelines on synthetic sentence-level videos.

Ground-truth words are embedded in the video filename:
    e.g.  HELLO_WORLD_HOW_ARE_YOU.mp4  →  ["HELLO", "WORLD", "HOW", "ARE", "YOU"]

Pipelines evaluated
-------------------
  1. hand_boundary          – one TRM call per hand-present segment
  2. auto_window            – N evenly-spaced windows per segment, avg softmax
  3. sliding_in_boundary    – fixed sliding window inside each segment, avg softmax
  4. sliding_window         – original fixed sliding window (no hand filtering)

Outputs (written to --output_dir)
----------------------------------
  prediction_log.jsonl   – one JSON line per video × pipeline
  accuracy_log.csv        – word-level and sentence-level metrics per pipeline

Usage
-----
  python -m eval.eval_sentence_pipelines \
      --video_dir  <path/to/sentence/videos> \
      --checkpoint checkpoints/asl100/best_model.pt \
      --dataset    asl100 \
      --output_dir eval/results/sentence_pipelines
"""

from __future__ import annotations

import argparse
import json
import csv
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Logger setup
# ─────────────────────────────────────────────────────────────────────────────

class _TqdmHandler(logging.StreamHandler):
    """StreamHandler that routes through tqdm.write() so progress bars stay intact."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(log_path: Path) -> logging.Logger:
    """
    Create a logger that writes to both stdout (INFO+) and a file (DEBUG+).
    Uses _TqdmHandler so tqdm progress bars are not corrupted.
    """
    log = logging.getLogger("eval_pipelines")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — DEBUG level, captures everything
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    # Console handler — INFO level, tqdm-safe
    ch = _TqdmHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    return log

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig
from features.extract_keypoints import extract_frame_keypoints, create_holistic_landmarker


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with appv2.py
# ─────────────────────────────────────────────────────────────────────────────

def has_hands(kp: np.ndarray) -> bool:
    return bool(np.any(kp[0:21] != 0) or np.any(kp[21:42] != 0))


def build_trm_feature_tensor(kps: np.ndarray, target_t: int = 48) -> np.ndarray:
    indices = np.linspace(0, len(kps) - 1, target_t).astype(int)
    kps = kps[indices]
    vel = np.zeros_like(kps); vel[1:] = kps[1:] - kps[:-1]
    acc = np.zeros_like(vel);  acc[1:] = vel[1:] - vel[:-1]
    lh = kps[:, 0:21].mean(axis=1, keepdims=True)
    rh = kps[:, 21:42].mean(axis=1, keepdims=True)
    rel = kps.copy(); rel[:, 0:21] -= lh; rel[:, 21:42] -= rh
    return np.concatenate([kps, vel, acc, rel], axis=-1).astype(np.float32)


def decode_gloss(idx_to_gloss, idx):
    if idx in idx_to_gloss:       return idx_to_gloss[idx]
    if str(idx) in idx_to_gloss:  return idx_to_gloss[str(idx)]
    return f"CLASS_{idx}"


def infer_buffer(buffer: list, model, idx_to_gloss, device) -> dict:
    """Run TRM on a keypoint buffer; return top-5 dict."""
    try:
        raw  = np.stack(buffer, axis=0)
        feat = build_trm_feature_tensor(raw, target_t=48)
        inp  = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(inp).squeeze(0), dim=0)
            top_v, top_i = probs.topk(min(5, probs.shape[0]))
        top5 = [{"word": decode_gloss(idx_to_gloss, int(i)), "conf": float(v) * 100}
                for i, v in zip(top_i, top_v)]
        return {"top1": top5[0]["word"], "conf": top5[0]["conf"], "top5": top5}
    except Exception as e:
        return {"top1": "UNKNOWN", "conf": 0.0, "top5": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Video → keypoints extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_keypoints_from_video(video_path: str) -> list[np.ndarray]:
    """Return list of (48,2) keypoint arrays, one per frame."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    frames_kp = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0
    with create_holistic_landmarker() as landmarker:
        with tqdm(total=total, desc="  keypoints", unit="fr", leave=False) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp_ms = int(frame_count * 1000 / fps)
                frames_kp.append(extract_frame_keypoints(frame, landmarker, timestamp_ms))
                frame_count += 1
                pbar.update(1)
    cap.release()
    return frames_kp


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline implementations
# ─────────────────────────────────────────────────────────────────────────────

def _segment_by_hands(frames_kp: list, min_word_frames: int, gap_frames: int) -> list[list]:
    """
    Return list of keypoint-list segments, one per detected word,
    using the same hand-presence boundary logic as appv2.py.
    """
    segments: list[list] = []
    buf:      list       = []
    no_hand:  int        = 0

    def flush():
        if len(buf) >= min_word_frames:
            segments.append(list(buf))

    for kp in frames_kp:
        if has_hands(kp):
            buf.append(kp)
            no_hand = 0
        else:
            no_hand += 1
            if no_hand >= gap_frames and buf:
                flush(); buf.clear(); no_hand = 0

    if buf:
        flush()

    return segments


def pipeline_hand_boundary(frames_kp, model, idx_to_gloss, device,
                            min_word_frames=8, gap_frames=5) -> list[dict]:
    """One TRM inference per hand-boundary segment."""
    segs = _segment_by_hands(frames_kp, min_word_frames, gap_frames)
    results = []
    for i, seg in enumerate(segs):
        r = infer_buffer(seg, model, idx_to_gloss, device)
        results.append({"word_idx": i + 1, "num_frames": len(seg), **r})
    return results


def pipeline_auto_window(frames_kp, model, idx_to_gloss, device,
                         min_word_frames=8, gap_frames=5,
                         num_windows=3, window_size=20) -> list[dict]:
    """N evenly-spaced fixed-size windows per segment, average softmax."""
    segs = _segment_by_hands(frames_kp, min_word_frames, gap_frames)
    results = []
    for i, seg in enumerate(segs):
        N  = len(seg)
        ws = max(4, min(window_size, N))
        if N < ws or num_windows < 2:
            r = infer_buffer(seg, model, idx_to_gloss, device)
            results.append({"word_idx": i + 1, "num_frames": N,
                             "num_windows": 1, "window_size": ws, "stride": 0, **r})
            continue

        stride    = max(1, (N - ws) // (num_windows - 1))
        all_probs = []
        for w in range(num_windows):
            start = w * stride
            end   = min(start + ws, N)
            chunk = seg[start:end]
            if len(chunk) < 2:
                continue
            try:
                raw  = np.stack(chunk, axis=0)
                feat = build_trm_feature_tensor(raw, target_t=48)
                inp  = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    all_probs.append(F.softmax(model(inp).squeeze(0), dim=0).cpu().numpy())
            except Exception:
                pass

        if not all_probs:
            r = infer_buffer(seg, model, idx_to_gloss, device)
        else:
            avg = np.mean(all_probs, axis=0)
            top_i = np.argsort(avg)[::-1][:5]
            top5  = [{"word": decode_gloss(idx_to_gloss, int(j)), "conf": float(avg[j]) * 100}
                     for j in top_i]
            r = {"top1": top5[0]["word"], "conf": top5[0]["conf"], "top5": top5}

        results.append({"word_idx": i + 1, "num_frames": N,
                        "num_windows": len(all_probs),
                        "window_size": ws, "stride": stride, **r})
    return results


def pipeline_sliding_in_boundary(frames_kp, model, idx_to_gloss, device,
                                  min_word_frames=8, gap_frames=5,
                                  window_size=30, overlap=0.5) -> list[dict]:
    """Fixed sliding window inside each segment, average softmax."""
    segs = _segment_by_hands(frames_kp, min_word_frames, gap_frames)
    results = []
    for i, seg in enumerate(segs):
        N      = len(seg)
        stride = max(1, int(window_size * (1 - overlap)))
        if N < window_size:
            r = infer_buffer(seg, model, idx_to_gloss, device)
            results.append({"word_idx": i + 1, "num_frames": N, "num_windows": 1, **r})
            continue

        all_probs = []
        start = 0
        while start + window_size <= N:
            chunk = seg[start: start + window_size]
            try:
                raw  = np.stack(chunk, axis=0)
                feat = build_trm_feature_tensor(raw, target_t=48)
                inp  = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    all_probs.append(F.softmax(model(inp).squeeze(0), dim=0).cpu().numpy())
            except Exception:
                pass
            start += stride

        if not all_probs:
            r = infer_buffer(seg, model, idx_to_gloss, device)
        else:
            avg   = np.mean(all_probs, axis=0)
            top_i = np.argsort(avg)[::-1][:5]
            top5  = [{"word": decode_gloss(idx_to_gloss, int(j)), "conf": float(avg[j]) * 100}
                     for j in top_i]
            r = {"top1": top5[0]["word"], "conf": top5[0]["conf"], "top5": top5}

        results.append({"word_idx": i + 1, "num_frames": N, "num_windows": len(all_probs), **r})
    return results


def pipeline_sliding_window(frames_kp, model, idx_to_gloss, device,
                             window_size=50, overlap=0.20) -> list[dict]:
    """Original sliding window with no hand filtering."""
    stride    = max(1, int(window_size * (1 - overlap)))
    all_preds = []
    buf       = []
    for i, kp in enumerate(frames_kp):
        buf.append(kp)
        if len(buf) == window_size:
            valid = [k for k in buf if k is not None]
            r = infer_buffer(valid, model, idx_to_gloss, device) if valid \
                else {"top1": "UNKNOWN", "conf": 0.0, "top5": []}
            all_preds.append({"window_start": i - window_size + 1,
                               "window_end": i, **r})
            buf = buf[stride:]

    if buf:
        valid = [k for k in buf if k is not None]
        if valid:
            r = infer_buffer(valid, model, idx_to_gloss, device)
            all_preds.append({"window_start": len(frames_kp) - len(buf),
                               "window_end": len(frames_kp), **r})
    return all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def word_accuracy(gt_words: list[str], pred_words: list[str]) -> dict:
    """
    Positional word accuracy: compare gt and pred word-by-word up to
    min(len(gt), len(pred)).  Also report length match and exact-sentence match.
    All comparisons are case-insensitive (uppercased).
    """
    gt_up   = [w.upper() for w in gt_words]
    pred_up = [w.upper() for w in pred_words]

    n_gt   = len(gt_up)
    n_pred = len(pred_up)
    n_cmp  = min(n_gt, n_pred)

    correct = sum(g == p for g, p in zip(gt_up[:n_cmp], pred_up[:n_cmp]))
    word_acc        = correct / n_gt if n_gt else 0.0
    length_match    = int(n_gt == n_pred)
    sentence_match  = int(gt_up == pred_up)

    return {
        "gt_len":        n_gt,
        "pred_len":      n_pred,
        "words_correct": correct,
        "word_accuracy": round(word_acc, 4),
        "length_match":  length_match,
        "sentence_match": sentence_match,
    }


# ── helpers for BLEU / ROUGE ─────────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def compute_nlp_metrics(gt_words: list[str], pred_words: list[str]) -> dict:
    """
    Compute BLEU-1, BLEU-2 (with brevity penalty + add-1 smoothing for short
    sentences) and ROUGE-1, ROUGE-2, ROUGE-L (F1) between two word sequences.
    All comparisons are uppercased.
    """
    gt   = [w.upper() for w in gt_words]
    pred = [w.upper() for w in pred_words]

    zero = {"bleu1": 0.0, "bleu2": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if not gt or not pred:
        return zero

    # ── BLEU-N ────────────────────────────────────────────────────────────────
    def bleu_n(n: int) -> float:
        pred_ng = _ngrams(pred, n)
        if not pred_ng:
            return 0.0
        gt_ng_counts = Counter(_ngrams(gt, n))
        # add-1 smoothing: add 1 match and 1 to denominator
        hits = sum(min(cnt, gt_ng_counts[ng]) for ng, cnt in Counter(pred_ng).items())
        precision = (hits + 1) / (len(pred_ng) + 1)
        # brevity penalty
        bp = 1.0 if len(pred) >= len(gt) else math.exp(1 - len(gt) / len(pred))
        return bp * precision

    # ── ROUGE-N (F1) ──────────────────────────────────────────────────────────
    def rouge_n(n: int) -> float:
        gt_ng   = Counter(_ngrams(gt, n))
        pred_ng = Counter(_ngrams(pred, n))
        gt_total   = sum(gt_ng.values())
        pred_total = sum(pred_ng.values())
        if gt_total == 0 or pred_total == 0:
            return 0.0
        hits      = sum(min(cnt, gt_ng[ng]) for ng, cnt in pred_ng.items())
        recall    = hits / gt_total
        precision = hits / pred_total
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ── ROUGE-L (LCS-based F1) ────────────────────────────────────────────────
    def lcs_len(a: list, b: list) -> int:
        m, n = len(a), len(b)
        # space-optimised DP (two rows)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]

    def rouge_l() -> float:
        lcs = lcs_len(gt, pred)
        if len(gt) == 0 or len(pred) == 0:
            return 0.0
        recall    = lcs / len(gt)
        precision = lcs / len(pred)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return {
        "bleu1":  round(bleu_n(1), 4),
        "bleu2":  round(bleu_n(2), 4),
        "rouge1": round(rouge_n(1), 4),
        "rouge2": round(rouge_n(2), 4),
        "rougeL": round(rouge_l(), 4),
    }


def aggregate_metrics(per_video: list[dict]) -> dict:
    n = len(per_video)
    if n == 0:
        return {}

    def avg(key: str) -> float:
        return round(sum(v.get(key, 0) for v in per_video) / n, 4)

    return {
        "num_videos":          n,
        "avg_word_accuracy":   avg("word_accuracy"),
        "sentence_accuracy":   avg("sentence_match"),
        "length_match_rate":   avg("length_match"),
        "total_words_gt":      sum(v["gt_len"]        for v in per_video),
        "total_words_correct": sum(v["words_correct"] for v in per_video),
        # NLP metrics
        "avg_bleu1":           avg("bleu1"),
        "avg_bleu2":           avg("bleu2"),
        "avg_rouge1":          avg("rouge1"),
        "avg_rouge2":          avg("rouge2"),
        "avg_rougeL":          avg("rougeL"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth extraction from filename
# ─────────────────────────────────────────────────────────────────────────────

def gt_from_filename(path: Path) -> list[str]:
    """
    e.g. "HELLO_WORLD_HOW_ARE_YOU.mp4"  →  ["HELLO", "WORLD", "HOW", "ARE", "YOU"]
         "0001_HELLO_WORLD.mp4"          →  ["HELLO", "WORLD"]
    Strips the extension, splits on underscore, and drops a leading numeric ID token.
    """
    parts = path.stem.upper().split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    video_dir  = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"run_{ts}.log"
    log      = setup_logger(log_path)

    log.info("=" * 60)
    log.info("ASL Sentence Pipeline Evaluation")
    log.info("=" * 60)
    log.info("video_dir   : %s", video_dir)
    log.info("output_dir  : %s", output_dir)
    log.info("checkpoint  : %s", args.checkpoint)
    log.info("dataset     : %s", args.dataset)
    log.info("pipeline(s) : %s", ", ".join(args.pipeline))

    # ── Discover videos ───────────────────────────────────────────────────────
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        log.error("No .mp4 files found in %s", video_dir)
        return
    log.info("Found %d video(s) in %s", len(videos), video_dir)

    # ── Load model ────────────────────────────────────────────────────────────
    device = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")
    log.info("Device: %s", device)

    log.info("Loading dataset config: %s", args.dataset)
    dataset_cfg = DatasetConfig(args.dataset)
    idx_to_gloss = {}
    for gloss, idx in dataset_cfg.label_map.items():
        idx_to_gloss[idx] = gloss
        try: idx_to_gloss[int(idx)] = gloss
        except (TypeError, ValueError): pass
    log.info("Label map loaded — %d classes", dataset_cfg.num_classes)

    log.info("Loading TRM checkpoint: %s", args.checkpoint)
    model = TRMMicro(num_classes=dataset_cfg.num_classes).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    log.info("Model loaded successfully")

    # ── Pipeline registry ─────────────────────────────────────────────────────
    pipelines = {
        "hand_boundary": lambda kps: pipeline_hand_boundary(
            kps, model, idx_to_gloss, device,
            min_word_frames=args.min_word_frames,
            gap_frames=args.gap_frames,
        ),
        "auto_window": lambda kps: pipeline_auto_window(
            kps, model, idx_to_gloss, device,
            min_word_frames=args.min_word_frames,
            gap_frames=args.gap_frames,
            num_windows=args.auto_num_windows,
            window_size=args.auto_window_size,
        ),
        "sliding_in_boundary": lambda kps: pipeline_sliding_in_boundary(
            kps, model, idx_to_gloss, device,
            min_word_frames=args.min_word_frames,
            gap_frames=args.gap_frames,
            window_size=args.sw_window_size,
            overlap=args.sw_overlap,
        ),
        "sliding_window": lambda kps: pipeline_sliding_window(
            kps, model, idx_to_gloss, device,
            window_size=args.sliding_window_size,
            overlap=args.sliding_overlap,
        ),
    }

    # ── Filter pipelines by --pipeline argument ───────────────────────────────
    selected = set(args.pipeline)
    if "all" not in selected:
        pipelines = {k: v for k, v in pipelines.items() if k in selected}
        if not pipelines:
            log.error("No valid pipelines matched: %s", args.pipeline)
            return

    log.info("Active pipeline(s): %s", ", ".join(pipelines))
    log.debug("Pipeline settings:")
    log.debug("  min_word_frames    = %d", args.min_word_frames)
    log.debug("  gap_frames         = %d", args.gap_frames)
    log.debug("  auto_num_windows   = %d", args.auto_num_windows)
    log.debug("  auto_window_size   = %d", args.auto_window_size)
    log.debug("  sw_window_size     = %d", args.sw_window_size)
    log.debug("  sw_overlap         = %.2f", args.sw_overlap)
    log.debug("  sliding_window_size= %d", args.sliding_window_size)
    log.debug("  sliding_overlap    = %.2f", args.sliding_overlap)

    # ── Output dirs ───────────────────────────────────────────────────────────
    # Single pipeline → write directly into output_dir (no nesting)
    # Multiple pipelines → each pipeline gets output_dir/{pipe_name}/
    multi_pipe = len(pipelines) > 1
    pipe_dirs: dict[str, Path] = {}
    for pipe_name in pipelines:
        d = output_dir / pipe_name if multi_pipe else output_dir
        d.mkdir(parents=True, exist_ok=True)
        pipe_dirs[pipe_name] = d
        log.debug("Output dir [%s] → %s", pipe_name, d)

    # ── Log files (one file per pipeline, in their respective dirs) ───────────
    acc_fieldnames = [
        "pipeline", "video", "gt_sentence", "pred_sentence",
        "gt_len", "pred_len", "words_correct",
        "word_accuracy", "length_match", "sentence_match",
    ]

    pred_log_paths = {p: pipe_dirs[p] / f"prediction_log_{ts}.jsonl" for p in pipelines}
    acc_log_paths  = {p: pipe_dirs[p] / f"accuracy_log_{ts}.csv"     for p in pipelines}

    for pipe_name in pipelines:
        log.info("  [%s] prediction_log → %s", pipe_name, pred_log_paths[pipe_name])
        log.info("  [%s] accuracy_log   → %s", pipe_name, acc_log_paths[pipe_name])
    log.info("Run log        : %s", log_path)
    log.info("-" * 60)

    # per-pipeline accumulator for aggregate stats
    per_pipeline_rows: dict[str, list[dict]] = {k: [] for k in pipelines}

    # open all per-pipeline file handles up front
    # Update fieldnames to include BLEU and ROUGE metrics
    acc_fieldnames_full = acc_fieldnames + ["bleu1", "bleu2", "rouge1", "rouge2", "rougeL"]

    # open all per-pipeline file handles up front
    pred_files = {p: open(pred_log_paths[p], "w", encoding="utf-8")                  for p in pipelines}
    acc_files  = {p: open(acc_log_paths[p],  "w", newline="", encoding="utf-8")      for p in pipelines}
    acc_writers = {}
    for p, f in acc_files.items():
        w = csv.DictWriter(f, fieldnames=acc_fieldnames_full)
        w.writeheader()
        acc_writers[p] = w

    try:
        video_bar = tqdm(videos, desc="Videos", unit="vid")
        for vid_idx, vid_path in enumerate(video_bar, 1):
            gt_words = gt_from_filename(vid_path)
            video_bar.set_postfix(video=vid_path.stem[:30])

            log.info("[%d/%d] %s", vid_idx, len(videos), vid_path.name)
            log.info("  GT  : %s  (%d words)", " ".join(gt_words), len(gt_words))

            # Extract keypoints once for all pipelines
            log.debug("  Extracting keypoints…")
            frames_kp = extract_keypoints_from_video(str(vid_path))
            hand_frames = sum(1 for kp in frames_kp if has_hands(kp))
            log.info("  Frames: %d total, %d with hands (%.0f%%)",
                     len(frames_kp), hand_frames,
                     100 * hand_frames / max(len(frames_kp), 1))

            pipe_bar = tqdm(pipelines.items(), desc="  Pipelines", unit="pipe",
                            leave=False)
            for pipe_name, pipe_fn in pipe_bar:
                pipe_bar.set_postfix(pipeline=pipe_name)
                log.debug("  [%s] running…", pipe_name)

                word_preds = pipe_fn(frames_kp)

                # For sliding_window mode predictions are windows, not words —
                # collapse by majority vote to get a word sequence
                if pipe_name == "sliding_window":
                    pred_words = _collapse_sliding_preds(word_preds)
                    log.debug("  [%s] %d windows → collapsed to: %s",
                              pipe_name, len(word_preds), " ".join(pred_words))
                else:
                    pred_words = [p["top1"] for p in word_preds]
                    for wp in word_preds:
                        log.debug("    word %d | pred=%-15s conf=%5.1f%%  frames=%d  windows=%s",
                                  wp["word_idx"], wp["top1"], wp["conf"],
                                  wp.get("num_frames", 0),
                                  wp.get("num_windows", 1))

                metrics = word_accuracy(gt_words, pred_words)
                nlp_metrics = compute_nlp_metrics(gt_words, pred_words)
                metrics.update(nlp_metrics)

                match_icon = "✓" if metrics["sentence_match"] else "✗"
                log.info("  [%s] %s  PRED: %-40s  word_acc=%s  sent=%s",
                         pipe_name, match_icon,
                         " ".join(pred_words),
                         f"{metrics['word_accuracy']:.2%}",
                         metrics["sentence_match"])
                log.info(
                    "  [%s] NLP  BLEU1=%.4f BLEU2=%.4f ROUGE1=%.4f ROUGE2=%.4f ROUGEL=%.4f",
                    pipe_name,
                    metrics.get("bleu1", 0.0),
                    metrics.get("bleu2", 0.0),
                    metrics.get("rouge1", 0.0),
                    metrics.get("rouge2", 0.0),
                    metrics.get("rougeL", 0.0),
                )

                if not metrics["sentence_match"]:
                    gt_up   = [w.upper() for w in gt_words]
                    pred_up = [w.upper() for w in pred_words]
                    pairs = list(zip(
                        gt_up   + ["—"] * max(0, len(pred_up) - len(gt_up)),
                        pred_up + ["—"] * max(0, len(gt_up) - len(pred_up)),
                    ))
                    for pos, (g, p) in enumerate(pairs, 1):
                        ok = "✓" if g == p else "✗"
                        log.debug("    pos %d: gt=%-15s pred=%-15s %s", pos, g, p, ok)

                # ── Prediction log (JSONL) ───────────────────────────────────
                pred_entry = {
                    "pipeline":      pipe_name,
                    "video":         vid_path.name,
                    "gt_words":      gt_words,
                    "pred_words":    pred_words,
                    "window_detail": word_preds,
                    "metrics":       metrics,
                }
                pred_files[pipe_name].write(json.dumps(pred_entry) + "\n")

                # ── Accuracy log (CSV) ───────────────────────────────────────
                acc_row = {
                    "pipeline":       pipe_name,
                    "video":          vid_path.name,
                    "gt_sentence":    " ".join(gt_words),
                    "pred_sentence":  " ".join(pred_words),
                    **metrics,
                }
                acc_writers[pipe_name].writerow(acc_row)

                per_pipeline_rows[pipe_name].append(metrics)

            log.debug("  %s done", vid_path.name)

    finally:
        for f in pred_files.values(): f.close()
        for f in acc_files.values():  f.close()

    # ── Aggregate summary (always in output_dir root) ─────────────────────────
    summary_path = output_dir / f"summary_{ts}.json"
    summary = {
        pipe: aggregate_metrics(rows)
        for pipe, rows in per_pipeline_rows.items()
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("")
    log.info("=" * 60)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 60)
    for pipe, stats in summary.items():
        log.info("")
        log.info("  Pipeline : %s", pipe)
        log.info("    %-28s %s", "num_videos",          stats.get("num_videos"))
        log.info("    %-28s %s", "avg_word_accuracy",   f"{stats.get('avg_word_accuracy', 0):.2%}")
        log.info("    %-28s %s", "sentence_accuracy",   f"{stats.get('sentence_accuracy', 0):.2%}")
        log.info("    %-28s %s", "length_match_rate",   f"{stats.get('length_match_rate', 0):.2%}")
        log.info("    %-28s %d / %d",
                 "words_correct/total",
                 stats.get("total_words_correct", 0),
                 stats.get("total_words_gt", 0))
        log.info("    %-28s %.4f", "avg_bleu1",          stats.get("avg_bleu1", 0))
        log.info("    %-28s %.4f", "avg_bleu2",          stats.get("avg_bleu2", 0))
        log.info("    %-28s %.4f", "avg_rouge1",         stats.get("avg_rouge1", 0))
        log.info("    %-28s %.4f", "avg_rouge2",         stats.get("avg_rouge2", 0))
        log.info("    %-28s %.4f", "avg_rougeL",         stats.get("avg_rougeL", 0))

    log.info("")
    for pipe_name in pipelines:
        log.info("  [%s] prediction_log → %s", pipe_name, pred_log_paths[pipe_name])
        log.info("  [%s] accuracy_log   → %s", pipe_name, acc_log_paths[pipe_name])
    log.info("Summary        → %s", summary_path)
    log.info("Run log        → %s", log_path)


def _collapse_sliding_preds(window_preds: list[dict]) -> list[str]:
    """
    Collapse sliding-window outputs into a word sequence by majority vote.
    All windows vote on one label; returned as a single-element list.
    (Sentence structure cannot be recovered from a pure sliding window without
    a CTC / boundary decoder — this gives the overall dominant prediction.)
    """
    from collections import Counter
    if not window_preds:
        return []
    counts = Counter(p["top1"] for p in window_preds)
    return [counts.most_common(1)[0][0]]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_CHOICES = ["hand_boundary", "auto_window", "sliding_in_boundary", "sliding_window", "all"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate appv2 sentence pipelines on synthetic videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Pipeline choices
        ----------------
        hand_boundary        One TRM call per hand-present word segment
        auto_window          N evenly-spaced fixed-size windows per segment, avg softmax
        sliding_in_boundary  Fixed sliding window inside each segment, avg softmax
        sliding_window       Original fixed sliding window (no hand filtering)
        all                  Run all four pipelines (default)

        Example commands
        ----------------
        # Run all pipelines (default)
        python -m eval.eval_sentence_pipelines \
            --video_dir  data/testing/sentence-level/sentence-level-stitched \
            --checkpoint checkpoints/asl100/best_model.pt \
            --dataset    asl100 \
            --output_dir eval/results/sentence_pipelines

        # Run only hand_boundary pipeline
        python -m eval.eval_sentence_pipelines \
            --video_dir  data/testing/sentence-level/sentence-level-stitched \
            --checkpoint checkpoints/asl100/best_model.pt \
            --pipeline   hand_boundary \
            --min_word_frames 8 --gap_frames 5

        # Run auto_window with custom settings
        python -m eval.eval_sentence_pipelines \
            --video_dir  data/testing/sentence-level/sentence-level-stitched \
            --checkpoint checkpoints/asl100/best_model.pt \
            --pipeline   auto_window \
            --auto_num_windows 3 --auto_window_size 20 \
            --min_word_frames 8 --gap_frames 5

        # Run sliding_in_boundary
        python -m eval.eval_sentence_pipelines \
            --video_dir  D:\self\projects\masters\capstone\data\testing\asl-100-citizen\new_videos \
            --checkpoint checkpoints/asl100/best_model.pt \
            --pipeline   sliding_in_boundary \
            --sw_window_size 30 --sw_overlap 0.5 \
            --min_word_frames 8 --gap_frames 5 \
            --output_dir eval/results/sentence_pipelines

        # Run sliding_window (no hand filtering)
        python -m eval.eval_sentence_pipelines \
            --video_dir  data/testing/sentence-level/sentence-level-stitched \
            --checkpoint checkpoints/asl100/best_model.pt \
            --pipeline   sliding_window \
            --sliding_window_size 50 --sliding_overlap 0.20

        # Run two specific pipelines together
        python -m eval.eval_sentence_pipelines \
            --video_dir  data/testing/sentence-level/sentence-level-stitched \
            --checkpoint checkpoints/asl100/best_model.pt \
            --pipeline   hand_boundary auto_window \
            --min_word_frames 8 --gap_frames 5 \
            --auto_num_windows 3 --auto_window_size 20
                """
    )
    p.add_argument("--video_dir",    required=True,
                   help="Directory of synthetic sentence MP4s (filename = ground truth)")
    p.add_argument("--checkpoint",   required=True,
                   help="Path to TRM checkpoint (.pt)")
    p.add_argument("--dataset",      default="asl100",
                   help="Dataset name for label map (default: asl100)")
    p.add_argument("--output_dir",   default="eval/results/sentence_pipelines",
                   help="Directory to write prediction_log and accuracy_log")
    p.add_argument("--pipeline", nargs="+", default=["all"],
                   choices=PIPELINE_CHOICES, metavar="PIPELINE",
                   help=(
                       "One or more pipelines to run: "
                       "hand_boundary | auto_window | sliding_in_boundary | sliding_window | all  "
                       "(default: all)"
                   ))

    # Hand boundary shared settings
    p.add_argument("--min_word_frames", type=int, default=8,
                   help="Min hand-present frames per word segment (default: 8)")
    p.add_argument("--gap_frames",      type=int, default=5,
                   help="Consecutive no-hand frames to declare word boundary (default: 5)")

    # Auto window settings
    p.add_argument("--auto_num_windows", type=int,   default=3,
                   help="Number of windows per word for auto_window pipeline (default: 3)")
    p.add_argument("--auto_window_size", type=int,   default=20,
                   help="Fixed window size (frames) for auto_window pipeline (default: 20)")

    # Sliding-in-boundary settings
    p.add_argument("--sw_window_size",   type=int,   default=30,
                   help="Window size for sliding_in_boundary pipeline (default: 30)")
    p.add_argument("--sw_overlap",       type=float, default=0.5,
                   help="Overlap fraction for sliding_in_boundary pipeline (default: 0.5)")

    # Pure sliding window settings
    p.add_argument("--sliding_window_size", type=int,   default=50,
                   help="Window size for sliding_window pipeline (default: 50)")
    p.add_argument("--sliding_overlap",     type=float, default=0.20,
                   help="Overlap fraction for sliding_window pipeline (default: 0.20)")

    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
