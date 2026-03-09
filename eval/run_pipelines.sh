#!/usr/bin/env bash
# eval/run_pipelines.sh
# Run one or more evaluation pipelines.
#
# Usage:
#   bash eval/run_pipelines.sh [OPTIONS]
#
# Options:
#   -p, --pipeline    Pipeline(s) to run: all | hand_boundary | auto_window |
#                     sliding_in_boundary | sliding_window  (default: all)
#                     Repeat flag for multiple: -p hand_boundary -p auto_window
#   --video_dir       Path to sentence videos        (default: data/testing/sentence-level/sentence-level-stitched)
#   --checkpoint      Path to model checkpoint       (default: checkpoints/app/best_model.pt)
#   --dataset         Dataset name                   (default: asl100)
#   --out_base        Base output directory          (default: eval/results/sentence_pipelines)
#   --min_word_frames Min hand-present frames/word   (default: 8)
#   --gap_frames      No-hand frames for boundary    (default: 5)
#   --auto_num_windows  Windows per word (auto)      (default: 3)
#   --auto_window_size  Window size frames (auto)    (default: 20)
#   --sw_window_size  Window size (sliding_in_boundary)  (default: 30)
#   --sw_overlap      Overlap (sliding_in_boundary)      (default: 0.5)
#   --sliding_window_size  Window size (sliding_window)  (default: 50)
#   --sliding_overlap      Overlap (sliding_window)      (default: 0.20)
#
# Examples:
#   bash eval/run_pipelines.sh
#   bash eval/run_pipelines.sh -p hand_boundary
#   bash eval/run_pipelines.sh -p hand_boundary -p auto_window
#   bash eval/run_pipelines.sh -p auto_window --auto_num_windows 5 --auto_window_size 24
#   bash eval/run_pipelines.sh --checkpoint checkpoints/exp2/best_model.pt --out_base eval/results/exp2

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
VIDEO_DIR="D:\self\projects\masters\capstone\data\testing\asl-100-citizen\new_videos"
CHECKPOINT="checkpoints/app/best_model.pt"
DATASET="asl100"
OUT_BASE="D:\self\projects\masters\capstone\git\asl_trm\eval\results\sentence_pipelines"
MIN_WORD_FRAMES=8
GAP_FRAMES=5
AUTO_NUM_WINDOWS=3
AUTO_WINDOW_SIZE=20
SW_WINDOW_SIZE=30
SW_OVERLAP=0.5
SLIDING_WINDOW_SIZE=50
SLIDING_OVERLAP=0.20
PIPELINES=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--pipeline)         PIPELINES+=("$2");       shift 2 ;;
        --video_dir)           VIDEO_DIR="$2";          shift 2 ;;
        --checkpoint)          CHECKPOINT="$2";         shift 2 ;;
        --dataset)             DATASET="$2";            shift 2 ;;
        --out_base)            OUT_BASE="$2";           shift 2 ;;
        --min_word_frames)     MIN_WORD_FRAMES="$2";    shift 2 ;;
        --gap_frames)          GAP_FRAMES="$2";         shift 2 ;;
        --auto_num_windows)    AUTO_NUM_WINDOWS="$2";   shift 2 ;;
        --auto_window_size)    AUTO_WINDOW_SIZE="$2";   shift 2 ;;
        --sw_window_size)      SW_WINDOW_SIZE="$2";     shift 2 ;;
        --sw_overlap)          SW_OVERLAP="$2";         shift 2 ;;
        --sliding_window_size) SLIDING_WINDOW_SIZE="$2"; shift 2 ;;
        --sliding_overlap)     SLIDING_OVERLAP="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default to all if none specified
if [[ ${#PIPELINES[@]} -eq 0 ]]; then
    PIPELINES=("all")
fi

# Expand "all"
ALL_PIPELINES=(hand_boundary auto_window sliding_in_boundary sliding_window)
if [[ " ${PIPELINES[*]} " == *" all "* ]]; then
    PIPELINES=("${ALL_PIPELINES[@]}")
fi

# ── Base args shared by every pipeline ───────────────────────────────────────
BASE_ARGS=(
    -m eval.eval_sentence_pipelines
    --video_dir  "$VIDEO_DIR"
    --checkpoint "$CHECKPOINT"
    --dataset    "$DATASET"
)

# ── Runner ────────────────────────────────────────────────────────────────────
run_pipeline() {
    local name="$1"
    echo ""
    echo "=== Running: $name ==="
    case "$name" in
        hand_boundary)
            uv run python "${BASE_ARGS[@]}" \
                --pipeline   hand_boundary \
                --min_word_frames "$MIN_WORD_FRAMES" \
                --gap_frames      "$GAP_FRAMES" \
                --output_dir      "$OUT_BASE/hand_boundary"
            ;;
        auto_window)
            uv run python "${BASE_ARGS[@]}" \
                --pipeline        auto_window \
                --min_word_frames "$MIN_WORD_FRAMES" \
                --gap_frames      "$GAP_FRAMES" \
                --auto_num_windows "$AUTO_NUM_WINDOWS" \
                --auto_window_size "$AUTO_WINDOW_SIZE" \
                --output_dir      "$OUT_BASE/auto_window"
            ;;
        sliding_in_boundary)
            uv run python "${BASE_ARGS[@]}" \
                --pipeline        sliding_in_boundary \
                --min_word_frames "$MIN_WORD_FRAMES" \
                --gap_frames      "$GAP_FRAMES" \
                --sw_window_size  "$SW_WINDOW_SIZE" \
                --sw_overlap      "$SW_OVERLAP" \
                --output_dir      "$OUT_BASE/sliding_in_boundary"
            ;;
        sliding_window)
            uv run python "${BASE_ARGS[@]}" \
                --pipeline            sliding_window \
                --sliding_window_size "$SLIDING_WINDOW_SIZE" \
                --sliding_overlap     "$SLIDING_OVERLAP" \
                --output_dir          "$OUT_BASE/sliding_window"
            ;;
        *)
            echo "ERROR: Unknown pipeline '$name'. Valid: ${ALL_PIPELINES[*]}"
            exit 1
            ;;
    esac
}

for pipeline in "${PIPELINES[@]}"; do
    run_pipeline "$pipeline"
done

echo ""
echo "Done."
