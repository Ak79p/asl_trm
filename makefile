VIDEO_DIR            ?= D:\self\projects\masters\capstone\data\testing\asl-100-citizen\new_videos
CHECKPOINT           ?= checkpoints/app/best_model.pt
DATASET              ?= asl100
OUT_BASE             ?= eval/results/sentence_pipelines
MIN_WORD_FRAMES      ?= 8
GAP_FRAMES           ?= 5
AUTO_NUM_WINDOWS     ?= 3
AUTO_WINDOW_SIZE     ?= 20
SW_WINDOW_SIZE       ?= 30
SW_OVERLAP           ?= 0.5
SLIDING_WINDOW_SIZE  ?= 50
SLIDING_OVERLAP      ?= 0.20

BASE_ARGS = \
	--video_dir  $(VIDEO_DIR) \
	--checkpoint $(CHECKPOINT) \
	--dataset    $(DATASET)

# ── Run all pipelines ─────────────────────────────────────────────────────────
eval_all:
	bash eval/run_pipelines.sh \
	    --video_dir            $(VIDEO_DIR) \
	    --checkpoint           $(CHECKPOINT) \
	    --dataset              $(DATASET) \
	    --out_base             $(OUT_BASE) \
	    --min_word_frames      $(MIN_WORD_FRAMES) \
	    --gap_frames           $(GAP_FRAMES) \
	    --auto_num_windows     $(AUTO_NUM_WINDOWS) \
	    --auto_window_size     $(AUTO_WINDOW_SIZE) \
	    --sw_window_size       $(SW_WINDOW_SIZE) \
	    --sw_overlap           $(SW_OVERLAP) \
	    --sliding_window_size  $(SLIDING_WINDOW_SIZE) \
	    --sliding_overlap      $(SLIDING_OVERLAP)

# ── Hand boundary ─────────────────────────────────────────────────────────────
eval_hand_boundary:
	bash eval/run_pipelines.sh -p hand_boundary \
	    --video_dir        $(VIDEO_DIR) \
	    --checkpoint       $(CHECKPOINT) \
	    --dataset          $(DATASET) \
	    --out_base         $(OUT_BASE) \
	    --min_word_frames  $(MIN_WORD_FRAMES) \
	    --gap_frames       $(GAP_FRAMES)

# ── Auto window ───────────────────────────────────────────────────────────────
eval_auto_window:
	bash eval/run_pipelines.sh -p auto_window \
	    --video_dir         $(VIDEO_DIR) \
	    --checkpoint        $(CHECKPOINT) \
	    --dataset           $(DATASET) \
	    --out_base          $(OUT_BASE) \
	    --min_word_frames   $(MIN_WORD_FRAMES) \
	    --gap_frames        $(GAP_FRAMES) \
	    --auto_num_windows  $(AUTO_NUM_WINDOWS) \
	    --auto_window_size  $(AUTO_WINDOW_SIZE)

# ── Sliding window in boundary ────────────────────────────────────────────────
eval_sliding_in_boundary:
	bash eval/run_pipelines.sh -p sliding_in_boundary \
	    --video_dir        $(VIDEO_DIR) \
	    --checkpoint       $(CHECKPOINT) \
	    --dataset          $(DATASET) \
	    --out_base         $(OUT_BASE) \
	    --min_word_frames  $(MIN_WORD_FRAMES) \
	    --gap_frames       $(GAP_FRAMES) \
	    --sw_window_size   $(SW_WINDOW_SIZE) \
	    --sw_overlap       $(SW_OVERLAP)

# ── Pure sliding window (no hand filtering) ───────────────────────────────────
eval_sliding_window:
	bash eval/run_pipelines.sh -p sliding_window \
	    --video_dir            $(VIDEO_DIR) \
	    --checkpoint           $(CHECKPOINT) \
	    --dataset              $(DATASET) \
	    --out_base             $(OUT_BASE) \
	    --sliding_window_size  $(SLIDING_WINDOW_SIZE) \
	    --sliding_overlap      $(SLIDING_OVERLAP)
