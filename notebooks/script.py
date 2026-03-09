import os
import json
import cv2
import numpy as np
import random
from pathlib import Path
import pandas as pd

test_csv_path = r"D:\self\projects\masters\capstone\git\asl_trm\data\app\val_features.csv"

word_df = pd.read_csv(test_csv_path)
word_df["video_names"] = word_df["video_path"].apply(lambda x: x.split("/")[-1])

video_dir = r"D:\self\projects\masters\capstone\data\testing\asl-100-citizen\new_videos"
# filtered video_paths based on val_df

val_video_names = word_df["video_names"].unique()

all_video_paths = os.listdir(video_dir)
print(f"""Total videos: {len(all_video_paths)}""") # 
filtered_video_paths = [video for video in all_video_paths if video in val_video_names]
print(f"""Filtered videos: {len(filtered_video_paths)}""")

# -----------------------------
# Configuration
# -----------------------------
SENTENCE_JSON = "./app_asl_sentences.json"

OUTPUT_DIR = r"D:\self\projects\masters\capstone\data\testing\sentence-level\sentence-level-stitched"
VIDEO_DIR = r"D:\self\projects\masters\capstone\data\testing\asl-100-citizen\new_videos"

BASE_FPS = 30

MIN_PAUSE = 5
MAX_PAUSE = 20

MIN_SPEED = 0.7
MAX_SPEED = 1.3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all available videos
filtered_video_paths = os.listdir(VIDEO_DIR)

# -----------------------------
# Helper Functions
# -----------------------------
def read_video_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return frames, width, height


def create_blank_frames(width, height, count):

    blank = np.zeros((height, width, 3), dtype=np.uint8)

    return [blank.copy() for _ in range(count)]


def apply_speed_variation(frames):

    speed = random.uniform(MIN_SPEED, MAX_SPEED)

    if speed > 1:
        step = max(1, int(speed))
        frames = frames[::step]

    else:
        repeat = max(1, int(1 / speed))
        new_frames = []
        for f in frames:
            new_frames.extend([f] * repeat)
        frames = new_frames

    return frames


# -----------------------------
# CV2 Video Writer (Streamlit-compatible)
# -----------------------------
def write_video_cv2(frames, output_path, fps):
    """Write frames to MP4 using H.264 (avc1) for browser/Streamlit playback.
    Falls back to mp4v if avc1 is unavailable on the system."""
    height, width, _ = frames[0].shape

    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            break
        writer.release()

    for frame in frames:
        writer.write(frame)

    writer.release()


# -----------------------------
# Load Dataset
# -----------------------------
with open(SENTENCE_JSON) as f:
    sentences = json.load(f)["sentences"]


# -----------------------------
# Generate Videos
# -----------------------------
for sentence in sentences:

    asl = sentence["asl"]
    print(f"\nProcessing: {asl}")

    words = asl.split()

    sentence_frames = []
    width = None
    height = None

    for i, word in enumerate(words):

        print(f"Processing word: {word.upper()}")

        # get videos related to word
        word_videos = [
            video for video in filtered_video_paths
            if f"-{word.upper()}.mp4" in video
        ]

        if not word_videos:
            print(f"Missing video for: {word}")
            break

        # randomly select video
        video_name = random.choice(word_videos)
        video_path = os.path.join(VIDEO_DIR, video_name)

        print(f"Selected video: {video_path}")

        frames, width, height = read_video_frames(video_path)

        # speed variation
        frames = apply_speed_variation(frames)

        sentence_frames.extend(frames)

        # random pause
        if i < len(words) - 1:

            pause = random.randint(MIN_PAUSE, MAX_PAUSE)

            blanks = create_blank_frames(width, height, pause)

            sentence_frames.extend(blanks)

    if not sentence_frames:
        continue

    output_name = asl.replace(" ", "_") + ".mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    write_video_cv2(sentence_frames, output_path, BASE_FPS)

    print(f"Saved: {output_path}")

    # break