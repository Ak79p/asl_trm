import json
import math
import os
import random

import numpy as np

import cv2
import torch
import torch.nn as nn

import utils

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# def compute_difference(x):
#     diff = []

#     for i, xx in enumerate(x):
#         temp = []
#         for j, xxx in enumerate(x):
#             if i != j:
#                 temp.append(xx - xxx)

#         diff.append(temp)

#     return diff


# def read_pose_file(filepath):
#     body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

#     try:
#         content = json.load(open(filepath))["people"][0]
#     except IndexError:
#         return None

#     path_parts = os.path.split(filepath)

#     frame_id = path_parts[1][:11]
#     vid = os.path.split(path_parts[0])[-1]

#     save_to = os.path.join('/home/dxli/workspace/nslt/code/Pose-GCN/posegcn/features', vid)

#     try:
#         ft = torch.load(os.path.join(save_to, frame_id + '_ft.pt'))

#         xy = ft[:, :2]
#         # angles = torch.atan(ft[:, 110:]) / 90
#         # ft = torch.cat([xy, angles], dim=1)
#         return xy

#     except FileNotFoundError:
#         print(filepath)
#         body_pose = content["pose_keypoints_2d"]
#         left_hand_pose = content["hand_left_keypoints_2d"]
#         right_hand_pose = content["hand_right_keypoints_2d"]

#         body_pose.extend(left_hand_pose)
#         body_pose.extend(right_hand_pose)

#         x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
#         y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]
#         # conf = [v for i, v in enumerate(body_pose) if i % 3 == 2 and i // 3 not in body_pose_exclude]

#         x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
#         y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)
#         # conf = torch.FloatTensor(conf)

#         x_diff = torch.FloatTensor(compute_difference(x)) / 2
#         y_diff = torch.FloatTensor(compute_difference(y)) / 2

#         zero_indices = (x_diff == 0).nonzero()

#         orient = y_diff / x_diff
#         orient[zero_indices] = 0

#         xy = torch.stack([x, y]).transpose_(0, 1)

#         ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

#         path_parts = os.path.split(filepath)

#         frame_id = path_parts[1][:11]
#         vid = os.path.split(path_parts[0])[-1]

#         save_to = os.path.join('code/Pose-GCN/posegcn/features', vid)
#         if not os.path.exists(save_to):
#             os.mkdir(save_to)
#         torch.save(ft, os.path.join(save_to, frame_id + '_ft.pt'))

#         xy = ft[:, :2]
#         # angles = torch.atan(ft[:, 110:]) / 90
#         # ft = torch.cat([xy, angles], dim=1)
#         #
#         return xy

#     # return ft

def compute_difference(x):
    diff = []
    for i, xx in enumerate(x):
        temp = []
        for j, xxx in enumerate(x):
            if i != j:
                temp.append(xx - xxx)
        diff.append(temp)
    return diff


def read_pose_file(filepath, cache_root="code/Pose-GCN/posegcn/features", image_size=256.0):
    """
    Reads a single per-frame keypoints json.

    Expected input (OpenPose-like):
      {
        "version": 1.3,
        "people": [{
          "pose_keypoints_2d": [x,y,conf, x,y,conf, ...]  # 25*3 = 75
          "hand_left_keypoints_2d": [x,y,conf, ...]       # 21*3 = 63
          "hand_right_keypoints_2d": [x,y,conf, ...]      # 21*3 = 63
        }]
      }

    Returns:
      xy: torch.FloatTensor of shape (55, 2) after excluding some body joints.
    """
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

    # ---- Parse JSON safely ----
    try:
        j = json.load(open(filepath, "r"))
    except Exception:
        return None

    people = j.get("people", [])
    if not people:
        return None

    content = people[0]

    pose2d = content.get("pose_keypoints_2d", [])
    lhand2d = content.get("hand_left_keypoints_2d", [])
    rhand2d = content.get("hand_right_keypoints_2d", [])

    # Must have something meaningful
    if len(pose2d) == 0 and len(lhand2d) == 0 and len(rhand2d) == 0:
        return None

    # ---- Derive vid + frame_id from filepath ----
    # e.g. .../<video_id>/image_00008_keypoints.json
    path_dir, filename = os.path.split(filepath)
    vid = os.path.basename(path_dir)
    frame_id = filename[:11]  # "image_00008" (matches original code)

    # ---- Cache path (NO hardcoded /home/...) ----
    save_to = os.path.join(cache_root, vid)
    os.makedirs(save_to, exist_ok=True)
    cache_path = os.path.join(save_to, frame_id + "_ft.pt")

    # ---- Use cache if exists ----
    if os.path.exists(cache_path):
        ft = torch.load(cache_path)
        return ft[:, :2]  # xy only

    # ---- Build combined list like original code ----
    body_pose = []
    body_pose.extend(pose2d)
    body_pose.extend(lhand2d)
    body_pose.extend(rhand2d)

    # If pose is from MediaPipe normalized (0..1), scale to pixel space first
    # We'll inspect max of x/y entries; if <= ~2 then assume normalized.
    xs_raw = [v for i, v in enumerate(body_pose) if i % 3 == 0]
    ys_raw = [v for i, v in enumerate(body_pose) if i % 3 == 1]
    if xs_raw and ys_raw:
        mx = max(xs_raw)
        my = max(ys_raw)
        if mx <= 2.0 and my <= 2.0:
            # normalized coords → pixel coords
            for i in range(0, len(body_pose), 3):
                body_pose[i]     = float(body_pose[i])     * image_size  # x
                body_pose[i + 1] = float(body_pose[i + 1]) * image_size  # y

    # ---- Extract x,y skipping excluded body joints ----
    x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
    y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]

    if len(x) == 0 or len(y) == 0:
        return None

    # ---- Normalize exactly like original code (assumes image_size=256) ----
    x = 2 * ((torch.FloatTensor(x) / image_size) - 0.5)
    y = 2 * ((torch.FloatTensor(y) / image_size) - 0.5)

    x_diff = torch.FloatTensor(compute_difference(x)) / 2
    y_diff = torch.FloatTensor(compute_difference(y)) / 2

    # avoid divide-by-zero
    zero_indices = (x_diff == 0).nonzero(as_tuple=False)
    orient = y_diff / x_diff
    orient[zero_indices[:, 0], zero_indices[:, 1]] = 0

    xy = torch.stack([x, y]).transpose_(0, 1)   # (55,2)
    ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

    # cache it
    torch.save(ft, cache_path)

    return xy


class Sign_Dataset(Dataset):
    def __init__(self, index_file_path, split, pose_root, sample_strategy='rnd_start', num_samples=25, num_copies=4,
                 img_transforms=None, video_transforms=None, test_index_file=None):
        assert os.path.exists(index_file_path), "Non-existent indexing file path: {}.".format(index_file_path)
        assert os.path.exists(pose_root), "Path to poses does not exist: {}.".format(pose_root)

        self.data = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        if type(split) == 'str':
            split = [split]

        self.test_index_file = test_index_file
        self._make_dataset(index_file_path, split)

        self.index_file_path = index_file_path
        self.pose_root = pose_root
        self.framename = 'image_{}_keypoints.json'
        self.sample_strategy = sample_strategy
        self.num_samples = num_samples

        self.img_transforms = img_transforms
        self.video_transforms = video_transforms

        self.num_copies = num_copies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, gloss_cat, frame_start, frame_end = self.data[index]
        # frames of dimensions (T, H, W, C)
        x = self._load_poses(video_id, frame_start, frame_end, self.sample_strategy, self.num_samples)

        if self.video_transforms:
            x = self.video_transforms(x)

        y = gloss_cat

        return x, y, video_id

    def _make_dataset(self, index_file_path, split):
        with open(index_file_path, 'r') as f:
            content = json.load(f)

        # create label encoder
        glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])

        self.label_encoder.fit(glosses)
        self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))

        if self.test_index_file is not None:
            print('Trained on {}, tested on {}'.format(index_file_path, self.test_index_file))
            with open(self.test_index_file, 'r') as f:
                content = json.load(f)

        # make dataset
        for gloss_entry in content:
            gloss, instances = gloss_entry['gloss'], gloss_entry['instances']
            gloss_cat = utils.labels2cat(self.label_encoder, [gloss])[0]

            for instance in instances:
                if instance['split'] not in split:
                    continue

                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']

                instance_entry = video_id, gloss_cat, frame_start, frame_end
                self.data.append(instance_entry)

    # def _load_poses(self, video_id, frame_start, frame_end, sample_strategy, num_samples):
    #     """ Load frames of a video. Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
    #      """
    #     poses = []

    #     if sample_strategy == 'rnd_start':
    #         frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
    #     elif sample_strategy == 'seq':
    #         frames_to_sample = sequential_sampling(frame_start, frame_end, num_samples)
    #     elif sample_strategy == 'k_copies':
    #         frames_to_sample = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples,
    #                                                                      self.num_copies)
    #     else:
    #         raise NotImplementedError('Unimplemented sample strategy found: {}.'.format(sample_strategy))

    #     for i in frames_to_sample:
    #         pose_path = os.path.join(self.pose_root, video_id, self.framename.format(str(i).zfill(5)))
    #         # pose = cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
    #         pose = read_pose_file(pose_path)

    #         if pose is not None:
    #             if self.img_transforms:
    #                 pose = self.img_transforms(pose)

    #             poses.append(pose)
    #         else:
    #             try:
    #                 poses.append(poses[-1])
    #             except IndexError:
    #                 print(pose_path)

    #     pad = None

    #     # if len(frames_to_sample) < num_samples:
    #     if len(poses) < num_samples:
    #         num_padding = num_samples - len(frames_to_sample)
    #         last_pose = poses[-1]
    #         pad = last_pose.repeat(1, num_padding)

    #     poses_across_time = torch.cat(poses, dim=1)
    #     if pad is not None:
    #         poses_across_time = torch.cat([poses_across_time, pad], dim=1)

    #     return poses_across_time

    def _load_poses(self, video_id, frame_start, frame_end, sample_strategy, num_samples):
        poses = []

        if sample_strategy == 'rnd_start':
            frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'seq':
            frames_to_sample = sequential_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'k_copies':
            frames_to_sample = k_copies_fixed_length_sequential_sampling(
                frame_start, frame_end, num_samples, self.num_copies
            )
        else:
            raise NotImplementedError(f"Unimplemented sample strategy: {sample_strategy}")

        # fallback pose if first frames are missing
        fallback_pose = None

        for i in frames_to_sample:
            pose_path = os.path.join(self.pose_root, video_id, self.framename.format(str(i).zfill(5)))
            pose = read_pose_file(pose_path)

            if pose is not None:
                if self.img_transforms:
                    pose = self.img_transforms(pose)
                poses.append(pose)
                if fallback_pose is None:
                    fallback_pose = pose
            else:
                # missing frame → repeat last known pose if exists
                if poses:
                    poses.append(poses[-1])
                elif fallback_pose is not None:
                    poses.append(fallback_pose)
                else:
                    # still nothing yet → create a zero pose shaped like (55,2)
                    # 55 is what your exclusion set yields (25+21+21 - 12 excluded body)
                    poses.append(torch.zeros((55, 2), dtype=torch.float32))
                    fallback_pose = poses[-1]

        # Now we should always have exactly num_samples entries
        if len(poses) != num_samples:
            poses = poses[:num_samples]
            while len(poses) < num_samples:
                poses.append(poses[-1])

        poses_across_time = torch.cat(poses, dim=1)  # (55, 2*num_samples)
        return poses_across_time


def rand_start_sampling(frame_start, frame_end, num_samples):
    """Randomly select a starting point and return the continuous ${num_samples} frames."""
    num_frames = frame_end - frame_start + 1

    if num_frames > num_samples:
        select_from = range(frame_start, frame_end - num_samples + 1)
        sample_start = random.choice(select_from)
        frames_to_sample = list(range(sample_start, sample_start + num_samples))
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def sequential_sampling(frame_start, frame_end, num_samples):
    """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []
    if num_frames > num_samples:
        frames_skip = set()

        num_skips = num_frames - num_samples
        interval = num_frames // num_skips

        for i in range(frame_start, frame_end + 1):
            if i % interval == 0 and len(frames_skip) <= num_skips:
                frames_skip.add(i)

        for i in range(frame_start, frame_end + 1):
            if i not in frames_skip:
                frames_to_sample.append(i)
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []

    if num_frames <= num_samples:
        num_pads = num_samples - num_frames

        frames_to_sample = list(range(frame_start, frame_end + 1))
        frames_to_sample.extend([frame_end] * num_pads)

        frames_to_sample *= num_copies

    elif num_samples * num_copies < num_frames:
        mid = (frame_start + frame_end) // 2
        half = num_samples * num_copies // 2

        frame_start = mid - half

        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * num_samples,
                                               frame_start + i * num_samples + num_samples)))

    else:
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * stride,
                                               frame_start + i * stride + num_samples)))

    return frames_to_sample


if __name__ == '__main__':
    # root = '/home/dxli/workspace/nslt'
    #
    # split_file = os.path.join(root, 'data/splits-with-dialect-annotated/asl100.json')
    # pose_data_root = os.path.join(root, 'data/pose/pose_per_individual_videos')
    #
    # num_samples = 64
    #
    # train_dataset = Sign_Dataset(index_file_path=split_file, split=['train', 'val'], pose_root=pose_data_root,
    #                              img_transforms=None, video_transforms=None,
    #                              num_samples=num_samples)
    #
    # train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    #
    # cnt = 0
    # for batch_idx, data in enumerate(train_data_loader):
    #     print(batch_idx)
    #     x = data[0]
    #     y = data[1]
    #     print(x.size())
    #     print(y.size())

    print(k_copies_fixed_length_sequential_sampling(0, 2, 20, num_copies=3))
