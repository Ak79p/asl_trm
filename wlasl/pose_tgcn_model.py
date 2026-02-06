#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np


import json
import torch
import numpy as np

class PoseTGCNInference:
    def __init__(self, model_path, mapping_path, num_samples=25):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        # Note: If the model was saved as a state_dict, you'll need to instantiate 
        # the class first: model = PoseTGCN(); model.load_state_dict(torch.load(...))
        # num_samples = 50
        hidden_feature = 64 
        drop_p   = 0.3
        num_class = 100
        num_stages = 20
        self.model  = GCN_muti_att(input_feature=num_samples*2, hidden_feature=hidden_feature,
                         num_class=num_class, p_dropout=drop_p, num_stage=num_stages).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        # self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Load the JSON mapping { "go": 0, "happy": 1 }
        with open(mapping_path, 'r') as f:
            self.class_map = json.load(f)
        
        # Create inverse mapping { 0: "go", 1: "happy" }
        self.idx_to_gloss = {v: k for k, v in self.class_map.items()}
        
        self.num_samples = num_samples
        self.body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

    def preprocess(self, sequence_buffer):
        """
        Mimics the 'read_pose_file' logic from your dataset class.
        sequence_buffer: list of 25 frames, each containing 137 keypoints.
        """
        processed_frames = []

        for frame_data in sequence_buffer:
            # 1. Flatten into the list format OpenPose uses [x,y,c, x,y,c...]
            full_pose = []
            full_pose.extend(frame_data['pose'].flatten())
            full_pose.extend(frame_data['left_hand'].flatten())
            full_pose.extend(frame_data['right_hand'].flatten())

            # 2. Extract X and Y while excluding lower body indices
            # i // 3 gives the joint index, i % 3 identifies x (0) or y (1)
            x_raw = [v for i, v in enumerate(full_pose) if i % 3 == 0 and i // 3 not in self.body_pose_exclude]
            y_raw = [v for i, v in enumerate(full_pose) if i % 3 == 1 and i // 3 not in self.body_pose_exclude]

            # 3. WLASL Normalization to [-1, 1] range using 256 as base
            x_norm = 2 * ((torch.FloatTensor(x_raw) / 256.0) - 0.5)
            y_norm = 2 * ((torch.FloatTensor(y_raw) / 256.0) - 0.5)

            # 4. Transpose to (N, 2)
            xy = torch.stack([x_norm, y_norm]).transpose(0, 1)
            processed_frames.append(xy)

        # 5. Temporal Concatenation: (N, T*2)
        # This matches: poses_across_time = torch.cat(poses, dim=1)
        input_tensor = torch.cat(processed_frames, dim=1)
        
        # Add batch dimension: (1, N, T*2)
        return input_tensor.unsqueeze(0).to(self.device)

    def predict(self, sequence_buffer):
        if len(sequence_buffer) < self.num_samples:
            return None, None

        input_data = self.preprocess(sequence_buffer)

        with torch.no_grad():
            logits = self.model(input_data)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        return self.idx_to_gloss.get(pred_idx, "Unknown"), confidence

class GraphConvolution_att(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init_A=0):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(55, 55))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # AHW
        support = torch.matmul(input, self.weight)  # HW
        output = torch.matmul(self.att, support)  # g
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):

    def __init__(self, in_features, p_dropout, bias=True, is_resi=True):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi

        self.gc1 = GraphConvolution_att(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)

        self.gc2 = GraphConvolution_att(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        if self.is_resi:
            return y + x
        else:
            return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_muti_att(nn.Module):
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage=1, is_resi=True):
        super(GCN_muti_att, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(55 * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi))

        self.gcbs = nn.ModuleList(self.gcbs)

        # self.gc7 = GraphConvolution_att(hidden_feature, output_feature)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        # self.fc1 = nn.Linear(55 * output_feature, fc1_out)
        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        # y = self.gc7(y)
        out = torch.mean(y, dim=1)
        out = self.fc_out(out)

        return out


# if __name__ == '__main__':
#     num_samples = 32

#     model = GCN_muti_att(input_feature=num_samples*2, hidden_feature=256,
#                          num_class=100, p_dropout=0.3, num_stage=2)
#     x = torch.ones([2, 55, num_samples*2])
#     print(model(x).size())