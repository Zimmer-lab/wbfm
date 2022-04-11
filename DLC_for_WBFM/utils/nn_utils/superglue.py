# FROM: https://github.com/HeatherJiaZG/SuperGlue-pytorch/blob/master/models/superglue.py
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_itertools import random_combination
from DLC_for_WBFM.utils.external.utils_pandas import df_to_matches
from DLC_for_WBFM.utils.neuron_matching.class_reference_frame import ReferenceFrame
from DLC_for_WBFM.utils.nn_utils.data_loading import AbstractNeuronImageFeaturesFromProject
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints_3d(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, depth, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * depth, one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        # self.encoder = MLP([3] + layers + [feature_dim])
        # NEW: 3d keypoints
        self.encoder = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(2, 3), scores.unsqueeze(2)]
        return self.encoder(torch.squeeze(torch.cat(inputs, dim=2), dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 128,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        # bin_score = torch.nn.Parameter(torch.tensor(1.))
        # self.register_parameter('bin_score', bin_score)

        self.bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.loss_epsilon = 1e-6

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # all_matches = data['all_matches'].permute(1, 2, 0)  # shape=torch.Size([1, 87, 2])
        # Ground truth
        all_matches = data['all_matches']  # shape=torch.Size([1, 87, 2])

        scores = self.calculate_match_scores(data)
        indices0, indices1, mscores0, mscores1 = self.process_scores_into_matches(scores)

        # check if indexed correctly
        # Note: if a keypoint doesn't have a match in the gt, then it is not penalized here by default
        loss = []
        for i in range(len(all_matches[0])):
            x = all_matches[0][i][0]
            y = all_matches[0][i][1]
            loss.append(-torch.log(scores[0][x][y].exp() + self.loss_epsilon))  # check batch size == 1 ?
        # This penalizes matches that should be unmatched, and assumes the gt is complete
        # for p0 in unmatched0:
        #     loss += -torch.log(scores[0][p0][-1])
        # for p1 in unmatched1:
        #     loss += -torch.log(scores[0][-1][p1])
        raw_loss = torch.stack(loss)
        loss_mean = torch.mean(raw_loss)
        loss_mean = torch.reshape(loss_mean, (1, -1))
        return {
            'matches0': indices0[0],  # use -1 for invalid match
            'matches1': indices1[0],  # use -1 for invalid match
            'matching_scores0': mscores0[0],
            'matching_scores1': mscores1[0],
            'loss': loss_mean[0],
            # 'raw_loss': raw_loss,
            # 'raw_scores': scores,
            'skip_train': False
        }

        # scores big value or small value means confidence? log can't take neg value

    def process_scores_into_matches(self, scores):
        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        return indices0, indices1, mscores0, mscores1

    def calculate_match_scores(self, data):
        # desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()
        # kpts0, kpts1 = data['keypoints0'].double(), data['keypoints1'].double()
        desc0, desc1 = data['descriptors0'].float(), data['descriptors1'].float()
        kpts0, kpts1 = data['keypoints0'].float(), data['keypoints1'].float()
        # desc0 = desc0.transpose(0, 1)
        # desc1 = desc1.transpose(0, 1)
        # kpts0 = torch.reshape(kpts0, (1, -1, 3)) # NEW: 3d
        # kpts1 = torch.reshape(kpts1, (1, -1, 3))
        # Batch is 0
        batch_sz = data['scores0'].shape[0]
        desc0 = desc0.transpose(1, 2)
        desc1 = desc1.transpose(1, 2)
        kpts0 = torch.reshape(kpts0, (batch_sz, 1, -1, 3))  # NEW: 3d
        kpts1 = torch.reshape(kpts1, (batch_sz, 1, -1, 3))
        # if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
        #     shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
        #     return {
        #         'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
        #         'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
        #         'matching_scores0': kpts0.new_zeros(shape0)[0],
        #         'matching_scores1': kpts1.new_zeros(shape1)[0],
        #         'skip_train': True
        #     }
        # Keypoint normalization.
        kpts0 = normalize_keypoints_3d(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints_3d(kpts1, data['image1'].shape)
        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, torch.transpose(data['scores0'], 1, 2))
        desc1 = desc1 + self.kenc(kpts1, torch.transpose(data['scores1'], 1, 2))
        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5
        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])
        return scores

    def match_and_output_list(self, data):
        scores = self.calculate_match_scores(data)
        indices0, _, mscores0, _ = self.process_scores_into_matches(scores)

        matches_with_conf = [[i, int(m), score] for i, (m, score) in enumerate(zip(indices0[0], mscores0[0]))]
        return matches_with_conf


## MY ADDITIONS


class SuperGlueModel(LightningModule):
    def __init__(self, feature_dim=840, criterion=None, lr=1e-3):
        super().__init__()

        self.superglue = SuperGlue(config=dict(descriptor_dim=feature_dim))
        self.lr = lr

    def forward(self, x):
        # Returns a dict with several values
        return self.superglue(x)

    def training_step(self, batch, batch_idx):
        # Designed to be used with SuperGlueUnpacker
        pred = self(batch)
        loss = pred['loss']
        self.log("loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = pred['loss']
        self.log("val_loss", loss)


@dataclass
class SuperGlueUnpacker:

    project_data: ProjectData = None

    t_template: int = 0
    data_template: dict = None

    def __post_init__(self):
        # Make a partial dictionary with the data from a single time point
        project_data = self.project_data
        t0 = self.t_template
        f0 = project_data.raw_frames[t0]

        # Unpack
        desc0, kpts0, scores0 = self.unpack_frame(f0)
        image0 = np.expand_dims(np.zeros_like(project_data.red_data[t0]), axis=0)

        # Repack
        data = dict(descriptors0=desc0, descriptors1=None,
                    keypoints0=kpts0, keypoints1=None, all_matches=None,
                    image0=image0, image1=image0,
                    scores0=scores0, scores1=None)

        self.data_template = data

    def unpack_frame(self, f0):
        desc0 = torch.tensor(f0.all_features).float()
        kpts0 = torch.tensor(f0.neuron_locs).float()
        scores0 = torch.ones((kpts0.shape[0], 1)).float()
        return desc0, kpts0, scores0

    def convert_frames_to_superglue_format(self, t0, t1, use_gt_matches=False):
        project_data = self.project_data

        f0 = project_data.raw_frames[t0]
        f1 = project_data.raw_frames[t1]

        # Unpack
        desc0, kpts0, scores0 = self.unpack_frame(f0)
        desc1, kpts1, scores1 = self.unpack_frame(f1)
        if use_gt_matches:
            df_gt = project_data.final_tracks
            all_matches = torch.tensor(df_to_matches(df_gt, t0, t1))
        else:
            all_matches = []

        image0 = torch.tensor(np.expand_dims(np.zeros_like(project_data.red_data[t0]), axis=0))
        # image1 = np.expand_dims(np.expand_dims(np.zeros_like(project_data.red_data[t1]), axis=0), axis=0)

        # Need expansion when not used in loop
        # all_matches = torch.unsqueeze(torch.tensor(df_to_matches(df_gt, t0, t1)), dim=0)
        # all_matches = torch.tensor(df_to_matches(df_gt, t0, t1))

        # Repack
        data = dict(descriptors0=desc0, descriptors1=desc1, keypoints0=kpts0, keypoints1=kpts1, all_matches=all_matches,
                    image0=image0, image1=image0,
                    scores0=scores0, scores1=scores1)

        return data

    def convert_single_frame_to_superglue_format(self, f1: ReferenceFrame, use_gt_matches=False):
        data = self.data_template.copy()
        project_data = self.project_data

        t0 = self.t_template
        t1 = f1.frame_ind

        desc1, kpts1, scores1 = self.unpack_frame(f1)
        if use_gt_matches:
            df_gt = project_data.final_tracks
            all_matches = torch.tensor(df_to_matches(df_gt, t0, t1))
        else:
            all_matches = []

        to_update = dict(descriptors1=desc1, keypoints1=kpts1, all_matches=all_matches, scores1=scores1)
        data.update(to_update)

        return data

    def expand_all_data(self, data):
        # Necessary when calling outside a pytorch dataloader
        new_data = {}
        for k, v in data.items():
            new_data[k] = torch.tensor(v).unsqueeze(0)

        return new_data


class SuperGlueFullVolumeNeuronImageFeaturesDatasetFromProject(AbstractNeuronImageFeaturesFromProject):

    def __init__(self, project_data: ProjectData, num_to_calculate=100, use_adjacent_time_points=False):
        super().__init__(project_data)
        self.num_to_calculate = num_to_calculate

        self.unpacker = SuperGlueUnpacker(project_data=project_data)

        t_list = list(range(project_data.num_frames - 1))
        self.time_pairs = []
        for i in range(num_to_calculate):
            if use_adjacent_time_points:
                if i + 1 >= project_data.num_frames - 1:
                    logging.warning(f"{num_to_calculate} requested, but only {i} available")
                    self.num_to_calculate = i
                    break
                new_pair = [i, i + 1]
            else:
                new_pair = random_combination(t_list, 2)
            self.time_pairs.append(new_pair)

        # Precalculate
        print("Precaculating training data")
        self._items = []
        for t0, t1 in tqdm(self.time_pairs):
            val = self.unpacker.convert_frames_to_superglue_format(t0, t1)
            self._items.append(val)

    def __getitem__(self, idx):
        return self._items[idx]

    def __len__(self):
        return self.num_to_calculate


