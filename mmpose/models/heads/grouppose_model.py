# Copyright (c) OpenMMLab. All rights reserved.
"""
GroupPose head for MMPose BottomupPoseEstimator.

This file lives at:
    mmpose/models/heads/transformer_heads/grouppose_model.py

It is registered via:
    mmpose/models/heads/transformer_heads/__init__.py
    mmpose/models/heads/__init__.py

The GroupPose transformer/criterion code lives in the top-level
``grouppose/`` package, which is installed alongside MMPose when you
run ``pip install -e .`` from the repo root.

Architecture
------------
Receives 3 feature maps from a ResNet-50 backbone (out_indices=(1,2,3)):
    [(B, 512, H1, W1), (B, 1024, H2, W2), (B, 2048, H3, W3)]

Internally:
  1. input_proj  — projects backbone features to 4 × 256-dim levels
                   (matches original GroupPose model.input_proj exactly)
  2. pos_encoder — PositionEmbeddingSineHW (no learnable parameters)
  3. transformer — original GroupPose transformer (encoder + decoder)
  4. class_embed / pose_embed — prediction MLP heads
  5. criterion   — SetCriterion with Hungarian matcher (training only)
  6. postprocessor — PostProcess: top-k selection + coordinate rescaling

Checkpoint key mapping
----------------------
Original checkpoint key              → MMPose model key
backbone.0.body.conv1.weight         → backbone.conv1.weight
backbone.0.body.bn1.*                → backbone.bn1.*
backbone.0.body.layer{n}.*.conv*.weight → backbone.layer{n}.*.conv*.weight
backbone.0.body.layer{n}.*.bn{k}.*  → backbone.layer{n}.*.bn{k}.*
input_proj.*                         → head.input_proj.*
transformer.*                        → head.transformer.*
class_embed.*                        → head.class_embed.*
pose_embed.*                         → head.pose_embed.*
"""

import copy
import math
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmpose.registry import MODELS
from mmpose.utils.typing import OptConfigType, SampleList
from torch import Tensor, nn

# ── GroupPose transformer / criterion imports ────────────────────────────────
from grouppose.transformer import build_transformer
from grouppose.criterion import SetCriterion
from grouppose.matcher import HungarianMatcher
from grouppose.postprocesses import PostProcess
from grouppose.utils import MLP
from grouppose.position_encoding import PositionEmbeddingSineHW
from grouppose.util.misc import NestedTensor, inverse_sigmoid
from grouppose.util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz


@MODELS.register_module()
class GroupPoseHead(nn.Module):
    """GroupPose transformer head for MMPose BottomupPoseEstimator.

    Designed to pair with a ResNet-50 backbone (out_indices=(1,2,3)) and
    no neck. The head handles multi-scale feature projection, positional
    encoding, transformer encoder-decoder, loss computation and inference
    post-processing.

    Args:
        num_queries (int): Number of instance queries. Default: 100.
        num_feature_levels (int): Number of feature pyramid levels. Default: 4.
        num_keypoints (int): Number of body keypoints. Default: 17.
        backbone_num_channels (list[int]): Output channel counts from backbone.
            For ResNet-50 out_indices=(1,2,3): [512, 1024, 2048].
        hidden_dim (int): Transformer hidden dimension. Default: 256.
        nheads (int): Multi-head attention heads. Default: 8.
        enc_layers (int): Transformer encoder layers. Default: 6.
        dec_layers (int): Transformer decoder layers. Default: 6.
        dim_feedforward (int): FFN hidden dimension. Default: 2048.
        dropout (float): Dropout rate. Default: 0.0.
        num_classes (int): Number of object classes. Default: 2.
        aux_loss (bool): Use auxiliary decoder losses. Default: True.
        two_stage_type (str): Two-stage mode. Default: 'standard'.
        dec_pred_class_embed_share (bool): Share class embed across decoder layers.
        dec_pred_pose_embed_share (bool): Share pose embed across decoder layers.
        focal_alpha (float): Focal loss alpha. Default: 0.25.
        cls_loss_coef (float): Classification loss weight. Default: 2.0.
        keypoints_loss_coef (float): Keypoint L1 loss weight. Default: 10.0.
        oks_loss_coef (float): OKS loss weight. Default: 4.0.
        num_select (int): Top-k queries selected at inference. Default: 100.
        pe_temperatureH (int): Positional encoding temperature (H). Default: 20.
        pe_temperatureW (int): Positional encoding temperature (W). Default: 20.
    """

    def __init__(
        self,
        num_queries: int = 100,
        num_feature_levels: int = 4,
        num_keypoints: int = 17,
        backbone_num_channels: List[int] = [512, 1024, 2048],
        hidden_dim: int = 256,
        nheads: int = 8,
        enc_layers: int = 6,
        dec_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        num_classes: int = 2,
        aux_loss: bool = True,
        two_stage_type: str = 'standard',
        dec_pred_class_embed_share: bool = False,
        dec_pred_pose_embed_share: bool = False,
        two_stage_bbox_embed_share: bool = False,
        two_stage_class_embed_share: bool = False,
        cls_no_bias: bool = False,
        focal_alpha: float = 0.25,
        cls_loss_coef: float = 2.0,
        keypoints_loss_coef: float = 10.0,
        oks_loss_coef: float = 4.0,
        interm_loss_coef: float = 1.0,
        no_interm_loss: bool = False,
        set_cost_class: float = 2.0,
        set_cost_keypoints: float = 10.0,
        set_cost_oks: float = 4.0,
        num_select: int = 100,
        pe_temperatureH: int = 20,
        pe_temperatureW: int = 20,
        # kept for MMPose API compatibility, unused
        decoder: Optional[dict] = None,
    ):
        super().__init__()

        self.num_body_points = num_keypoints
        self.num_select = num_select
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss
        self.num_classes = num_classes

        # ── build args namespace for GroupPose transformer builders ──────────
        args = SimpleNamespace(
            hidden_dim=hidden_dim,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=False,
            return_intermediate_dec=True,
            enc_n_points=4,
            dec_n_points=4,
            num_feature_levels=num_feature_levels,
            learnable_tgt_init=False,
            transformer_activation='relu',
            two_stage_type=two_stage_type,
            num_queries=num_queries,
            num_body_points=num_keypoints,
        )

        # ── input projection ─────────────────────────────────────────────────
        # Matches original GroupPose model.input_proj exactly so checkpoint
        # weights map 1:1 under the 'head.' prefix.
        num_bb_outs = len(backbone_num_channels)
        proj_list = []
        for in_ch in backbone_num_channels:
            proj_list.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        for i in range(num_feature_levels - num_bb_outs):
            in_ch = backbone_num_channels[-1] if i == 0 else hidden_dim
            proj_list.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        self.input_proj = nn.ModuleList(proj_list)

        # ── positional encoding (no learnable params — not in checkpoint) ────
        self.pos_encoder = PositionEmbeddingSineHW(
            hidden_dim // 2,
            temperatureH=pe_temperatureH,
            temperatureW=pe_temperatureW,
            normalize=True,
        )

        # ── transformer ──────────────────────────────────────────────────────
        self.transformer = build_transformer(args)

        # ── prediction heads ─────────────────────────────────────────────────
        _cls = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            bias_val = -math.log((1 - 0.01) / 0.01)
            _cls.bias.data = torch.ones(num_classes) * bias_val

        _pose = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pose.layers[-1].weight.data, 0)
        nn.init.constant_(_pose.layers[-1].bias.data, 0)

        cls_list = ([_cls] * dec_layers if dec_pred_class_embed_share
                    else [copy.deepcopy(_cls) for _ in range(dec_layers)])
        pose_list = ([_pose] * dec_layers if dec_pred_pose_embed_share
                     else [copy.deepcopy(_pose) for _ in range(dec_layers)])

        self.class_embed = nn.ModuleList(cls_list)
        self.pose_embed = nn.ModuleList(pose_list)
        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.num_body_points = num_keypoints

        # two-stage encoder heads
        _kpt = MLP(hidden_dim, 2 * hidden_dim, 2 * num_keypoints, 4)
        nn.init.constant_(_kpt.layers[-1].weight.data, 0)
        nn.init.constant_(_kpt.layers[-1].bias.data, 0)

        self.transformer.enc_pose_embed = (
            _kpt if two_stage_bbox_embed_share else copy.deepcopy(_kpt))
        self.transformer.enc_out_class_embed = (
            _cls if two_stage_class_embed_share else copy.deepcopy(_cls))

        self._reset_parameters()

        # ── criterion ────────────────────────────────────────────────────────
        matcher = HungarianMatcher(
            cost_class=set_cost_class,
            focal_alpha=focal_alpha,
            cost_keypoints=set_cost_keypoints,
            cost_oks=set_cost_oks,
            num_body_points=num_keypoints,
        )
        weight_dict = {
            'loss_ce': cls_loss_coef,
            'loss_keypoints': keypoints_loss_coef,
            'loss_oks': oks_loss_coef,
        }
        clean_wd = copy.deepcopy(weight_dict)
        if aux_loss:
            for i in range(dec_layers - 1):
                for k, v in clean_wd.items():
                    weight_dict[f'{k}_{i}'] = v
        if two_stage_type != 'no':
            coeff = 0.0 if no_interm_loss else 1.0
            weight_dict.update({
                f'{k}_interm': v * interm_loss_coef * coeff
                for k, v in clean_wd.items()
            })
        self.criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=focal_alpha,
            losses=['labels', 'keypoints', 'matching'],
            num_body_points=num_keypoints,
        )

        # ── postprocessor ────────────────────────────────────────────────────
        self.postprocessor = PostProcess(
            num_select=num_select,
            num_body_points=num_keypoints,
        )

    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, feats: Tuple[Tensor]) -> dict:
        """Run GroupPose transformer.

        Args:
            feats: Tuple of tensors from ResNet backbone
                   [(B,512,H1,W1), (B,1024,H2,W2), (B,2048,H3,W3)]

        Returns:
            dict with keys 'pred_logits', 'pred_keypoints', and optionally
            'aux_outputs', 'interm_outputs'.
        """
        srcs, masks, poss = [], [], []
        device = feats[0].device

        # project backbone features + generate positional encodings
        for i, feat in enumerate(feats):
            src = self.input_proj[i](feat)
            B, _, H, W = src.shape
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            pos = self.pos_encoder(NestedTensor(src, mask)).to(src.dtype)
            srcs.append(src)
            masks.append(mask)
            poss.append(pos)

        # extra feature levels via stride-2 convolutions
        for i in range(len(feats), self.num_feature_levels):
            src_in = feats[-1] if i == len(feats) else srcs[-1]
            src = self.input_proj[i](src_in)
            B, _, H, W = src.shape
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            pos = self.pos_encoder(NestedTensor(src, mask)).to(src.dtype)
            srcs.append(src)
            masks.append(mask)
            poss.append(pos)

        hs_pose, refpoint_pose, mix_refpoint, mix_embedding = (
            self.transformer(srcs, masks, poss))

        outputs_class, outputs_kpts = [], []
        for hs_i, ref_i, pose_emb, cls_emb in zip(
                hs_pose, refpoint_pose,
                self.pose_embed, self.class_embed):
            bs, nq, np_ = ref_i.shape
            ref_i = ref_i.reshape(bs, nq, np_ // 2, 2)
            delta = pose_emb(hs_i[:, :, 1:])
            out_unsig = inverse_sigmoid(ref_i[:, :, 1:]) + delta
            vis = torch.ones_like(out_unsig[..., -1:], device=device)
            out_unsig = torch.cat([out_unsig, vis], dim=-1).flatten(-2).sigmoid()
            outputs_kpts.append(keypoint_xyzxyz_to_xyxyzz(out_unsig))
            outputs_class.append(cls_emb(hs_i[:, :, 0]))

        out = {
            'pred_logits': outputs_class[-1],
            'pred_keypoints': outputs_kpts[-1],
        }
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_keypoints': k}
                for a, k in zip(outputs_class[:-1], outputs_kpts[:-1])
            ]
        if mix_refpoint is not None and mix_embedding is not None:
            out['interm_outputs'] = {
                'pred_logits': self.transformer.enc_out_class_embed(mix_embedding),
                'pred_keypoints': mix_refpoint,
            }
        return out

    # ── MMPose head interface ────────────────────────────────────────────────

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: SampleList,
        train_cfg: OptConfigType = None,
    ) -> dict:
        """Compute weighted GroupPose losses.

        Args:
            feats: Feature tuple from backbone.
            batch_data_samples: List of PoseDataSample with GT annotations.

        Returns:
            dict[str, Tensor]: Weighted loss dict.
        """
        outputs = self.forward(feats)
        targets = self._build_targets(feats[0], batch_data_samples)
        loss_dict = self.criterion(outputs, targets)
        wd = self.criterion.weight_dict
        return {k: loss_dict[k] * wd[k] for k in loss_dict if k in wd}

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: SampleList,
        test_cfg: OptConfigType = None,
    ) -> List[InstanceData]:
        """Run inference and return predicted instances.

        Keypoints are returned in the preprocessed-image coordinate space
        (0..img_H, 0..img_W). BottomupPoseEstimator.add_pred_to_datasample
        then maps them to original image coordinates using the resize
        transform's metainfo (input_size, input_center, input_scale).

        Args:
            feats: Feature tuple from backbone.
            batch_data_samples: List of PoseDataSample with metainfo.

        Returns:
            List[InstanceData]: Each with keypoints (N,K,2),
                keypoint_scores (N,K), keypoints_visible (N,K), scores (N,).
        """
        outputs = self.forward(feats)

        # PostProcess returns keypoints in [0,1] normalized space.
        # We pass dummy target_sizes; scaling is done manually below.
        img_shapes = [
            ds.metainfo.get('img_shape', feats[0].shape[-2:])[:2]
            for ds in batch_data_samples
        ]
        target_sizes = torch.tensor(
            img_shapes, dtype=torch.float32, device=feats[0].device)

        raw = self.postprocessor(outputs, target_sizes)

        batch_pred_instances = []
        for result, data_sample in zip(raw, batch_data_samples):
            kps = result['keypoints'].reshape(-1, self.num_body_points, 3)
            # PostProcess already scaled to input pixel space using target_sizes
            kps_xy = kps[..., :2].cpu().numpy()  # (N, K, 2)
            kps_vis = kps[..., 2].cpu().numpy()   # (N, K)

            cls_scores = result['scores'].cpu().numpy()  # (N,) — ranks TP above FP
            pred = InstanceData()
            pred.keypoints = kps_xy                # numpy (N, K, 2)
            pred.keypoint_scores = kps_vis         # numpy (N, K)
            pred.keypoints_visible = kps_vis       # numpy (N, K)
            pred.scores = cls_scores               # numpy (N,)
            pred.bbox_scores = cls_scores          # used by score_mode='bbox'
            batch_pred_instances.append(pred)

        return batch_pred_instances

    # ── GT conversion ────────────────────────────────────────────────────────

    def _build_targets(
        self, ref_feat: Tensor, batch_data_samples: SampleList
    ) -> List[dict]:
        """Convert MMPose PoseDataSample GT to GroupPose target format.

        GroupPose criterion expects per-image dicts:
            labels    (N,)     int64  — person class (all 0)
            keypoints (N,K*3)  float  — [x1,y1,…,xK,yK,v1,…,vK] normalised
            area      (N,)     float  — bbox area in pixels²
            boxes     (N,4)    float  — xyxy bbox normalised to [0,1]
        """
        H, W = ref_feat.shape[-2:]
        device = ref_feat.device
        targets = []

        for ds in batch_data_samples:
            gt = ds.gt_instances
            n = len(gt) if hasattr(gt, '__len__') else 0

            if n == 0 or not hasattr(gt, 'keypoints'):
                targets.append({
                    'labels': torch.zeros(0, dtype=torch.int64, device=device),
                    'keypoints': torch.zeros(
                        0, self.num_body_points * 3, device=device),
                    'area': torch.zeros(0, device=device),
                    'boxes': torch.zeros(0, 4, device=device),
                })
                continue

            kps = torch.as_tensor(
                gt.keypoints, dtype=torch.float32, device=device)
            vis = torch.as_tensor(
                gt.keypoints_visible, dtype=torch.float32, device=device)

            kps_n = kps.clone()
            kps_n[..., 0] /= W
            kps_n[..., 1] /= H
            kps_flat = torch.cat([kps_n.flatten(1), vis], dim=1)

            if hasattr(gt, 'bboxes') and len(gt.bboxes) > 0:
                bboxes = torch.as_tensor(
                    gt.bboxes, dtype=torch.float32, device=device)
                area = ((bboxes[:, 2] - bboxes[:, 0]) *
                        (bboxes[:, 3] - bboxes[:, 1]))
                boxes_n = bboxes.clone()
                boxes_n[:, [0, 2]] /= W
                boxes_n[:, [1, 3]] /= H
            else:
                area = torch.full(
                    (n,), float(H * W) * 0.1, device=device)
                boxes_n = torch.zeros(n, 4, device=device)

            targets.append({
                'labels': torch.zeros(n, dtype=torch.int64, device=device),
                'keypoints': kps_flat,
                'area': area,
                'boxes': boxes_n,
            })

        return targets
