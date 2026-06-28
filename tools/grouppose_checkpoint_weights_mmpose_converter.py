"""
GroupPose → MMPose checkpoint converter
========================================

Converts the original GroupPose ResNet-50 checkpoint into a format that loads
cleanly into the standard MMPose BottomupPoseEstimator
(ResNet backbone + GroupPoseHead, no neck).

Key mapping
-----------
Original checkpoint (saved by GroupPose training):

    {
        'model': {
            'backbone.0.body.conv1.weight':  ...,
            'backbone.0.body.bn1.*':         ...,
            'backbone.0.body.layer{n}.*.weight': ...,
            'input_proj.*':                  ...,
            'transformer.*':                 ...,
            'class_embed.*':                 ...,
            'pose_embed.*':                  ...,
        },
        'optimizer': ...,
        'epoch': ...,
    }

MMPose model (BottomupPoseEstimator) state-dict keys:

    backbone.conv1.weight          ← backbone.0.body.conv1.weight
    backbone.bn1.*                 ← backbone.0.body.bn1.*
    backbone.layer{n}.*.weight     ← backbone.0.body.layer{n}.*.weight
    head.input_proj.*              ← input_proj.*
    head.transformer.*             ← transformer.*
    head.class_embed.*             ← class_embed.*
    head.pose_embed.*              ← pose_embed.*

Keys containing 'num_batches_tracked' are dropped — MMPose uses
norm_eval=True which does not require tracked batch statistics.

Usage
-----
    python grouppose_checkpoint_weights_mmpose_converter.py \\
        --input  checkpoints/resnet50.pth \\
        --output checkpoints/resnet50_mmpose.pth

    # Also verify that the converted weights load into the model:
    python grouppose_checkpoint_weights_mmpose_converter.py \\
        --input  checkpoints/resnet50.pth \\
        --output checkpoints/resnet50_mmpose.pth \\
        --verify
"""

import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime

import torch


def _remap_key(k: str) -> str | None:
    """Map a single original-checkpoint key to its MMPose equivalent.

    Returns None for keys that should be dropped.
    """
    # Drop batch-norm tracking stats (MMPose uses norm_eval=True)
    if 'num_batches_tracked' in k:
        return None

    # ── backbone ──────────────────────────────────────────────────────────────
    # Original: backbone.0.body.<rest>
    # MMPose:   backbone.<rest>
    if k.startswith('backbone.0.body.'):
        return 'backbone.' + k[len('backbone.0.body.'):]

    # ── head-level keys ───────────────────────────────────────────────────────
    for prefix in ('input_proj.', 'transformer.', 'class_embed.', 'pose_embed.'):
        if k.startswith(prefix):
            return 'head.' + k

    # Drop anything else (e.g. old backbone wrapper keys)
    return None


def convert(src_path: str, dst_path: str) -> OrderedDict:
    """Load a GroupPose checkpoint and convert to MMPose format.

    Args:
        src_path: Path to the original GroupPose checkpoint (.pth).
        dst_path: Destination path for the MMPose-compatible checkpoint.

    Returns:
        The converted state dict (for optional verification).
    """
    print(f'Loading  : {src_path}')
    ckpt = torch.load(src_path, map_location='cpu')

    # ── extract the state dict ───────────────────────────────────────────────
    if 'model' in ckpt:
        src_state = ckpt['model']
        src_epoch = ckpt.get('epoch', 'unknown')
    elif 'state_dict' in ckpt:
        src_state = ckpt['state_dict']
        src_epoch = ckpt.get('meta', {}).get('epoch', 'unknown')
    else:
        src_state = ckpt
        src_epoch = 'unknown'

    print(f'  Source epoch : {src_epoch}')
    print(f'  Source keys  : {len(src_state)}')

    # ── remap keys ───────────────────────────────────────────────────────────
    new_state = OrderedDict()
    dropped = []
    unmapped = []
    for k, v in src_state.items():
        new_k = _remap_key(k)
        if new_k is None:
            dropped.append(k)
        elif new_k == k:
            unmapped.append(k)
        else:
            new_state[new_k] = v

    print(f'  Converted keys : {len(new_state)}')
    if dropped:
        print(f'  Dropped keys   : {len(dropped)} '
              f'(num_batches_tracked etc.)')
    if unmapped:
        print(f'  Unmapped keys  : {len(unmapped)} '
              f'(did not match any rule — inspect manually):')
        for k in unmapped[:10]:
            print(f'    {k}')

    # ── sanity checks ────────────────────────────────────────────────────────
    nan_keys = [k for k, v in new_state.items()
                if torch.is_tensor(v) and torch.isnan(v).any()]
    if nan_keys:
        print(f'  WARNING: {len(nan_keys)} tensors contain NaN values!')
        for k in nan_keys[:5]:
            print(f'    {k}')

    inf_keys = [k for k, v in new_state.items()
                if torch.is_tensor(v) and torch.isinf(v).any()]
    if inf_keys:
        print(f'  WARNING: {len(inf_keys)} tensors contain Inf values!')

    if not nan_keys and not inf_keys:
        print('  Sanity check   : PASSED (no NaN / Inf)')

    # ── build MMPose checkpoint ──────────────────────────────────────────────
    mmpose_ckpt = {
        'state_dict': new_state,
        'meta': {
            'author': 'GroupPose (converted)',
            'converted_from': os.path.basename(src_path),
            'converted_on': datetime.now().isoformat(timespec='seconds'),
            'source_epoch': src_epoch,
            'mmpose_version': '1.x',
            'config': (
                'configs/body_2d_keypoint/grouppose/coco/'
                'grouppose_config.py'
            ),
            'dataset_info': {
                'dataset_name': 'coco',
                'num_keypoints': 17,
                'keypoint_info': {
                    0:  dict(name='nose',            id=0,  color=[51,  153, 255]),
                    1:  dict(name='left_eye',         id=1,  color=[51,  153, 255]),
                    2:  dict(name='right_eye',        id=2,  color=[51,  153, 255]),
                    3:  dict(name='left_ear',         id=3,  color=[51,  153, 255]),
                    4:  dict(name='right_ear',        id=4,  color=[51,  153, 255]),
                    5:  dict(name='left_shoulder',    id=5,  color=[0,   255,   0]),
                    6:  dict(name='right_shoulder',   id=6,  color=[255,  128,   0]),
                    7:  dict(name='left_elbow',       id=7,  color=[0,   255,   0]),
                    8:  dict(name='right_elbow',      id=8,  color=[255,  128,   0]),
                    9:  dict(name='left_wrist',       id=9,  color=[0,   255,   0]),
                    10: dict(name='right_wrist',      id=10, color=[255,  128,   0]),
                    11: dict(name='left_hip',         id=11, color=[0,   255,   0]),
                    12: dict(name='right_hip',        id=12, color=[255,  128,   0]),
                    13: dict(name='left_knee',        id=13, color=[0,   255,   0]),
                    14: dict(name='right_knee',       id=14, color=[255,  128,   0]),
                    15: dict(name='left_ankle',       id=15, color=[0,   255,   0]),
                    16: dict(name='right_ankle',      id=16, color=[255,  128,   0]),
                },
                'skeleton_info': {
                    0:  dict(link=('left_ankle',     'left_knee'),      id=0,  color=[0,   255,   0]),
                    1:  dict(link=('left_knee',      'left_hip'),       id=1,  color=[0,   255,   0]),
                    2:  dict(link=('right_ankle',    'right_knee'),     id=2,  color=[255,  128,   0]),
                    3:  dict(link=('right_knee',     'right_hip'),      id=3,  color=[255,  128,   0]),
                    4:  dict(link=('left_hip',       'right_hip'),      id=4,  color=[51,  153, 255]),
                    5:  dict(link=('left_shoulder',  'left_hip'),       id=5,  color=[51,  153, 255]),
                    6:  dict(link=('right_shoulder', 'right_hip'),      id=6,  color=[51,  153, 255]),
                    7:  dict(link=('left_shoulder',  'right_shoulder'), id=7,  color=[51,  153, 255]),
                    8:  dict(link=('left_shoulder',  'left_elbow'),     id=8,  color=[0,   255,   0]),
                    9:  dict(link=('right_shoulder', 'right_elbow'),    id=9,  color=[255,  128,   0]),
                    10: dict(link=('left_elbow',     'left_wrist'),     id=10, color=[0,   255,   0]),
                    11: dict(link=('right_elbow',    'right_wrist'),    id=11, color=[255,  128,   0]),
                    12: dict(link=('left_eye',       'right_eye'),      id=12, color=[51,  153, 255]),
                    13: dict(link=('nose',           'left_eye'),       id=13, color=[51,  153, 255]),
                    14: dict(link=('nose',           'right_eye'),      id=14, color=[51,  153, 255]),
                    15: dict(link=('left_eye',       'left_ear'),       id=15, color=[51,  153, 255]),
                    16: dict(link=('right_eye',      'right_ear'),      id=16, color=[51,  153, 255]),
                },
                'joint_weights': [
                    1.0, 1.2, 1.2, 1.5, 1.5,
                    1.0, 1.0, 1.2, 1.2, 1.5,
                    1.5, 1.0, 1.0, 1.2, 1.2,
                    1.5, 1.5,
                ],
                'sigmas': [
                    0.026, 0.025, 0.025, 0.035, 0.035,
                    0.079, 0.079, 0.072, 0.072, 0.062,
                    0.062, 0.107, 0.107, 0.087, 0.087,
                    0.089, 0.089,
                ],
            },
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    torch.save(mmpose_ckpt, dst_path)
    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f'Saved    : {dst_path}  ({size_mb:.1f} MB)')

    return new_state


def verify(dst_path: str) -> None:
    """Load the converted checkpoint into MMPose and check key alignment."""
    print('\nVerifying converted checkpoint …')

    ckpt = torch.load(dst_path, map_location='cpu')
    state = ckpt['state_dict']

    backbone_keys = [k for k in state if k.startswith('backbone.')]
    head_keys     = [k for k in state if k.startswith('head.')]
    other_keys    = [k for k in state
                     if not k.startswith('backbone.') and not k.startswith('head.')]

    print(f'  backbone keys : {len(backbone_keys)}')
    print(f'  head keys     : {len(head_keys)}')
    if other_keys:
        print(f'  Other keys    : {len(other_keys)} (unexpected — inspect):')
        for k in other_keys[:10]:
            print(f'    {k}')

    # Sample a few backbone keys
    print('\n  Sample backbone keys:')
    for k in sorted(backbone_keys)[:5]:
        print(f'    {k}  {tuple(state[k].shape)}')

    # Sample a few head keys
    print('\n  Sample head keys:')
    for k in sorted(head_keys)[:5]:
        print(f'    {k}  {tuple(state[k].shape)}')

    # Optionally try loading into the MMPose model
    _try_mmpose_load(state)

    print('\nVerification complete.')


def _try_mmpose_load(state: dict) -> None:
    """Try loading state_dict into BottomupPoseEstimator + GroupPoseHead."""
    try:
        from mmpose.models import build_pose_estimator
        from mmengine.config import Config
    except ImportError:
        print('\n  (MMPose not found — skipping live load check)')
        return

    _ROOT = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(_ROOT, 'grouppose_config.py')
    if not os.path.isfile(cfg_path):
        print('\n  (grouppose_config.py not found — skipping live load check)')
        return

    try:
        cfg = Config.fromfile(cfg_path)
        model = build_pose_estimator(cfg.model)
        model.eval()

        model_keys = set(model.state_dict().keys())
        ckpt_keys  = set(state.keys())
        missing    = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys

        if missing:
            print(f'\n  MISSING  ({len(missing)}):')
            for k in sorted(missing)[:10]:
                print(f'    {k}')
        if unexpected:
            print(f'\n  UNEXPECTED ({len(unexpected)}):')
            for k in sorted(unexpected)[:10]:
                print(f'    {k}')
        if not missing and not unexpected:
            print('\n  Key match : PERFECT — all keys align')

        result = model.load_state_dict(state, strict=False)
        if not result.missing_keys and not result.unexpected_keys:
            print('  load_state_dict : STRICT PASS')
        else:
            print(f'  load_state_dict : missing={len(result.missing_keys)}, '
                  f'unexpected={len(result.unexpected_keys)}')
    except Exception as exc:
        print(f'\n  Live load check failed: {exc}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert GroupPose checkpoint to MMPose format')
    parser.add_argument(
        '--input', '-i',
        default='checkpoints/resnet50.pth',
        help='Path to the original GroupPose checkpoint '
             '(default: checkpoints/resnet50.pth)')
    parser.add_argument(
        '--output', '-o',
        default='checkpoints/resnet50_mmpose.pth',
        help='Output path for the MMPose checkpoint '
             '(default: checkpoints/resnet50_mmpose.pth)')
    parser.add_argument(
        '--verify', action='store_true',
        help='After conversion, inspect key structure and attempt '
             'live load into BottomupPoseEstimator + GroupPoseHead')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'ERROR: input checkpoint not found: {args.input}')
        print('Download it from:')
        print('  https://drive.google.com/drive/folders/1exJMkr7j_HbItRM-u7DWT7scx1n4htiF')
        sys.exit(1)

    convert(args.input, args.output)

    if args.verify:
        verify(args.output)

    print('\nDone. Run evaluation with:')
    print(
        '  GROUPPOSE_ROOT=$(pwd) \\\n'
        '  python tools/test.py \\\n'
        '    configs/body_2d_keypoint/grouppose/coco/'
        'grouppose_config.py \\\n'
        f'    {args.output}'
    )


if __name__ == '__main__':
    main()
