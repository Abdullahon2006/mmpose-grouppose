# GroupPose — MMPose BottomupPoseEstimator config
#
# Evaluation:
#   python tools/test.py \
#     configs/body_2d_keypoint/grouppose/coco/grouppose_config.py \
#     checkpoints/resnet50_mmpose.pth
#
# Training:
#   python tools/train.py \
#     configs/body_2d_keypoint/grouppose/coco/grouppose_config.py

# ── runtime ──────────────────────────────────────────────────────────────────
default_scope = 'mmpose'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco/AP',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# ── dataset ──────────────────────────────────────────────────────────────────
dataset_type = 'CocoDataset'
data_mode = 'bottomup'
data_root = 'data/coco/'

# ── model ─────────────────────────────────────────────────────────────────────
# Standard BottomupPoseEstimator with ResNet-50 backbone + GroupPoseHead.
# No neck — GroupPoseHead contains input_proj to build the 4-level feature
# pyramid internally, matching the original GroupPose architecture exactly.
model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    # No neck — input_proj inside GroupPoseHead handles feature projection
    neck=None,
    head=dict(
        type='GroupPoseHead',
        # ── backbone interface ───────────────────────────────────────────────
        # ResNet-50 out_indices=(1,2,3) → channels [512, 1024, 2048]
        backbone_num_channels=[512, 1024, 2048],
        num_feature_levels=4,       # 3 backbone levels + 1 downsampled
        # ── queries ─────────────────────────────────────────────────────────
        num_queries=100,
        num_select=100,
        num_keypoints=17,
        # ── transformer ─────────────────────────────────────────────────────
        hidden_dim=256,
        nheads=8,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        two_stage_type='standard',
        dec_pred_class_embed_share=False,
        dec_pred_pose_embed_share=False,
        two_stage_bbox_embed_share=False,
        two_stage_class_embed_share=False,
        cls_no_bias=False,
        # ── positional encoding ──────────────────────────────────────────────
        pe_temperatureH=20,
        pe_temperatureW=20,
        # ── loss weights ────────────────────────────────────────────────────
        num_classes=2,
        aux_loss=True,
        focal_alpha=0.25,
        cls_loss_coef=2.0,
        keypoints_loss_coef=10.0,
        oks_loss_coef=4.0,
        interm_loss_coef=1.0,
        no_interm_loss=False,
        # ── matcher ─────────────────────────────────────────────────────────
        set_cost_class=2.0,
        set_cost_keypoints=10.0,
        set_cost_oks=4.0,
    ),
    train_cfg=dict(),
    test_cfg=dict(flip_test=False),
)

# ── pipelines ─────────────────────────────────────────────────────────────────
# Val/test: shorter-side=800, max-longer=1333 — matches original GroupPose eval.
# Same preprocessing as EDPose (BottomupRandomChoiceResize with scales=[(800,1333)]).
_input_size = (800, 1333)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=_input_size),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='PackPoseInputs',
        meta_keys=(
            'id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
            'input_size', 'input_center', 'input_scale',
            'flip', 'flip_direction', 'flip_indices',
        ),
    ),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomChoiceResize',
        scales=[_input_size],
        keep_ratio=True,
    ),
    dict(
        type='PackPoseInputs',
        meta_keys=(
            'id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
            'img_shape', 'input_size', 'input_center', 'input_scale',
            'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
            'skeleton_links',
        ),
    ),
]

# ── dataloaders ───────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# ── evaluator ─────────────────────────────────────────────────────────────────
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    nms_mode='none',   # GroupPose top-k replaces NMS
    score_mode='bbox', # rank by classification score (pred.bbox_scores)
)
test_evaluator = val_evaluator

# ── optimizer ─────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        }
    ),
)

# ── scheduler ─────────────────────────────────────────────────────────────────
max_epochs = 50

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        milestones=[40],
        gamma=0.1,
        by_epoch=True,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=5)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=16)
find_unused_parameters = True
