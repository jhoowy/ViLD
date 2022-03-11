_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/iter_runtime.py',
]

# Test config for CLIP on cropped regions

# dataset settings
dataset_type = 'LVISClipRareDataset'
data_root = '/data/project/rw/lvis_v1/' 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
image_size = (1024, 1024)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadEmbeddingFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_embeds', 'gt_embed_bboxes', 'gt_embed_scores']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val=dict(
        type='LVISClipRareDataset',
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        emb_prefix=data_root + 'proposal_embeddings/',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type='LVISClipRareDataset',
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        emb_prefix=data_root + 'proposal_embeddings/',
        img_prefix=data_root,
        pipeline=test_pipeline))

model = dict(
    type='CLIPDetector',
    backbone=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        type='CLIPRoIHead',
        novel_emb_path=data_root + 'text_embeddings/lvis_r.pickle',
        with_ens=True,
        bbox_head=dict(
            type='CLIPBBoxHead',
            num_classes=866,
            reg_class_agnostic=True),
        mask_head=dict(
            class_agnostic=True,
            num_classes=866)),
    # model training and testing settings
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.6),
            # LVIS allows up to 300
            max_per_img=300)),
    )
