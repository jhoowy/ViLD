_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/iter_runtime.py',
]

# Selective Loss version

# dataset settings
dataset_type = 'LVISClipCFDataset'
data_root = '/data/project/rw/lvis_v1/' 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)

find_unused_parameters = True

# Large Scale Jittering
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadEmbeddingFromFile', 
         with_score=True, 
         ann_file=data_root + 'annotations/lvis_v1_train_embed_ens.json'),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True,
        use_embeds=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),  # padding to image_size leads 0.5+ mAP
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_masks', 'gt_labels', 'gt_embeds', 'gt_embed_weights']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
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
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_train.json',
            img_prefix=data_root,
            emb_prefix=data_root + 'img_embeddings_ens/',
            pipeline=train_pipeline)),
    val=dict(
        type='LVISClipRareDataset',
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type='LVISClipRareDataset',
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.032, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[162000, 171000, 175500])

zero_shot_head = 'EmbeddingBBoxHead'
evaluation = dict(interval=20000, metric='segm')
runner = dict(type='IterBasedRunner', max_iters=180000)

model = dict(
    type='ViLD',
    roi_head=dict(
        type='ViLDRoIHead',
        bbox_head=dict(
            type=zero_shot_head,
            num_classes=866,
            reg_class_agnostic=True,
            base_emb_path=data_root + 'text_embeddings/lvis_cf.pickle',
            novel_emb_path=data_root + 'text_embeddings/lvis_r.pickle',
            loss_cls=dict(_delete_=True,
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(loss_weight=1.0),
            loss_img=dict(type='L1Loss', loss_weight=1.0)),
        mask_head=dict(class_agnostic=True)),
    # model training and testing settings
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)),
    )
