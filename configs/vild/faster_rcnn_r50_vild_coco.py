_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/default_runtime.py',
]


# custom_imports = dict(imports=['mmdet/datasets/pipelines/load_embedding.py'],
#                       allow_failed_imports=False)

# dataset settings
dataset_type = 'CocoCLIP48Dataset'
data_root = '/project/CLIP-FADI/data/COCO2017/' 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)

# class settings
class_indices = [0, ]

find_unused_parameters = True

# Large Scale Jittering
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadEmbeddingFromFile'),
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_embeds']),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            emb_prefix=data_root + 'embeddings/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.00004)
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    step=[162000, 171000, 175500])

runner = dict(type='IterBasedRunner', max_iters=180000)
# runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=1, metric='bbox')

model = dict(
    type='ViLD',
    roi_head=dict(
        type='ViLDRoIHead',
        bbox_head=dict(
            type='EmbeddingBBoxHead',
            num_classes=48,
            reg_class_agnostic=True,
            emb_path=data_root + 'text_embeddings/coco48.pickle',
            loss_cls=dict(
                type='ViLDCrossEntropyLoss', loss_weight=1.0, T=0.01),
            loss_bbox=dict(loss_weight=1.0),
            loss_img=dict(type='L1Loss', loss_weight=0.5))),
    # model training and testing settings
    train_cfg=dict(
        rpn_proposal=dict(
            max_per_img=300)),
    )