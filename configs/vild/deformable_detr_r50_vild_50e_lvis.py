_base_ = [
    '../_base_/models/deformable_detr_r50.py',
    '../_base_/iter_runtime.py',
]

# dataset settings
dataset_type = 'LVISClipCFDataset'
data_root = '/data/project/rw/lvis_v1/' 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ViLD version model
model = dict(
    type='DeformableDETRViLD',
    bbox_head=dict(
        type='DeformableDETRViLDHead',
        num_classes=866,
        word_emb_dim=512,
        base_emb_path=data_root + 'text_embeddings/lvis_cf.pickle',
        novel_emb_path=data_root + 'text_embeddings/lvis_r.pickle',
        loss_img=dict(type='L1Loss', loss_weight=1.0)))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadEmbeddingFromFile', 
         with_score=True, 
         ann_file=data_root + 'annotations/lvis_v1_train_embed.json'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                    use_embeds=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_embeds', 'gt_embed_weights'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
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
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(interval=5, metric='bbox')

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
            emb_prefix=data_root + 'img_embeddings/',
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

zero_shot_head = 'DeformableDETRViLDHead'