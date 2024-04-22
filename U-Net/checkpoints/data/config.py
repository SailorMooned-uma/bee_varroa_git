default_scope = 'mmseg'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='PSPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1, 0.3])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=0.4, class_weight=[1, 0.3])),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
dataset_type = 'SuperviselyDataset'
data_root = '/app/data/sly_seg_project'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (2336, 3504)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='SlyImgAugs', config_path='/app/data/augs_config.json'),
    dict(type='SlyImgAugs', config_path='/app/data/augs_config.json'),
    dict(type='Resize', img_scale=(2336, 3504), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'scale_factor', 'img_norm_cfg'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2336, 3504),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='SuperviselyDataset',
        data_root='/app/data/sly_seg_project',
        img_dir='img',
        ann_dir='seg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='SlyImgAugs', config_path='/app/data/augs_config.json'),
            dict(type='SlyImgAugs', config_path='/app/data/augs_config.json'),
            dict(
                type='Resize', img_scale=(2336, 3504), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'scale_factor', 'img_norm_cfg'))
        ],
        split='/app/data/train.txt',
        classes=['varroa', '__bg__'],
        palette=[[19, 99, 185], [0, 0, 0]],
        img_suffix='',
        seg_map_suffix='.png'),
    val=dict(
        type='SuperviselyDataset',
        data_root='/app/data/sly_seg_project',
        img_dir='img',
        ann_dir='seg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2336, 3504),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='/app/data/val.txt',
        classes=['varroa', '__bg__'],
        palette=[[19, 99, 185], [0, 0, 0]],
        img_suffix='',
        seg_map_suffix='.png'),
    test=dict(
        type='SuperviselyDataset',
        data_root='/app/data/sly_seg_project',
        img_dir='img',
        ann_dir='seg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2336, 3504),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split=None,
        classes=['varroa', '__bg__'],
        palette=[[19, 99, 185], [0, 0, 0]],
        img_suffix='',
        seg_map_suffix='.png'),
    persistent_workers=True)
log_config = dict(
    interval=2, hooks=[dict(type='SuperviselyLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/app/data/pspnet_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf_20211210_201823-53d492fa.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=False)
optimizer_config = dict(type='OptimizerHook')
lr_config = dict(
    policy='Poly',
    by_epoch=False,
    warmup=None,
    warmup_by_epoch=False,
    warmup_iters=0,
    warmup_ratio=0.1,
    min_lr=0.0001,
    power=0.9)
runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=2,
    save_last=True,
    out_dir='/sly-app-data/artifacts/checkpoints',
    meta=dict(
        CLASSES=['varroa', '__bg__'], PALETTE=[[19, 99, 185], [0, 0, 0]]),
    type='CheckpointHook')
evaluation = dict(
    interval=1,
    metric=['mIoU', 'mDice'],
    pre_eval=True,
    save_best='auto',
    rule='greater',
    out_dir='/sly-app-data/artifacts/checkpoints',
    by_epoch=True)
pretrained_model = 'UNet'
gpu_ids = range(0, 1)
work_dir = '/app/data'
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2336, 3504),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
seed = 0
