default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = 'checkpoints/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221222-24dabf78.pth'
resume = True
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN',
        graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        gcn_attention=False),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
dataset_type = 'PoseDataset'
ann_file = 'data/skeleton/ntu60_3d.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='PoseDataset',
            ann_file='data/skeleton/ntu60_3d.pkl',
            pipeline=[
                dict(type='PreNormalize3D'),
                dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
                dict(type='UniformSampleFrames', clip_len=100),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=2),
                dict(type='PackActionInputs')
            ],
            split='xsub_train')))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PoseDataset',
        ann_file='data/skeleton/ntu60_3d.pkl',
        pipeline=[
            dict(type='PreNormalize3D'),
            dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
            dict(
                type='UniformSampleFrames',
                clip_len=100,
                num_clips=1,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='PackActionInputs')
        ],
        split='xsub_val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PoseDataset',
        ann_file='data/skeleton/ntu60_3d.pkl',
        pipeline=[
            dict(type='PreNormalize3D'),
            dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
            dict(
                type='UniformSampleFrames',
                clip_len=100,
                num_clips=10,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='PackActionInputs')
        ],
        split='xsub_val',
        test_mode=True))
val_evaluator = [dict(type='AccMetric')]
test_evaluator = [dict(type='AccMetric')]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=16, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True))
auto_scale_lr = dict(enable=False, base_batch_size=128)
launcher = 'none'
work_dir = '/Users/merrheimmaissane/Documents/Cassiopee_donnees/Cassiopee-github/HandSignLanguageClassification/Work_Dir'
randomness = dict(seed=None, diff_rank_seed=False, deterministic=False)
