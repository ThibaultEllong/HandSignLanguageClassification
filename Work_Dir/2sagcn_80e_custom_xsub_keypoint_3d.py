model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='AGCN',
        in_channels=3,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='agcn')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=2,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)
dataset_type = 'PoseDataset'
ann_file_train = '"G:\TSP\DA\Cassiopée\HandSignLanguageClassification\Dataset\CassiopéePKL\xsub\train.pkl"'
ann_file_val = '"G:\TSP\DA\Cassiopée\HandSignLanguageClassification\Dataset\CassiopéePKL\xsub\val.pkl"'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='PoseDataset',
        ann_file=
        ann_file_train,
        data_prefix='',
        pipeline=[
            dict(type='PaddingWithLoop', clip_len=300),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', input_format='NCTVM'),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ]),
    val=dict(
        type='PoseDataset',
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=[
            dict(type='PaddingWithLoop', clip_len=300),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', input_format='NCTVM'),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ]),
    test=dict(
        type='PoseDataset',
        ann_file='data/ntu/nturgb+d_skeletons_60_3d/xsub/val.pkl',
        data_prefix='',
        pipeline=[
            dict(type='PaddingWithLoop', clip_len=300),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', input_format='NCTVM'),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ]))
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 40])
total_epochs = 80
checkpoint_config = dict(interval=3)
evaluation = dict(interval=3, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'G:/TSP/DA/Cassiopée/HandSignLanguageClassification/Work_Dir'
load_from = None
resume_from = None
workflow = [('train', 1)]
omnisource = False
module_hooks = []
