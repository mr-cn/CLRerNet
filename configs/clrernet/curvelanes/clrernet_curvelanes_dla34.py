_base_ = [
    "../base_clrernet.py",
    "dataset_curvelanes_clrernet.py",
    "../../_base_/default_runtime.py",
]
default_scope = 'mmdet'

# custom imports
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
    ],
    allow_failed_imports=False,
)

cfg_name = "clrernet_curvelanes_dla34.py"

model = dict(
    type="CLRerNet",
    bbox_head=dict(
        type="CLRerHead",
        loss_iou=dict(
            type="LaneIoULoss",
            lane_width=2.5 / 224,
            loss_weight=4.0,
        ),
        loss_seg=dict(
            loss_weight=2.0,
            num_classes=2,  # 1 lane + 1 background
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            iou_dynamick=dict(
                type="LaneIoUCost",
                lane_width=2.5 / 224,
                use_pred_start_end=False,
                use_giou=True,
            ),
            iou_cost=dict(
                type="LaneIoUCost",
                lane_width=10 / 224,
                use_pred_start_end=True,
                use_giou=True,
            ),
        )
    ),
    test_cfg=dict(
        conf_threshold=0.42,
        cut_height=0,
        use_nms=True,
        as_lanes=True,
        nms_thres=15,
        nms_topk=16,
    ),
)

total_epochs = 15
checkpoint_config = dict(interval=total_epochs)

# 添加MMEngine所需的训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 使用新格式定义dataloader
train_dataloader = dict(batch_size=24)  # single GPU setting

# seed
randomness = dict(seed=0, deterministic=True)

# optimizer (新格式)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type="AdamW", lr=6e-4),
)

# learning rate policy (新格式)
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        begin=0,
        T_max=total_epochs,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
