_base_ = "yolov7_tiny_syncbn_fast_8x16b-300e_coco.py"

data_root = "/home/bizon/data/YOLOv7_exp/od_faces_3"
class_name = ("face",)
# data_root = '/home/bizon/data/YOLOv7_exp/od_v7_13'
# class_name = ('door', 'window', 'mirror', 'basin', 'toilet', 'fridge', 'oven', 'cooktop', 'range', 'wine',
#     'bathtub', 'microwave', 'dishwasher', 'sink', 'showerhead', 'shower', 'stairs_up', 'stairs_down',
#     'washer', 'dryer', 'wd', 'picture frame', 'fireplace', 'countertop', 'curtain', 'dining table',
#     'tv', 'person', 'cabinet',)
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[
        (220, 20, 60),
        # (119, 11, 32),
    ],
)
# metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
#          (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
#          (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
#          (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
#          (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
#          (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
#          (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),])

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)],  # P5/32
]

max_epochs = 200
train_batch_size_per_gpu = 12
train_num_workers = 8

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth"  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
    ),
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + "/data/train.json",
        # data_prefix=dict(img='images/')
    ),
)

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=data_root + "/data/val.json",
        # data_prefix=dict(img='images/')
    )
)

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + "/data/val.json")
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, save_best="auto"),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type="LoggerHook", interval=1),
)
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
visualizer = dict(
    vis_backends=[dict(type="LocalVisBackend"), dict(type="WandbVisBackend")]
)  # noqa
