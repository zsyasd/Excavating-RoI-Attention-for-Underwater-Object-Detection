"""
_base_ = [
    '../_base_/datasets/utdac_detection_coco.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
"""

_base_ = [
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1, # retinanet anchor才用
        add_extra_convs='on_input', # retinanet anchor才用
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        # anchor_generator=dict(
        #     type='AnchorGenerator',
        #     scales=[8],
        #     ratios=[0.5, 1.0, 2.0],
        #     strides=[4, 8, 16, 32, 64]),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='TrHead',
        num_token=40, # D
        inchannel=7*7*256,
        emb_dim=256,
        num_heads=4,
        mlp_dim = 1024,
        depth = 2,
        mode = 'external_attention',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            # featmap_strides=[4, 8, 16, 32]),
            featmap_strides=[8, 16, 32, 64, 128]),
        bbox_head=dict(
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            # num_classes=4, # UTDAC
            # num_classes = 80, # COCO
            num_classes = 20, # voc
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))
        # bbox_head=dict(
        #     type='Shared2FCBBoxHead',
        #     in_channels=256,
        #     fc_out_channels=1024,
        #     roi_feat_size=7,
        #     num_classes=4,
        #     bbox_coder=dict(
        #         type='DeltaXYWHBBoxCoder',
        #         target_means=[0., 0., 0., 0.],
        #         target_stds=[0.1, 0.1, 0.2, 0.2]),
        #     reg_class_agnostic=False,
        #     loss_cls=dict(
        #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        #     loss_bbox=dict(type='L1Loss', loss_weight=1.0))
))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    # rpn_proposal=dict(
    #     nms_across_levels=False,
    #     nms_pre=2000,
    #     nms_post=1000,
    #     max_num=1000,
    #     nms_thr=0.7,
    #     min_bbox_size=0),
    rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    #rpn=dict(
    #   nms_across_levels=False,
    #    nms_pre=1000,
    #    nms_post=1000,
    #    max_num=1000,
    #    nms_thr=0.7,
    #    min_bbox_size=0),
    #rcnn=dict(
    #    score_thr=0.05,
    #    nms=dict(type='nms', iou_threshold=0.5),
    #    max_per_img=100)
    
    rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
    
    # rcnn=dict(
    #     score_thr=0.001,
    #     nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001),
    #     max_per_img=100)
)