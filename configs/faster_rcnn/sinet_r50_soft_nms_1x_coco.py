# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 20:46
# @Author  : Linhui Dai
# @FileName: sinet_r50_soft_nms_1x_coco.py
# @Software: PyCharm
_base_ = [
    '../_base_/models/sinet_r50.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

test_cfg = dict(
    rcnn=dict(
        score_thr=0.0001,
        nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.0001),
        max_per_img=100))