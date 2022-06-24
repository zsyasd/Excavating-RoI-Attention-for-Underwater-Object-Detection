# Excavating-RoI-Attention-for-Underwater-Object-Detection
This paper was accepted by ICIP2022, code will be released soon

### Dataset

We use dataset UTDAC2020, the download link of which is shown as follows.

https://drive.google.com/file/d/1avyB-ht3VxNERHpAwNTuBRFOxiXDMczI/view?usp=sharing

It is recommended to symlink the dataset file to the root.

```
FERNet
├── data
│   ├── UTDAC2020
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── annotations
```

### Train

```
python tools/train.py configs/faster_rcnn/roitransformer_r50_fpn_1x_coco.py
```

