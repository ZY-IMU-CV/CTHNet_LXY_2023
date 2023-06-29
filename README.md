## Requirements
### 1. Environment:
The implementation is based on mmdetection. So the requirements are exactly the same as [mmdetection v2.3.0rc0+8194b16](https://github.com/open-mmlab/mmdetection/tree/v2.3.0). We tested on the following settings:

- python 3.7.7
- cuda 10.1
- pytorch 1.5.0 
- torchvision 0.6.0
- mmcv 1.0.4

With settings above, please refer to [official guide of mmdetection](https://github.com/open-mmlab/mmdetection/blob/v2.3.0/docs/install.md) for installation.
### 2. Data:
. Data:
a. For dataset images:
# Make sure you are in dir CTHNet
mkdir data
cd data
mkdir lvis
mkdir pretrained_models
mkdir download_models
If you already have COCO2017 dataset, it will be great. Link train2017 and val2017 folders under folder lvis.
If you do not have COCO2017 dataset, please download: COCO train set and COCO val set and unzip these files and mv them under folder lvis.

### 3.Training

Both training and test commands are exactly the same as mmdetection, so please refer to mmdetection for basic usage.
```train
# Single GPU
python tools/train.py ${CONFIG_FILE}
```
Please make sure the path of datasets in config file is right.  

For example, to train a **CTHNet** model with Faster R-CNN R50-FPN for trainset of LVIS:
```train
# Single GPU
python tools/train.py configs/mylvis/f_lw_24_0.5.py 
python tools/train.py configs/mylvis/f_lw_24_0.5_jo.py 
``` 
Multi-gpu training and test are also supported as mmdetection.

### 4.Test
```test
# Single GPU
python tools/tests.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
 ```
 
For example (assuming that you have downloaded the corresponding chechkpoints file or train a model by yourself to proper path), to evaluate the trained **CTHNet** model with Faster R-CNN R50-FPN for LVIS:
```test
# Single GPU
python tools/test.py configs/mylvis/_lw_24_0.5_jo.py work_dirs/f_lw_24_0.5_jo/latest.pth --eval bbox
 ```

