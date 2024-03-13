# AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation

The completed code will be released when the paper is accepted by ECCV2024.

## Abstract
<div align=center><img width="80%" src="assets/fig.2.png"></div> 

Few-shot learning aims to recognize novel concepts by leveraging prior knowledge learned from a few samples. However, for visually intensive tasks such as few-shot semantic segmentation, pixel-level annotations are time-consuming and costly. Therefore, in this work, we utilize the more challenging image-level annotations and propose an adaptive frequency-aware network (AFANet) for weakly-supervised few-shot semantic segmentation (WFSS). Specifically, we first propose a cross-granularity frequency-aware module (CFM) that decouples RGB images into high-frequency and low-frequency distributions and further optimizes semantic structural information by realigning them. Unlike most existing WFSS methods using the textual information from the language-vision model CLIP in an offline learning manner, we further propose a CLIP-guided spatial-adapter module (CSM), which performs spatial domain adaptive transformation on textual information through online learning, thus providing cross-modal semantic information for CFM. Extensive experiments on the Pascal-5\textsuperscript{i} and COCO-20\textsuperscript{i} datasets demonstrate that AFANet has achieved state-of-the-art performance.
<!-- 
## :fire: News
- AFANet is accepted by xxx.
-->
<!-- 
For further details and visualization results, please check out our [paper](XXXXXX).
-->
## Installation (Recommend)
1. Download the environment we have packaged: [Google drive](https://drive.google.com/file/d/1z1bjhJON1z2-T8bjL8wiwXloY3NOaGbZ/view?usp=drive_link) or [Baidu drive](https://pan.baidu.com/s/1E4f4Xl8epAguo0hxGijF_w?pwd=n4nm). Baidu extraction code: n4nm
2. Create the folder afanet in the conda envs directory
```bash
mkdir afanet
```
3. Put afanet_envs.tar.gz into afanet folder and type:
```bash
tar -zxvf afanet_envs.tar.gz
```
## Installation (Regular)

Conda environment settings:

```bash
conda create -n afanet python=3.7
conda activate afanet

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
pip install tensorboardX
```

## Data Preparation
#### PASCAL-5<sup>i</sup>
1. Download PASCAL VOC2012 devkit (train/val data):
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
2. Download PASCAL VOC2012 SDS extended mask annotations from HSNet [[Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)].

#### 2. COCO-20<sup>i</sup>
1. Download COCO2014 train/val images and annotations: 
```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```
2. Download COCO2014 train/val annotations from HSNet Google Drive: [[train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing)].

3. Create a directory '../Datasets_AFANet' for the above three few-shot segmentation datasets and appropriately place each dataset to have following directory structure:

    ../                         # parent directory
    ├── ./                      # current (project) directory
    │   ├── common/             # (dir.) helper functions
    │   ├── data/               # (dir.) dataloaders and splits for each FSS dataset
    │   ├── model/              # (dir.) implementation of Hypercorrelation Squeeze Network model 
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training AFANet
    │   └── test.py             # code for testing AFANet
    └── Datasets_HSN/
        ├── VOC2012/            # PASCAL VOC2012 devkit
        │   ├── Annotations/
        │   ├── ImageSets/
        │   ├── ...
        │   └── SegmentationClassAug/
        ├── COCO2014/           
        │   ├── annotations/
        │   │   ├── train2014/  # (dir.) training masks (from Google Drive) 
        │   │   ├── val2014/    # (dir.) validation masks (from Google Drive)
        │   │   └── ..some json files..
        │   ├── train2014/
        │   └── val2014/
        ├── CAM_VOC_Train/ 
        ├── CAM_VOC_Val/ 
        └── CAM_COCO/

## Preparing CAM for Weakly Few-Shot Segmentation Datasets
1. PASCAL-5<sup>i</sup>
```bash
python generate_cam_voc.py --traincampath ../Datasets_AFANet/CAM_VOC_Train/
                           --valcampath ../Datasets_AFANet/CAM_VOC_Val/
```
2. COCO-20<sup>i</sup>
```bash
python generate_cam_coco.py --campath ../Datasets_AFANet/CAM_COCO/
``

## Training
1. PASCAL-5<sup>i</sup>
```bash
python train.py --backbone {vgg16, resnet50} 
                --fold {0, 1, 2, 3} 
                --benchmark pascal
                --lr 4e-4
                --bsz 16
                --stage 2
                --logpath "your_experiment_name"
                --traincampath ../Datasets_AFANet/CAM_VOC_Train/
```               
Training takes approx. 7 hours until convergence (trained four RTX3090 GPUs).      
2. COCO-20<sup>i</sup>
```bash
python train.py --backbone {vgg16, resnet50}
                --fold {0, 1, 2, 3} 
                --benchmark coco 
                --lr 1e-4
                --bsz 20
                --stage 3
                --logpath "your_experiment_name"
                --traincampath ../Datasets_AFANet/CAM_COCO/
                --valcampath ../Datasets_AFANet/CAM_COCO/
```
Training takes approx. 3 days until convergence (trained four RTX3090 GPUs).
## Testing
Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/18Qd_7nBZgzMyaBUWFJhH1yA3y9aWCZwA?usp=drive_link)], or [[Baidu Drive](https://pan.baidu.com/s/1blUq0H7rnlKBlIytT1Fihg?pwd=cqbe)]. Extraction code: cqbe

1. PASCAL-5<sup>i</sup>
```bash
python test.py --backbone {vgg16, resnet50} 
               --fold {0, 1, 2, 3} 
               --benchmark pascal
               --nshot {1, 5} 
               --load "path_to_trained_model/best_model.pt" 
```
2. COCO-20<sup>i</sup>
```bash
python test.py --backbone {vgg16, resnet50} 
               --fold {0, 1, 2, 3} 
               --benchmark coco 
               --nshot {1, 5} 
               --load "path_to_trained_model/best_model.pt"
```
The inference time is reported on a single RTX 3090 GPU.


## Acknowledgement
The codes are implemented based on IMR-HSNet (https://github.com/Whileherham/IMR-HSNet), CLIP (https://github.com/openai/CLIP), and Grad-CAM (https://github.com/jacobgil/pytorch-grad-cam). Thanks for their great work!
