# AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation

The completed code will be released when the paper is accepted.

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
## AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation
This is the implementation of the paper "AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation".  

The codes are implemented based on IMR-HSNet(https://github.com/juhongm999/hsnet), CLIP(https://github.com/openai/CLIP), and https://github.com/jacobgil/pytorch-grad-cam. Thanks for their great work!  

## Requirements (For Chinese, please refer to alternative installation methods below.)

## Environment settings:
```bash
conda create -n afanet python=3.7
conda create -n afanet python=3.9
conda activate afanet

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
pip3 install torch torchvision torchaudio
pip install tensorflow
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
## Preparing Few-Shot Segmentation Datasets
Download following datasets:

@@ -22,8 +61,6 @@ Download following datasets:
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> ```
> Download PASCAL VOC2012 SDS extended mask annotations from HSNet [[Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)].
> #### 2. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations: 
> ```bash
@@ -33,11 +70,8 @@ Download following datasets:
> ```
> Download COCO2014 train/val annotations from HSNet Google Drive: [[train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing)].
> (and locate both train2014/ and val2014/ under annotations/ directory).
## Preparing Pre-trained models：
https://drive.google.com/file/d/1HKOGnvijW_KGTja_AdyHisRMXztoT10J/view?usp=sharing
Create a directory 'Dataset' for the above three few-shot segmentation datasets and appropriately place each dataset to have following directory structure:
    Dataset/                       
    └── Datasets_AFANet/
@@ -57,7 +91,6 @@ Create a directory 'Dataset' for the above three few-shot segmentation datasets
        ├── CAM_VOC_Val/ 
        └── CAM_COCO/
            
## Preparing CAM for Few-Shot Segmentation Datasets
> ### 1. PASCAL-5<sup>i</sup>
> * Generate Grad CAM for images
@@ -69,84 +102,3 @@ Create a directory 'Dataset' for the above three few-shot segmentation datasets
### 2. COCO-20<sup>i</sup>
> ```bash
> python generate_cam_coco.py --campath ../afanet_data/CAM_COCO/
## Training
> ### 1. PASCAL-5<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50} 
>                 --fold {0, 1, 2, 3} 
>                 --benchmark pascal
>                 --lr 4e-4
>                 --bsz 16
>                 --stage 2
>                 --logpath "your_experiment_name"
>                 --traincampath ../afanet_data/CAM_VOC_Train/
>                 --valcampath ../afanet_data/CAM_VOC_Val/
> ```
> ############
> 
> 
> 
> * Training takes approx. 6~7 hours until convergence (trained with four RTX 3090 GPUs).
> ### 2. COCO-20<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50}
>                 --fold {0, 1, 2, 3} 
>                 --benchmark coco 
>                 --lr 2e-4
>                 --bsz 20
>                 --stage 3
>                 --logpath "your_experiment_name"
>                 --traincampath ../afanet_data/CAM_COCO/
>                 --valcampath ../afanet_data/CAM_COCO/
> ```
> * Training takes approx. 3 days until convergence (trained with four RTX 3090 GPUs).
> ### Babysitting training:
> Use tensorboard to babysit training progress:
> - For each experiment, a directory that logs training progress will be automatically generated under logs/ directory. 
> - From terminal, run 'tensorboard --logdir logs/' to monitor the training progress.
> - Choose the best model when the validation (mIoU) curve starts to saturate. 
## Testing
> ### 1. PASCAL-5<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1fB3_jUEw972lDZIs3_S7lj2F5rZVq4Nu?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50} 
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5} 
>                --load "/opt/data/private/Code/AFANet/Pretrain/vis/voc_fold_0.pt" 
> ```
> ### 2. COCO-20<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1fB3_jUEw972lDZIs3_S7lj2F5rZVq4Nu?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50} 
>                --fold {0, 1, 2, 3} 
>                --benchmark coco 
>                --nshot {1, 5} 
>                --load "/opt/data/private/Code/AFANet/Pretrain/vis/coco_fold_0.pt"
> ```
## Visualization
> python test.py --backbone {vgg16, resnet50} 
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5}
>                --visualize 'visualize'
>                --load "/opt/data/private/Code/AFANet/Pretrain/vis/voc_fold_0.pt" 
   
## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
````
