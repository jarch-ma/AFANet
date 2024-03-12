# AFANet
The code will be released when the paper is accepted by ECCV2024.


## AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation
This is the implementation of the paper "AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation".  

The codes are implemented based on IMR-HSNet(https://github.com/juhongm999/hsnet), CLIP(https://github.com/openai/CLIP), and https://github.com/jacobgil/pytorch-grad-cam. Thanks for their great work!  

Requirements (For Chinese, please refer to alternative installation methods below.)

We have two installation methods

Method 1： Download the packaged environment (Recommend)
1. Download AFANet environment from [[Google Drive](https://drive.google.com/file/d/1z1bjhJON1z2-T8bjL8wiwXloY3NOaGbZ/view?usp=sharing)]
2. Create an empty folder named AFANet in the conda envs directory, and move the file afanet_envs.tar.gz into this folder
3. CD AFANet folder，then type:
>  tar -zxvf afanet_envs.tar

Method 2：, following IMR-HSNet:
- Python 3.7
- PyTorch 1.5.1
- cuda 10.1
- tensorboard 1.14

Conda environment settings:
```bash
conda create -n afanet python=3.7
conda activate afanet
conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
pip install tensorboardX
```

Preparing Few-Shot Segmentation Datasets
Download following datasets:

> #### 1. PASCAL-5<sup>i</sup>
> Download PASCAL VOC2012 devkit (train/val data):
> ```bash
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> ```
> Download PASCAL VOC2012 SDS extended mask annotations from HSNet [[Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)].


> #### 2. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations: 
> ```bash
> wget http://images.cocodataset.org/zips/train2014.zip
> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> ```
> Download COCO2014 train/val annotations from HSNet Google Drive: [[train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing)].
> (and locate both train2014/ and val2014/ under annotations/ directory).

## 对于国人，我们更推荐以下安装方式，全都打包好了：
包含：1.conda环境；2.CAM；3.Data；4.pretrain model
https://www.alipan.com/s/C6UqfuZ8sAg

Create a directory 'Dataset' for the above three few-shot segmentation datasets and appropriately place each dataset to have following directory structure:
Dataset/                       
└── Datasets_AFANet/
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
            

## Preparing CAM for Few-Shot Segmentation Datasets
1. PASCAL-5<sup>i</sup>
> * Generate Grad CAM for images
> ```bash
> python generate_cam_voc.py --traincampath ../Datasets_AFANet/CAM_VOC_Train/
>                            --valcampath ../Datasets_AFANet/CAM_VOC_Val/
> ```

2. COCO-20<sup>i</sup>
> ```bash
> python generate_cam_coco.py --campath ../Datasets_AFANet/CAM_COCO/




## Training
> ### 1. PASCAL-5<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50} 
>                 --fold {0, 1, 2, 3} 
>                 --benchmark pascal
>                 --lr 4e-4
>                 --bsz 16
>                 --stage 2
>                 --logpath "your_experiment_name"  # (option)  
>                 --traincampath ../Datasets_AFANet/CAM_VOC_Train/
>                 --valcampath ../Datasets_AFANet/CAM_VOC_Val/
> ```

Set the above parameters to their default values, then run:
> ############
> 
> 
> 
> * Training takes approx. 10 hours until convergence (trained with four 3090 GPUs).


> ### 2. COCO-20<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50}
>                 --fold {0, 1, 2, 3} 
>                 --benchmark coco 
>                 --lr 1e-4
>                 --bsz 20
>                 --stage 3
>                 --logpath "your_experiment_name" # (option)  
>                 --traincampath ../Datasets_AFANet/CAM_COCO/
>                 --valcampath ../Datasets_AFANet/CAM_COCO/
> ```
> * Training takes approx. 3 days until convergence (trained four 3090 GPUs).


> ### Babysitting training:
> Use tensorboard to babysit training progress:
> - For each experiment, a directory that logs training progress will be automatically generated under logs/ directory. 
> - From terminal, run 'tensorboard --logdir logs/' to monitor the training progress.
> - Choose the best model when the validation (mIoU) curve starts to saturate. 



## Testing

> ### 1. PASCAL-5<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/18Qd_7nBZgzMyaBUWFJhH1yA3y9aWCZwA?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50} 
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5} 
>                --load "path_to_trained_model/fold_0.pt"  
> ```

> ### 2. COCO-20<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/18Qd_7nBZgzMyaBUWFJhH1yA3y9aWCZwA?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50} 
>                --fold {0, 1, 2, 3} 
>                --benchmark coco 
>                --nshot {1, 5} 
>                --load "path_to_trained_model/fold_0.pt"
> ```



   



