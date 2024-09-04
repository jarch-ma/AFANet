
## AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation
This is the implementation of the paper "AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation".  

The codes are implemented based on IMR-HSNet(https://github.com/juhongm999/hsnet), CLIP(https://github.com/openai/CLIP), and https://github.com/jacobgil/pytorch-grad-cam. Thanks for their great work!  

## Environment settings:
```bash
conda create -n afanet python=3.9
conda activate afanet
pip3 install torch torchvision torchaudio
pip install tensorflow
pip install tensorboardX
```

## Preparing Few-Shot Segmentation Datasets
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

## Preparing Pre-trained models：
https://drive.google.com/file/d/1HKOGnvijW_KGTja_AdyHisRMXztoT10J/view?usp=sharing


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
> ### 1. PASCAL-5<sup>i</sup>
> * Generate Grad CAM for images
> ```bash
> python generate_cam_voc.py --traincampath ../afanet_data/CAM_VOC_Train/
>                            --valcampath ../afanet_data/CAM_VOC_Val/
> ```
> # python generate_cam_voc.py --traincampath ../afanet_data/CAM_VOC_Train --valcampath ../afanet_data/CAM_VOC_Val
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
