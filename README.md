
## Iterative Few-shot Semantic Segmentation from Image Label Text
This is the implementation of the paper "Iterative Few-shot Semantic Segmentation from Image Label Text" (IJCAI 2022).  
The codes are implemented based on HSNet(https://github.com/juhongm999/hsnet), CLIP(https://github.com/openai/CLIP), and https://github.com/jacobgil/pytorch-grad-cam. Thanks for their great work!  

## Requirements
Following HSNet:
- Python 3.7
- PyTorch 1.5.1
- cuda 10.1
- tensorboard 1.14

Conda environment settings:
```bash
conda create -n hsnet python=3.7
conda activate hsnet

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
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



Create a directory '../Datasets_HSN' for the above three few-shot segmentation datasets and appropriately place each dataset to have following directory structure:

    ../                         # parent directory
    ├── ./                      # current (project) directory
    │   ├── common/             # (dir.) helper functions
    │   ├── data/               # (dir.) dataloaders and splits for each FSSS dataset
    │   ├── model/              # (dir.) implementation of Hypercorrelation Squeeze Network model 
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training HSNet
    │   └── test.py             # code for testing HSNet
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
            

## Preparing CAM for Few-Shot Segmentation Datasets
> ### 1. PASCAL-5<sup>i</sup>
> * Generate Grad CAM for images
> ```bash
> python generate_cam_voc.py --traincampath ../Datasets_HSN/CAM_VOC_Train/
>                            --valcampath ../Datasets_HSN/CAM_VOC_Val/
> ```
> # python generate_cam_voc.py --traincampath /home/s02009/data/hsnet_date/CAM_VOC_Train --valcampath /home/s02009/data/hsnet_date/CAM_VOC_Val
### 2. COCO-20<sup>i</sup>
> ```bash
> python generate_cam_coco.py --campath ../Datasets_HSN/CAM_COCO/




## Training
> ### 1. PASCAL-5<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50} 
>                 --fold {0, 1, 2, 3} 
>                 --benchmark pascal
>                 --lr 4e-4
>                 --bsz 40
>                 --stage 2
>                 --logpath "your_experiment_name"
>                 --traincampath ../Datasets_HSN/CAM_VOC_Train/
>                 --valcampath ../Datasets_HSN/CAM_VOC_Val/
> ```
> ########
>python train.py --backbone resnet50 --fold 0 --benchmark pascal --lr 4e-4 --bsz 40 --stage 2 --logpath "/home/s02009/code/IMR-HSNet/voc_train.log" --traincampath /home/s02009/data/hsnet_date/CAM_VOC_Train --valcampath /home/s02009/data/hsnet_date/CAM_VOC_Val
> 
> ############
> 
> 
> 
> * Training takes approx. 1 days until convergence (trained with four V100 GPUs).


> ### 2. COCO-20<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50}
>                 --fold {0, 1, 2, 3} 
>                 --benchmark coco 
>                 --lr 2e-4
>                 --bsz 20
>                 --stage 3
>                 --logpath "your_experiment_name"
>                 --traincampath ../Datasets_HSN/CAM_COCO/
>                 --valcampath ../Datasets_HSN/CAM_COCO/
> ```
> * Training takes approx. 1 week until convergence (trained four V100 GPUs).


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
>                --load "path_to_trained_model/best_model.pt" 
> ```
###   nohup python test.py --backbone resnet50 --fold 0 --benchmark pascal --nshot 1 --load /home/s02009/code/IMR-HSNet/voc_train.log/best_model.pt  >my_test.log 2>&1&

> ### 2. COCO-20<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1fB3_jUEw972lDZIs3_S7lj2F5rZVq4Nu?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50} 
>                --fold {0, 1, 2, 3} 
>                --benchmark coco 
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```



   
## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
@inproceedings{ijcai2022p193,
  title     = {Iterative Few-shot Semantic Segmentation from Image Label Text},
  author    = {Wang, Haohan and Liu, Liang and Zhang, Wuhao and Zhang, Jiangning and Gan, Zhenye and Wang, Yabiao and Wang, Chengjie and Wang, Haoqian},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {1385--1392},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/193},
  url       = {https://doi.org/10.24963/ijcai.2022/193},
}
````
