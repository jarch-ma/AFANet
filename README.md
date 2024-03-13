# AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation

The code will be released when the paper is accepted by ECCV2024.
<!-- 这是一个注释，不会在最终的文档中显示。 -->
## Abstract
<img width="80%" src="assets/main_arch.png"><br>
Few-shot learning aims to recognize novel concepts by leveraging prior knowledge learned from a few samples. However, for visually intensive tasks such as few-shot semantic segmentation, pixel-level annotations are time-consuming and costly. Therefore, in this work, we utilize the more challenging image-level annotations and propose an adaptive frequency-aware network (AFANet) for weakly-supervised few-shot semantic segmentation (WFSS). Specifically, we first propose a cross-granularity frequency-aware module (CFM) that decouples RGB images into high-frequency and low-frequency distributions and further optimizes semantic structural information by realigning them. Unlike most existing WFSS methods using the textual information from the language-vision model CLIP in an offline learning manner, we further propose a CLIP-guided spatial-adapter module (CSM), which performs spatial domain adaptive transformation on textual information through online learning, thus providing cross-modal semantic information for CFM. Extensive experiments on the Pascal-5\textsuperscript{i} and COCO-20\textsuperscript{i} datasets demonstrate that AFANet has achieved state-of-the-art performance.

## Installation
 

## Data Preparation


## Training


### Training script
```bash
sh run.sh

# For 
sh run.sh configs/convnextB_768.yaml 4 output/
# For 
sh run.sh c
```

## Evaluation


## Results
<img width="50%" src="assets/trade-off.png"><br>
We provide pretrained weights for our models reported in the paper. All of the models were evaluated with 4 RTX 3090 GPUs, and can be reproduced with the evaluation script above. 
The inference time is reported on a single RTX 3090 GPU.


## Citation
...

## Acknowledgement
We would like to acknowledge the contributions of public projects, such as ..., whose code has been utilized in this repository.
