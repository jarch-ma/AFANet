# AFANet: Adaptive Frequency-Aware Network for Weakly-Supervised Few-Shot Semantic Segmentation

The code will be released when the paper is accepted by ECCV2024.

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

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CLIP</th>
<th valign="bottom">A-847</th>
<th valign="bottom">PC-459</th>
<th valign="bottom">A-150</th>
<th valign="bottom">PC-59</th>
<th valign="bottom">PAS-20</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: SED (B) -->
<tr>
<td align="center">SED (B)</a></td>
<td align="center">ConvNeXt-B</td>
<td align="center">11.2</td>
<td align="center">18.6</td>
<td align="center">31.8</td>
<td align="center">57.7</td>
<td align="center">94.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1qx6zGZgSPkF6TObregRz4uzQqSRHrgUw/view?usp=drive_link">ckpt</a>&nbsp;
</tr>
<!-- ROW: SED (B) -->
<tr>
<td align="center">SED-fast (B)</a></td>
<td align="center">ConvNeXt-B</td>
<td align="center">11.4</td>
<td align="center">18.6</td>
<td align="center">31.6</td>
<td align="center">57.3</td>
<td align="center">94.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1qx6zGZgSPkF6TObregRz4uzQqSRHrgUw/view?usp=drive_link">ckpt</a>&nbsp;
</tr>
<!-- ROW: SED (L) -->
<tr>
<td align="center">SED (L)</a></td>
<td align="center">ConvNeXt-L</td>
<td align="center">13.7</td>
<td align="center">22.1</td>
<td align="center">35.3</td>
<td align="center">60.9</td>
<td align="center">96.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view?usp=drive_link">ckpt</a>&nbsp;
</tr>
<!-- ROW: SED-fast (L) -->
 <tr><td align="center">SED-fast (L)</a></td>
<td align="center">ConvNeXt-L</td>
<td align="center">13.9</td>
<td align="center">22.6</td>
<td align="center">35.2</td>
<td align="center">60.6</td>
<td align="center">96.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view?usp=drive_link">ckpt</a>&nbsp;
</tr>
</tbody></table>



## Citation
...

## Acknowledgement
We would like to acknowledge the contributions of public projects, such as ..., whose code has been utilized in this repository.
