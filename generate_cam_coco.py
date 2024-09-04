import torch
import clip
from PIL import Image
from pytorch_grad_cam import GradCAM
import cv2
import argparse
from data.dataset import FSSDataset
import pdb

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def get_cam_from_alldata(clip_model, preprocess, split='train', d0=None, d1=None, d2=None, d3=None,
                               datapath=None, campath=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    d0 = d0.dataset.img_metadata_classwise
    d1 = d1.dataset.img_metadata_classwise
    d2 = d2.dataset.img_metadata_classwise
    d3 = d3.dataset.img_metadata_classwise
    dd = [d0, d1, d2, d3]
    dataset_all = {}

    if split == 'train':
        for ii in range(80):
            index = ii % 4 + 1
            if ii % 4 == 3:
                index = 0
            dataset_all[ii] = dd[index][ii]
    else:
        for ii in range(80):
            index = ii % 4
            dataset_all[ii] = dd[index][ii]
    del d0, d1, d2, d3, dd

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in COCO_CLASSES]).to(device)
    for cls_id in range(80):
        L = len(dataset_all[cls_id])
        for ll in range(L):
            img_path = datapath + dataset_all[cls_id][ll]
            img = Image.open(img_path)
            img_input = preprocess(img).unsqueeze(0).to(device)
            class_name_id = cls_id

            # CAM
            clip_model.get_text_features(text_inputs)
            target_layers = [clip_model.visual.layer4[-1]]
            input_tensor = img_input
            cam = GradCAM(model=clip_model, target_layers=target_layers, use_cuda=True)
            target_category = class_name_id
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (50, 50))
            grayscale_cam = torch.from_numpy(grayscale_cam)
            save_path = campath + dataset_all[cls_id][ll] + '--' + str(class_name_id) + '.pt'
            torch.save(grayscale_cam, save_path)
            print('cam saved in ', save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMR')
    parser.add_argument('--imgpath', type=str, default='../Datasets_HSN/COCO2014/')
    parser.add_argument('--campath', type=str, default='../Datasets_HSN/CAM_Val_COCO/')
    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_clip, preprocess = clip.load('RN50', device, jit=False)
    FSSDataset.initialize(img_size=400, datapath='../Datasets_HSN', use_original_imgsize=False)


    # COCO-meta-train
    # train
    dataloader_test0 = FSSDataset.build_dataloader('coco', 1, 0, 0, 'train', 1)
    dataloader_test1 = FSSDataset.build_dataloader('coco', 1, 0, 1, 'train', 1)
    dataloader_test2 = FSSDataset.build_dataloader('coco', 1, 0, 2, 'train', 1)
    dataloader_test3 = FSSDataset.build_dataloader('coco', 1, 0, 3, 'train', 1)
    get_cam_from_alldata(model_clip, preprocess, split='train',
                         d0=dataloader_test0, d1=dataloader_test1,
                         d2=dataloader_test2, d3=dataloader_test3,
                         datapath=args.imgpath, campath=args.campath)

    # val
    dataloader_test0 = FSSDataset.build_dataloader('coco', 1, 0, 0, 'val', 1)
    dataloader_test1 = FSSDataset.build_dataloader('coco', 1, 0, 1, 'val', 1)
    dataloader_test2 = FSSDataset.build_dataloader('coco', 1, 0, 2, 'val', 1)
    dataloader_test3 = FSSDataset.build_dataloader('coco', 1, 0, 3, 'val', 1)
    get_cam_from_alldata(model_clip, preprocess, split='val',
                         d0=dataloader_test0, d1=dataloader_test1,
                         d2=dataloader_test2, d3=dataloader_test3,
                         datapath=args.imgpath, campath=args.campath)
