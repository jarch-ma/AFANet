import os
import torch
import os
import clip
from PIL import Image
from pytorch_grad_cam import GradCAM
import cv2
import argparse
from data.dataset import FSSDataset
import pdb


PASCAL_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def get_cam_from_alldata(clip_model, preprocess, d=None, datapath=None, campath=None):  # d=Datorloader

    
    dataset_all = d.dataset.img_metadata  
    L = len(dataset_all) 
    
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in PASCAL_CLASSES]).to(device) 


    for ll in range(L):
   
        img_path = datapath + dataset_all[ll][0] + '.jpg'   
        img = Image.open(img_path)
        img_input = preprocess(img).unsqueeze(0).to(device)
        class_name_id = dataset_all[ll][1] 
        clip_model.get_text_features(text_inputs)
    
        target_layers = [clip_model.visual.layer4[-1]]    
        input_tensor = img_input
        cam = GradCAM(model=clip_model, target_layers=target_layers, use_cuda=True)  
        target_category = class_name_id         
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (50, 50))
        grayscale_cam = torch.from_numpy(grayscale_cam)
        save_path = campath + dataset_all[ll][0] + '--' + str(class_name_id) + '.pt'
        torch.save(grayscale_cam, save_path)

 



if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='AFANet')
    parser.add_argument('--imgpath', type=str, default='../Datasets_AFANet/voc2012/JPEGImages/') # Specify your data path
    parser.add_argument('--traincampath', type=str, default='../Datasets_AFANet/CAM_VOC_Train/')
    parser.add_argument('--valcampath', type=str, default='../Datasets_AFANet/CAM_VOC_Val/')
    parser.add_argument('--bsz', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:',device)
    model_clip, preprocess = clip.load('RN50', device, jit=False)   
    FSSDataset.initialize(img_size=400, datapath='../Datasets_AFANet/', use_original_imgsize=False) # Specify your data path

    # VOC
    # train
    dataloader_test0 = FSSDataset.build_dataloader('pascal', args.bsz, 1, 0, 'train', 1)
    dataloader_test1 = FSSDataset.build_dataloader('pascal', args.bsz, 1, 1, 'train', 1) # bsz nworker shot
    dataloader_test2 = FSSDataset.build_dataloader('pascal', args.bsz, 1, 2, 'train', 1)
    dataloader_test3 = FSSDataset.build_dataloader('pascal', args.bsz, 1, 3, 'train', 1)

    get_cam_from_alldata(model_clip, preprocess, d=dataloader_test0, datapath=args.imgpath, campath=args.traincampath)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_test1, datapath=args.imgpath, campath=args.traincampath)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_test2, datapath=args.imgpath, campath=args.traincampath)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_test3, datapath=args.imgpath, campath=args.traincampath)

    # val
    dataloader_test0 = FSSDataset.build_dataloader('pascal', args.bsz, 1, 0, 'val', 1)
    dataloader_test1 = FSSDataset.build_dataloader('pascal', args.bsz, 1, 1, 'val', 1)
    dataloader_test2 = FSSDataset.build_dataloader('pascal', args.bsz, 1, 2, 'val', 1)
    dataloader_test3 = FSSDataset.build_dataloader('pascal', args.bsz, 1, 3, 'val', 1)

    get_cam_from_alldata(model_clip, preprocess, d=dataloader_test0, datapath=args.imgpath, campath=args.valcampath)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_test1, datapath=args.imgpath, campath=args.valcampath)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_test2, datapath=args.imgpath, campath=args.valcampath)
    get_cam_from_alldata(model_clip, preprocess, d=dataloader_test3, datapath=args.imgpath, campath=args.valcampath)

    print('Done!')
