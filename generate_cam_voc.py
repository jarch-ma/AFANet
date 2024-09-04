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
    # device = torch.device("cuda:" if torch.cuda.is_available() else "cpu")
    # print('device:',device)
    
    dataset_all = d.dataset.img_metadata    #  读取的全部数据 ['2007_000648', 5]...
    L = len(dataset_all) # 11394
    
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in PASCAL_CLASSES]).to(device)  # prompt在这里!!!


    for ll in range(L):
        # dataset_all 是这样的 ['2007_000648', 5], ['2007_000768', 5]
        img_path = datapath + dataset_all[ll][0] + '.jpg'   # '/home/s02009/data/hsnet_date/voc2012/JPEGImages/2007_000648.jpg'
        img = Image.open(img_path)
        img_input = preprocess(img).unsqueeze(0).to(device)
        class_name_id = dataset_all[ll][1]   # 5
        clip_model.get_text_features(text_inputs)
        ###############生成 CAM#################
        target_layers = [clip_model.visual.layer4[-1]]     # cam指定网络层
        input_tensor = img_input
        cam = GradCAM(model=clip_model, target_layers=target_layers, use_cuda=True)  # cam
        target_category = class_name_id         # 类别
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (50, 50))
        grayscale_cam = torch.from_numpy(grayscale_cam)
        save_path = campath + dataset_all[ll][0] + '--' + str(class_name_id) + '.pt'
        torch.save(grayscale_cam, save_path)
        ##########################################生成 CAM###########################
        print('cam已经保存', save_path)



if __name__ == '__main__':

    torch.cuda.set_device(4)

    parser = argparse.ArgumentParser(description='IMR')
    parser.add_argument('--imgpath', type=str, default='/ssd/s02009/data/irnet_data/voc2012/JPEGImages/')
    parser.add_argument('--traincampath', type=str, default='/ssd/s02009/out/irnet_out/irnet_original_out/test/CAM_VOC_Train/') # pt文件
    parser.add_argument('--valcampath', type=str, default='/ssd/s02009/out/irnet_out/irnet_original_out/test/CAM_VOC_Val/')
    
    # 自己编写
    parser.add_argument('--bsz', type=int, default=32)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:',device)
    model_clip, preprocess = clip.load('RN50', device, jit=False)   # 原来是"RN50"
    FSSDataset.initialize(img_size=400, datapath='/ssd/s02009/data/irnet_data/', use_original_imgsize=False) # 对图像进行预处理

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
