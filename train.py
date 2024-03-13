import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import pdb
from datetime import datetime, timedelta
import torch.optim as optim
import torch.nn as nn
import torch
import random
import os
import numpy as np
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from model.AFANet import afanet  
import clip


print("cuda device:",os.environ['CUDA_VISIBLE_DEVICES'])
linux_os_start_time = datetime.now()    

print(f'Start time: {linux_os_start_time}')


def setup_seed(seed):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    # torch.use_deterministic_algorithms(True)   

######

def train(epoch, model, dataloader, optimizer, training, stage):
    r""" Train AFANet """  

    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. forward propagation
        batch = utils.to_cuda(batch)

        logit_mask_q, logit_mask_s, losses = model(
            query_img=batch['query_img'],              
            support_img=batch['support_imgs'].squeeze(1),    # Remove invalid color channel dimensions
            support_cam=batch['support_cams'].squeeze(1),   
            query_cam=batch['query_cam'], stage=stage,     
            query_mask=batch['query_mask'],
            support_mask=batch['support_masks'].squeeze(1),
            class_id = batch['class_id'])
        pred_mask_q = logit_mask_q.argmax(dim=1)  # Get the category with the highest probability as a mask

        # 2. Compute loss & update model parameters           
        loss = losses.mean()
        if training:
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
        

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask_q, batch)  # Calculate IOU
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())  
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    
    
    # Arguments parsing
    parser = argparse.ArgumentParser(description='AFANet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/ssd/s02009/data/irnet_data/')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)  
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--niter', type=int, default=35)  
    parser.add_argument('--nworker', type=int, default=40)  
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--stage', type=int, default=3)         # 迭代次数！！！！
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
       
    # parser.add_argument('--traincampath', type=str, default='/opt/data/private/Out/fp_irnet/CAM_VOC_Train/')
    # parser.add_argument('--valcampath', type=str, default='/opt/data/private/Out/fp_irnet/CAM_VOC_Val/')
    
    parser.add_argument('--traincampath', type=str, default='/ssd/s02009/out/fp_irnet/njust/coco_cam/')
    parser.add_argument('--valcampath', type=str, default='/ssd/s02009/out/fp_irnet/njust/coco_cam/')
    
    parser.add_argument('--seed', type=int, default=6776)   


    args = parser.parse_args()

    setup_seed(args.seed) # fix seed
    ####
    Logger.initialize(args, training=True)
    # assert args.bsz % torch.cuda.device_count() == 0



    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    clip_model, _ = clip.load('RN50', device= device, jit=False)
    model = afanet(args.backbone, False, args.benchmark, clip_model)
    Logger.log_params(model)

    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()  

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)
    
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn',1,
                                                 cam_train_path=args.traincampath, cam_val_path=args.valcampath)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val',1,
                                                 cam_train_path=args.traincampath, cam_val_path=args.valcampath)

    # Train AFANet

    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    best_epoch = float()
    for epoch in range(args.niter):  
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True,
                                               stage=args.stage)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False,
                                                   stage=args.stage)
        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()

        linux_os_epoch_time = datetime.now()
        epoch_total_time = linux_os_epoch_time - linux_os_start_time
        print(f'epoch: {epoch}, total_time:{epoch_total_time}')
        
    print(f"epoch:{best_epoch} best_val_miou: {best_val_miou}")
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
    
    # Finish time statistics
    linux_os_end_time = datetime.now()
    total_time = linux_os_end_time - linux_os_start_time

    print(f'Start time: {linux_os_start_time}')
    print(f'end_time: {linux_os_end_time}')
    print(f'total_time:{total_time}')
