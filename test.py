r""" AFANet testing code  """
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn.functional as F
import torch.nn as nn
import torch
from model.AFANet import afanet
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

import clip


def test(model, dataloader, nshot, stage):
    r""" Test AFANet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(6776)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)

        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot, class_id=batch['class_id'],
                                                    stage=stage)  # AFANet
        
        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask,
                                                  batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float(),
                                                  batch['query_name'])

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='AFANet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/opt/data/private/Data/afanet_data/')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)  # must be 1
    parser.add_argument('--nworker', type=int, default=32)
    parser.add_argument('--load', type=str,
                        default='/opt/data/private/Out/new_work_logs/afa_4/ori_voc_67.76/best_model.pt')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', default='visualize')  # or just: default='visualize' action='store_true'
    parser.add_argument('--use_original_imgsize', action='store_true')  # action='store_true'
    parser.add_argument('--stage', type=int, default=2)  ##  coco set 3

    parser.add_argument('--traincampath', type=str, default='/opt/data/private/Out/afanet/voc_cam/CAM_VOC_Train/')
    parser.add_argument('--valcampath', type=str, default='/opt/data/private/Out/afanet/voc_cam/CAM_VOC_Val/')

    # parser.add_argument('--traincampath', type=str, default='/opt/data/private/Out/afanet/coco_cam/')
    # parser.add_argument('--valcampath', type=str, default='/opt/data/private/Out/afanet/coco_cam/')

    parser.add_argument('--vispath', type=str, default='/opt/data/private/Code/new_work/afa_4/vis/')

    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    clip_model, _ = clip.load('RN50', device=device, jit=False)

    model = afanet(args.backbone, args.use_original_imgsize, args.benchmark, clip_model)  # AFANet model
    model.eval()

    Logger.log_params(model)
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '':
        raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load, weights_only=True))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)

    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.nshot,
                                                  cam_train_path=args.traincampath, cam_val_path=args.valcampath)

    # Test AFANet

    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot, args.stage)

    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
