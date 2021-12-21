import argparse
import os
import time
import uuid
import torch
from dataset.baseobject import DatasetBase
from backbone.basenet import BackboneBase
from config.eval_config import EvalConfig as Config
from evaluator import Evaluator
from logger import Logger as Log
from model import Model

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',        type=str,   default='voc2007',      help='name of dataset')
parser.add_argument('--backbone',       type=str,   default='resnet101',    help='resnet18, resnet50, resnet101')
parser.add_argument('--data_dir',       type=str,   default='./data',       help='path to data directory')
parser.add_argument('--checkpoint',     type=str,   default='./checkpoint', help='path to checkpoint')
parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
parser.add_argument('--anchor_ratios',  type=str,   help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
parser.add_argument('--anchor_sizes',   type=str,   help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
#parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
parser.add_argument('--anchor_smooth_l1_loss_beta', type=float, help='default: {:g}'.format(Config.ANCHOR_SMOOTH_L1_LOSS_BETA))
parser.add_argument('--proposal_smooth_l1_loss_beta', type=float, help='default: {:g}'.format(Config.PROPOSAL_SMOOTH_L1_LOSS_BETA))
parser.add_argument('--batch_size', type=int, help='default: {:g}'.format(Config.BATCH_SIZE))
parser.add_argument('--learning_rate', type=float, help='default: {:g}'.format(Config.LEARNING_RATE))
parser.add_argument('--momentum', type=float, help='default: {:g}'.format(Config.MOMENTUM))
parser.add_argument('--weight_decay', type=float, help='default: {:g}'.format(Config.WEIGHT_DECAY))
parser.add_argument('--step_lr_sizes', type=str, help='default: {!s}'.format(Config.STEP_LR_SIZES))
parser.add_argument('--step_lr_gamma', type=float, help='default: {:g}'.format(Config.STEP_LR_GAMMA))
parser.add_argument('--warm_up_factor', type=float, help='default: {:g}'.format(Config.WARM_UP_FACTOR))
parser.add_argument('--warm_up_num_iters', type=int, help='default: {:d}'.format(Config.WARM_UP_NUM_ITERS))
parser.add_argument('--cuda', default=False, type=str2bool)
args = parser.parse_args()

device  = torch.device("cuda" if args.cuda else "cpu")

def _eval(path_to_checkpoint: str, dataset_name: str, backbone_name: str, path_to_data_dir: str, path_to_results_dir: str):
    dataset = DatasetBase.from_name(dataset_name)(path_to_data_dir, DatasetBase.Mode.EVAL, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
    evaluator = Evaluator(dataset, path_to_data_dir, path_to_results_dir)

    Log.i('Found {:d} samples'.format(len(dataset)))

    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    model = Model(backbone,
                  dataset.num_classes(),
                  #pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS,
                  anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N,
                  rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).to(device)

    model.load(path_to_checkpoint)

    Log.i('Start evaluating with 1 GPU (1 batch per GPU)')
    mean_ap, detail = evaluator.evaluate(model)
    Log.i('Done')

    Log.i('mean AP = {:.4f}'.format(mean_ap))
    Log.i('\n' + detail)

if __name__ == '__main__':
#def main():
    '''parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
    parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
    parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to data directory')
    parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
    parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
    parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
    parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
    parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
    parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
    parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
    parser.add_argument('checkpoint', type=str, help='path to evaluating checkpoint')
    args = parser.parse_args()'''

    path_to_checkpoint = args.checkpoint
    dataset_name = args.dataset
    backbone_name = args.backbone
    path_to_data_dir = args.data_dir

    path_to_results_dir = os.path.join(os.path.dirname(path_to_checkpoint),
                                       'results-{:s}-{:s}-{:s}'.format( time.strftime('%Y%m%d%H%M%S'),
                                                                        path_to_checkpoint.split(os.path.sep)[-1].split(os.path.curdir)[0],
                                                                        str(uuid.uuid4()).split('-')[0]))
    os.makedirs(path_to_results_dir)

    Config.setup(image_min_side=args.image_min_side,
                 image_max_side=args.image_max_side,
                 anchor_ratios=args.anchor_ratios,
                 anchor_sizes=args.anchor_sizes,
                 #pooler_mode=args.pooler_mode,
                 rpn_pre_nms_top_n=args.rpn_pre_nms_top_n,
                 rpn_post_nms_top_n=args.rpn_post_nms_top_n)

    Log.initialize(os.path.join(path_to_results_dir, 'eval.log'))
    Log.i('Arguments:')
    for k, v in vars(args).items():
        Log.i(f'\t{k} = {v}')
    Log.i(Config.describe())

    _eval(path_to_checkpoint, dataset_name, backbone_name, path_to_data_dir, path_to_results_dir)
#main()
