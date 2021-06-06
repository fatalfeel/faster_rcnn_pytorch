import argparse
import os
import random
import torch

from dataset.base   import DatasetBase
from backbone.base  import BackboneBase
from torchvision.transforms import transforms
from PIL import ImageDraw
from bbox import BBox
from model import Model
#from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config

def str2bool(b_str):
    if b_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',                type=str,   default='voc2007',      help='name of dataset')
parser.add_argument('--backbone',               type=str,   default='resnet101',    help='resnet18, resnet50, resnet101')
parser.add_argument('--checkpoint_dir',         type=str,   default='./checkpoint', help='path to checkpoint')
parser.add_argument('--probability_threshold',  type=float, default=0.6,            help='threshold of detection probability')
parser.add_argument('--image_min_side',         type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
parser.add_argument('--image_max_side',         type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
parser.add_argument('--anchor_ratios',          type=str,   help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
parser.add_argument('--anchor_sizes',           type=str,   help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
#parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
parser.add_argument('--rpn_pre_nms_top_n',      type=int,   help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
parser.add_argument('--rpn_post_nms_top_n',     type=int,   help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
parser.add_argument('--input',  type=str,       default='./input/test.jpg',     help='path to input image')
parser.add_argument('--output', type=str,       default='./output/result.jpg',  help='path to output result image')
parser.add_argument('--cuda',   type=str2bool,  default=False)
args = parser.parse_args()

def _infer(path_to_input_image: str, path_to_output_image: str, dataset_name: str, backbone_name: str, prob_thresh: float):
    device          = torch.device("cuda" if args.cuda else "cpu")

    dataset_class   = DatasetBase.from_name(dataset_name)
    backbone        = BackboneBase.from_name(backbone_name)(pretrained=False)

    model = Model(backbone,
                  dataset_class.num_classes(),
                  #pooler_mode=Config.POOLER_MODE,
                  anchor_ratios     = Config.ANCHOR_RATIOS,
                  anchor_sizes      = Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n = Config.RPN_PRE_NMS_TOP_N,
                  rpn_post_nms_top_n= Config.RPN_POST_NMS_TOP_N).to(device)

    model.load(args.checkpoint_dir)

    with torch.no_grad():
        image               = transforms.Image.open(path_to_input_image)
        image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

        detection_bboxes, \
        detection_classes, \
        detection_probs, _  =  model.eval().forward(image_tensor.unsqueeze(dim=0).to(device))

        detection_bboxes   /= scale

        kept_indices        = (detection_probs > prob_thresh)
        detection_bboxes    = detection_bboxes[kept_indices]
        detection_classes   = detection_classes[kept_indices]
        detection_probs     = detection_probs[kept_indices]

        draw = ImageDraw.Draw(image)

        for bbox, dtclass, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            color       = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox        = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            category    = dataset_class.LABEL_TO_CATEGORY_DICT[dtclass]

            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

        image.save(path_to_output_image)
        print(f'Output image is saved to {path_to_output_image}')


if __name__ == '__main__':
    path_to_input_image     = args.input
    path_to_output_image    = args.output
    dataset_name            = args.dataset
    backbone_name           = args.backbone
    prob_thresh             = args.probability_threshold

    os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)

    Config.setup(image_min_side=args.image_min_side,
                 image_max_side=args.image_max_side,
                 anchor_ratios=args.anchor_ratios,
                 anchor_sizes=args.anchor_sizes,
                 #pooler_mode=args.pooler_mode,
                 rpn_pre_nms_top_n=args.rpn_pre_nms_top_n,
                 rpn_post_nms_top_n=args.rpn_post_nms_top_n)

    print('Arguments:')
    for k, v in vars(args).items():
        print(f'\t{k} = {v}')
    print(Config.describe())

    _infer(path_to_input_image, path_to_output_image, dataset_name, backbone_name, prob_thresh)
