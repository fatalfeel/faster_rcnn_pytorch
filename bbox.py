import torch
from torch import Tensor
from typing import List

class BBox(object):
    def __init__(self, left: float, top: float, right: float, bottom: float):
        super().__init__()
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self) -> str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}]'.format(
            self.left, self.top, self.right, self.bottom)

    def tolist(self) -> List[float]:
        return [self.left, self.top, self.right, self.bottom]

    '''(x0,y0,x1,y1) to (c0,c1,w,h)'''
    @staticmethod
    #def to_center_base(bboxes: Tensor):
    def center_wh_transform(bboxes: Tensor) -> Tensor:
        center_wh = torch.stack(   [(bboxes[..., 0] + bboxes[..., 2]) / 2,
                                    (bboxes[..., 1] + bboxes[..., 3]) / 2,
                                    bboxes[..., 2] - bboxes[..., 0],
                                    bboxes[..., 3] - bboxes[..., 1]],
                                    dim=-1)

        return center_wh

    '''(c0,c1,w,h) to (x0,y0,x1,y1)'''
    @staticmethod
    #def from_center_base(center_based_bboxes: Tensor) -> Tensor:
    def lt_rb_transform(center_based_bboxes: Tensor) -> Tensor:
        '''return torch.stack([center_based_bboxes[..., 0] - center_based_bboxes[..., 2] / 2,
                            center_based_bboxes[..., 1] - center_based_bboxes[..., 3] / 2,
                            center_based_bboxes[..., 0] + center_based_bboxes[..., 2] / 2,
                            center_based_bboxes[..., 1] + center_based_bboxes[..., 3] / 2],
                            dim=-1)'''
        half_width  = center_based_bboxes[..., 2] / 2
        half_height = center_based_bboxes[..., 3] / 2
        return torch.stack([center_based_bboxes[..., 0] - half_width,
                            center_based_bboxes[..., 1] - half_height,
                            center_based_bboxes[..., 0] + half_width,
                            center_based_bboxes[..., 1] + half_height],
                            dim=-1)

    '''
    https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11
    same as bbox2loc of simple-faster-rcnn
    localization loss using
    '''
    @staticmethod
    #def calc_transformer(gen_bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
    def offset_from_gt_center(gen_bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
        gen_bboxes_centerbase   = BBox.center_wh_transform(gen_bboxes)
        gt_bboxes_centerbase    = BBox.center_wh_transform(gt_bboxes)
        # log is ln
        ''' (x1-x0)/w0, 
            (y1-y0)/h0, 
            ln(w1/w0), 
            ln(h1/h0) '''
        gt_offset = torch.stack([(gt_bboxes_centerbase[..., 0] - gen_bboxes_centerbase[..., 0]) / gen_bboxes_centerbase[..., 2],
                                 (gt_bboxes_centerbase[..., 1] - gen_bboxes_centerbase[..., 1]) / gen_bboxes_centerbase[..., 3],
                                 torch.log(gt_bboxes_centerbase[..., 2] / gen_bboxes_centerbase[..., 2]),
                                 torch.log(gt_bboxes_centerbase[..., 3] / gen_bboxes_centerbase[..., 3])],
                                 dim=-1)
        return gt_offset

    '''
    https://blog.csdn.net/dongapple/article/details/75468810
    same as loc2bbox of simple-faster-rcnn
    confidence using
    '''
    @staticmethod
    #def apply_transformer(src_bboxes: Tensor, transformers: Tensor) -> Tensor:
    def offset_form_pred_ltrb(gen_bboxes: Tensor, pred_bboxes: Tensor) -> Tensor:
        gen_bboxes_centerbase = BBox.center_wh_transform(gen_bboxes)
        # exp is e^n,
        ''' x1*w0 + x0, 
            y1*h0 + y0, 
            e^(w1*w0),
            e^(h1*h0) '''
        pred_offset = torch.stack( [pred_bboxes[..., 0] * gen_bboxes_centerbase[..., 2] + gen_bboxes_centerbase[..., 0],
                                    pred_bboxes[..., 1] * gen_bboxes_centerbase[..., 3] + gen_bboxes_centerbase[..., 1],
                                    torch.exp(pred_bboxes[..., 2]) * gen_bboxes_centerbase[..., 2],
                                    torch.exp(pred_bboxes[..., 3]) * gen_bboxes_centerbase[..., 3]],
                                    dim=-1 )

        pred_offset = BBox.lt_rb_transform(pred_offset)

        return pred_offset

    @staticmethod
    def getIoUs(source: Tensor, gtboxes: Tensor) -> Tensor:
        #torch.set_printoptions(profile="full")
        #source, gtboxes = source.unsqueeze(dim=-2).repeat(1, 1, gtboxes.shape[-2], 1), gtboxes.unsqueeze(dim=-3).repeat(1, source.shape[-2], 1, 1)
        source_ext      = source.unsqueeze(dim=-2).repeat(1, 1, gtboxes.shape[-2], 1)
        gtboxes_ext     = gtboxes.unsqueeze(dim=-3).repeat(1, source.shape[-2], 1, 1)

        #get max point
        intersection_left   = torch.max(source_ext[..., 0], gtboxes_ext[..., 0])
        intersection_top    = torch.max(source_ext[..., 1], gtboxes_ext[..., 1])
        intersection_right  = torch.min(source_ext[..., 2], gtboxes_ext[..., 2])
        intersection_bottom = torch.min(source_ext[..., 3], gtboxes_ext[..., 3])
        intersection_width  = torch.clamp(intersection_right - intersection_left, min=0)
        intersection_height = torch.clamp(intersection_bottom - intersection_top, min=0)
        intersection_area   = intersection_width * intersection_height

        # source and gtboxes rectangle area
        source_area     = (source_ext[..., 2] - source_ext[..., 0]) * (source_ext[..., 3] - source_ext[..., 1])
        gtboxes_area    = (gtboxes_ext[..., 2] - gtboxes_ext[..., 0]) * (gtboxes_ext[..., 3] - gtboxes_ext[..., 1])
        ious            = intersection_area / (source_area + gtboxes_area - intersection_area)

        return ious

    @staticmethod
    def InsideBound(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        #return ((bboxes[..., 0] >= left) * (bboxes[..., 1] >= top) * (bboxes[..., 2] <= right) * (bboxes[..., 3] <= bottom))
        b0  = bboxes[..., 0] >= left
        b1  = bboxes[..., 1] >= top
        b2  = bboxes[..., 2] <= right
        b3  = bboxes[..., 3] <= bottom
        ret = b0*b1*b2*b3

        return ret

    @staticmethod
    def clip(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=left, max=right)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=top, max=bottom)

        return bboxes
