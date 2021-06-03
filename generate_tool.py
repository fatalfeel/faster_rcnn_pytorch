import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as tnf
from torchvision import ops
from typing import Tuple, List
from bbox import BBox

# refer to: simple-faster-rcnn/rpn/creator_tools.py
class GenerateTool(object):
    def __init__(self,
                 anchor_ratios:     List[Tuple[int, int]],
                 anchor_sizes:      List[int],
                 pre_nms_top_n:     int,
                 post_nms_top_n:    int):

        self._anchor_ratios     = anchor_ratios
        self._anchor_sizes      = anchor_sizes
        self._pre_nms_top_n     = pre_nms_top_n
        self._post_nms_top_n    = post_nms_top_n

    def anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int) -> Tensor:
        '''array[num_y_anchors + 2][1:-1] = array[1] ~ array[num_y_anchors+2-2]'''
        y_center    = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
        x_center    = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]
        #ratios      = np.array(self._anchor_ratios)
        #ratios      = ratios[:, 0] / ratios[:, 1]
        sizes       = np.array(self._anchor_sizes) #16 * scale[8, 16, 32]

        '''np.set_printoptions(threshold=np.inf)
        print(x_center)
        print('')
        print(y_center)
        y_center , x_center = np.meshgrid(y_center, x_center, indexing='ij')
        print(x_center)
        print('')
        print(y_center)
        plt.plot(x_center, y_center, marker='.', color='r', linestyle='none')
        plt.show()'''

        # combine x[], y[] to a mesh grid
        y_center, x_center, ratios, sizes = np.meshgrid(y_center, x_center, self._anchor_ratios, sizes, indexing='ij')

        # to 1d
        y_center    = y_center.reshape(-1)
        x_center    = x_center.reshape(-1)
        ratios      = ratios.reshape(-1)
        sizes       = sizes.reshape(-1)

        #widths  = sizes * np.sqrt(1.0 / ratios)
        #heights = sizes * np.sqrt(ratios)
        h_ratios = np.sqrt(ratios) #faster way
        heights  = sizes * h_ratios
        widths   = sizes * (1.0 / h_ratios)

        center_based_anchor_bboxes  = np.stack((x_center, y_center, widths, heights), axis=1)
        center_based_anchor_bboxes  = torch.from_numpy(center_based_anchor_bboxes).float()
        anchor_gen_bboxes           = BBox.lt_rb_transform(center_based_anchor_bboxes)

        return anchor_gen_bboxes

    def proposals(self,
                  anchor_gen_bboxes:    Tensor,
                  anchor_score:         Tensor,
                  anchor_pred_bboxes:   Tensor,
                  image_width:  int,
                  image_height: int) -> Tensor:
        nms_proposal_bboxes_batch   = []
        padded_proposal_bboxes      = []

        batch_size          = anchor_gen_bboxes.shape[0]
        pred_offset         = BBox.offset_form_pred_ltrb(anchor_gen_bboxes, anchor_pred_bboxes)
        proposal_bboxes     = BBox.clip(pred_offset, left=0, top=0, right=image_width, bottom=image_height)
        proposal_fg_probs   = tnf.softmax(anchor_score[:, :, 1], dim=-1)
        _, sorted_indices   = torch.sort(proposal_fg_probs, dim=-1, descending=True)

        for batch_index in range(batch_size):
            sorted_bboxes   = proposal_bboxes[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]
            sorted_probs    = proposal_fg_probs[batch_index][sorted_indices[batch_index]][:self._pre_nms_top_n]
            threshold       = 0.7
            #kept_indices = nms(sorted_bboxes, sorted_probs, threshold)
            kept_indices    = ops.nms(sorted_bboxes, sorted_probs, threshold)
            nms_bboxes      = sorted_bboxes[kept_indices][:self._post_nms_top_n] #keep the most is 2000 bboxes
            nms_proposal_bboxes_batch.append(nms_bboxes)

        #compare each list which have max len
        max_nms_proposal_bboxes_length = max([len(it) for it in nms_proposal_bboxes_batch])

        # if nms_proposal_bboxes not enough to max len then add zeros
        for nms_proposal_bboxes in nms_proposal_bboxes_batch:
            '''padded_proposal_bboxes.append(torch.cat([nms_proposal_bboxes,
                                                        torch.zeros(max_nms_proposal_bboxes_length - len(nms_proposal_bboxes), 4).to(nms_proposal_bboxes)]))'''
            remain = max_nms_proposal_bboxes_length - len(nms_proposal_bboxes)
            if remain > 0:
                zeros       = torch.zeros(remain, 4).to(nms_proposal_bboxes)
                nms_zeros   = torch.cat([nms_proposal_bboxes, zeros])
                padded_proposal_bboxes.append(nms_zeros)
            else:
                padded_proposal_bboxes.append(nms_proposal_bboxes)

        padded_proposal_bboxes = torch.stack(padded_proposal_bboxes, dim=0)

        return padded_proposal_bboxes
