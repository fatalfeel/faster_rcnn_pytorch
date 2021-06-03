import torch
from torch import nn, Tensor
from torch.nn import functional as tnf
#from support.layer.nms import nms
from torchvision import ops
from typing import Union, Tuple, Optional
from bbox import BBox

class RoiPooler(object):
    def __init__(self):
        self.scale          = 1 / 16
        self.output_size    = (7 * 2, 7 * 2)
        self.RoIAlign       = ops.RoIAlign(self.output_size, self.scale, 0, True)

    def apply(self,
              resnet_features: Tensor,
              proposal_gen_bboxes: Tensor,
              batch_indices: Tensor) -> Tensor:

        #pool               = self.RoIAlign(resnet_features, torch.cat([batch_indices.view(-1, 1).float(), proposal_gen_bboxes], dim=1))
        proposal_indices    = batch_indices.view(-1, 1).float()
        indices_bboxes      = torch.cat([proposal_indices, proposal_gen_bboxes], dim=1)
        pool                = self.RoIAlign(resnet_features, indices_bboxes)
        pool                = tnf.max_pool2d(input=pool, kernel_size=2, stride=2)

        return pool

class Detection(nn.Module):
    # def __init__(self, pooler_mode: Pooler.Mode, hidden: nn.Module, num_hidden_out: int, num_classes: int, proposal_smooth_l1_loss_beta: float):
    def __init__(self,
                 hidden_layer: nn.Module,
                 num_hidden_out: int,
                 num_classes: int,
                 proposal_smooth_l1_loss_beta: float):
        super().__init__()
        # self._pooler_mode = pooler_mode
        self.hidden_layer                   = hidden_layer
        self.num_classes                    = num_classes
        self._proposal_class                = nn.Linear(num_hidden_out, num_classes)
        self._proposal_boxpred              = nn.Linear(num_hidden_out, num_classes * 4)
        self._proposal_smooth_l1_loss_beta  = proposal_smooth_l1_loss_beta
        self._detectbox_normalize_mean      = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float)
        self._detectbox_normalize_std       = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float)
        self.roipooler                      = RoiPooler()

    def forward(self,
                resnet_features: Tensor,
                proposal_gen_bboxes: Tensor,
                gt_bboxes_batch: Optional[Tensor] = None,
                gt_labels_batch: Optional[Tensor] = None,) -> Union[Tuple[Tensor, Tensor],
                                                                    Tuple[Tensor, Tensor, Tensor, Tensor]]:
        batch_size = resnet_features.shape[0]

        if self.training:
            # find labels for each `proposal_gen_bboxes`
            labels = torch.full((batch_size, proposal_gen_bboxes.shape[1]), -1, dtype=torch.long, device=proposal_gen_bboxes.device)
            ious = BBox.getIoUs(proposal_gen_bboxes, gt_bboxes_batch)
            proposal_max_ious, proposal_assignments = ious.max(dim=2) #row max value, column index
            labels[proposal_max_ious < 0.5] = 0
            fg_masks = (proposal_max_ious >= 0.5)

            # if len(fg_masks.nonzero()) > 0:
            #    labels[fg_masks] = gt_labels_batch[fg_masks.nonzero()[:, 0], proposal_assignments[fg_masks]]
            #fg_indices = torch.nonzero(fg_masks)                       # get ture row_index, col_index of fg_masks
            #if len(fg_indices) > 0:                                    # make sure there is true index to process
            #morethan_p5_row     = fg_indices[:, 0]                      # > 0.5 row index
            morethan_p5_col     = proposal_assignments[fg_masks]        # if fg_masks is ture then give column index
            #gt_match_class      = gt_labels_batch[morethan_p5_row, morethan_p5_col]
            gt_match_class      = gt_labels_batch[0, morethan_p5_col]   #col pick class in gt_labels_batch
            labels[fg_masks]    = gt_match_class                        #if fg_masks is ture then give gt_class number

            # select 128 x `batch_size` samples
            '''
            fg_indices  = (labels > 0).nonzero()
            bg_indices  = (labels == 0).nonzero()
            '''
            fg_indices  = torch.nonzero(labels > 0) #same as torch.nonzero(fg_masks)
            bg_indices  = torch.nonzero(labels == 0)

            #fg_samples = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 32 * batch_size)]]
            #bg_samples = bg_indices[torch.randperm(len(bg_indices))[:128 * batch_size - len(fg_samples)]]
            fg_rand     = torch.randperm(len(fg_indices))       # random number 1~len
            fg_size     = min(len(fg_indices), 32 * batch_size) # select min value
            fg_range    = fg_rand[:fg_size]                     # #pick 0~fg_size of random number
            fg_samples  = fg_indices[fg_range]

            bg_rand     = torch.randperm(len(bg_indices))
            bg_size     = 128 * batch_size - len(fg_samples)
            bg_range    = bg_rand[:bg_size]
            bg_samples  = bg_indices[bg_range]

            #selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
            #selected_indices   = selected_indices[torch.randperm(len(selected_indices))].unbind(dim=1)
            fgbg_samples        = torch.cat([fg_samples, bg_samples], dim=0)
            selected_rand       = torch.randperm(len(fgbg_samples))
            selected_indices    = fgbg_samples[selected_rand].unbind(dim=1)

            proposal_gen_bboxes = proposal_gen_bboxes[selected_indices]
            gt_bboxes           = gt_bboxes_batch[selected_indices[0], proposal_assignments[selected_indices]]
            gt_proposal_classes = labels[selected_indices]
            gt_proposal_offset  = BBox.offset_from_gt_center(proposal_gen_bboxes, gt_bboxes)
            batch_indices       = selected_indices[0]

            # pool = Pooler.apply(resnet_features, proposal_gen_bboxes, proposal_batch_indices=batch_indices, mode=self._pooler_mode)
            pool    = self.roipooler.apply(resnet_features, proposal_gen_bboxes, batch_indices)
            hidden  = self.hidden_layer(pool)
            hidden  = tnf.adaptive_max_pool2d(input=hidden, output_size=1)
            hidden  = hidden.view(hidden.shape[0], -1)

            proposal_classes    = self._proposal_class(hidden)
            proposal_boxpred    = self._proposal_boxpred(hidden)
            proposal_class_losses, proposal_boxpred_losses = self.getLoss(proposal_classes,
                                                                          proposal_boxpred,
                                                                          gt_proposal_classes,
                                                                          gt_proposal_offset,
                                                                          batch_size,
                                                                          batch_indices  )

            return proposal_classes, proposal_boxpred, proposal_class_losses, proposal_boxpred_losses

        else:
            batch_indices = torch.arange(end    = batch_size,
                                         dtype  = torch.long,
                                         device = proposal_gen_bboxes.device).view(-1, 1).repeat(1, proposal_gen_bboxes.shape[1])
            # pool = Pooler.apply(resnet_features, proposal_gen_bboxes.view(-1, 4), batch_indices.view(-1), mode=self._pooler_mode)
            pool    = self.roipooler.apply(resnet_features, proposal_gen_bboxes.view(-1, 4), batch_indices.view(-1))
            hidden  = self.hidden_layer(pool)
            hidden  = tnf.adaptive_max_pool2d(input=hidden, output_size=1)
            hidden  = hidden.view(hidden.shape[0], -1)

            proposal_classes    = self._proposal_class(hidden)
            proposal_boxpred    = self._proposal_boxpred(hidden)

            proposal_classes    = proposal_classes.view(batch_size, -1, proposal_classes.shape[-1])
            proposal_boxpred    = proposal_boxpred.view(batch_size, -1, proposal_boxpred.shape[-1])

            return proposal_classes, proposal_boxpred

    def generate_detections(self,
                            proposal_gen_bboxes: Tensor,
                            proposal_classes: Tensor,
                            proposal_boxpred: Tensor,
                            image_width: int,
                            image_height: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size                  = proposal_gen_bboxes.shape[0]
        proposal_boxpred            = proposal_boxpred.view(batch_size, -1, self.num_classes, 4)
        detectbox_normalize_std     = self._detectbox_normalize_std.to(device=proposal_boxpred.device)
        detectbox_normalize_mean    = self._detectbox_normalize_mean.to(device=proposal_boxpred.device)
        proposal_boxpred            = proposal_boxpred * detectbox_normalize_std + detectbox_normalize_mean

        proposal_gen_bboxes         = proposal_gen_bboxes.unsqueeze(dim=2).repeat(1, 1, self.num_classes, 1)
        pred_offset                 = BBox.offset_form_pred_ltrb(proposal_gen_bboxes, proposal_boxpred)
        detection_bboxes            = BBox.clip(pred_offset, left=0, top=0, right=image_width, bottom=image_height)
        detection_probs             = tnf.softmax(proposal_classes, dim=-1)

        all_detection_bboxes        = []
        all_detection_classes       = []
        all_detection_probs         = []
        all_detection_batch_indices = []

        for batch_index in range(batch_size):
            for c in range(1, self.num_classes):
                class_bboxes = detection_bboxes[batch_index, :, c, :]
                class_probs = detection_probs[batch_index, :, c]
                threshold = 0.3
                # kept_indices = nms(class_bboxes, class_probs, threshold)
                kept_indices    = ops.nms(class_bboxes, class_probs, threshold)
                class_bboxes    = class_bboxes[kept_indices]
                class_probs     = class_probs[kept_indices]

                all_detection_bboxes.append(class_bboxes)
                all_detection_classes.append(torch.full((len(kept_indices),), c, dtype=torch.int))
                all_detection_probs.append(class_probs)
                all_detection_batch_indices.append(torch.full((len(kept_indices),), batch_index, dtype=torch.long))

        all_detection_bboxes        = torch.cat(all_detection_bboxes, dim=0)
        all_detection_classes       = torch.cat(all_detection_classes, dim=0)
        all_detection_probs         = torch.cat(all_detection_probs, dim=0)
        all_detection_batch_indices = torch.cat(all_detection_batch_indices, dim=0)

        return all_detection_bboxes, all_detection_classes, all_detection_probs, all_detection_batch_indices

    def getLoss(self,
                proposal_classes: Tensor,
                proposal_boxpred: Tensor,
                gt_proposal_classes: Tensor,
                gt_proposal_offset: Tensor,
                batch_size,
                batch_indices) -> Tuple[Tensor, Tensor]:

        proposal_boxpred            = proposal_boxpred.view(-1, self.num_classes, 4)[torch.arange(end=len(proposal_boxpred), dtype=torch.long), gt_proposal_classes]
        detectbox_normalize_mean    = self._detectbox_normalize_mean.to(device=gt_proposal_offset.device)
        detectbox_normalize_std     = self._detectbox_normalize_std.to(device=gt_proposal_offset.device)
        gt_proposal_offset          = (gt_proposal_offset - detectbox_normalize_mean) / detectbox_normalize_std  # scale up target to make regressor easier to learn

        cross_entropies     = torch.empty(batch_size, dtype=torch.float, device=proposal_classes.device)
        smooth_l1_losses    = torch.empty(batch_size, dtype=torch.float, device=proposal_boxpred.device)

        for batch_index in range(batch_size):
            # selected_indices = (batch_indices == batch_index).nonzero().view(-1)
            selected_indices = torch.nonzero(batch_indices == batch_index).view(-1)

            cross_entropy = tnf.cross_entropy(input =proposal_classes[selected_indices],
                                              target=gt_proposal_classes[selected_indices])

            # fg_indices = gt_proposal_classes[selected_indices].nonzero().view(-1)
            fg_indices = torch.nonzero(gt_proposal_classes[selected_indices]).view(-1)

            smooth_l1_loss = self.beta_smooth_l1_loss(  pred    = proposal_boxpred[selected_indices][fg_indices],
                                                        offset  = gt_proposal_offset[selected_indices][fg_indices],
                                                        beta    = self._proposal_smooth_l1_loss_beta  )

            cross_entropies[batch_index]    = cross_entropy
            smooth_l1_losses[batch_index]   = smooth_l1_loss

        return cross_entropies, smooth_l1_losses

    def beta_smooth_l1_loss(self, pred: Tensor, offset: Tensor, beta: float) -> Tensor:
        diff = torch.abs(pred - offset)
        # loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        # =(flag * 0.5 * (diff ** 2)/beta + (1 - flag) * (diff - 0.5 * beta)
        loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
        loss = loss.sum() / (pred.numel() + 1e-8)
        return loss
