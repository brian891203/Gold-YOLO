import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.assigners.assigner_utils import (dist_calculator, iou_calculator,
                                             select_candidates_in_gts,
                                             select_highest_overlaps)


class TaskAlignedAssigner(nn.Module):
    def __init__(self,
                 topk=13,
                 num_classes=80,
                 alpha=1.0,
                 beta=6.0,
                 eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    @torch.no_grad()
    def forward(self,
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), \
                torch.zeros_like(pd_bboxes).to(device), \
                torch.zeros_like(pd_scores).to(device), \
                torch.zeros_like(pd_scores[..., 0]).to(device)
        
        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_boxes <= 100 else (self.bs, 1, 1)
        target_labels_lst, target_bboxes_lst, target_scores_lst, fg_mask_lst = [], [], [], []
        # loop batch dim in case of numerous object box
        for i in range(cycle):
            start, end = i * step, (i + 1) * step
            pd_scores_ = pd_scores[start:end, ...]
            pd_bboxes_ = pd_bboxes[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            mask_gt_ = mask_gt[start:end, ...]
            
            mask_pos, align_metric, overlaps = self.get_pos_mask(
                    pd_scores_, pd_bboxes_, gt_labels_, gt_bboxes_, anc_points, mask_gt_)
            
            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
                    mask_pos, overlaps, self.n_max_boxes)
            
            # assigned target
            target_labels, target_bboxes, target_scores = self.get_targets(
                    gt_labels_, gt_bboxes_, target_gt_idx, fg_mask)
            
            # # normalize
            # align_metric *= mask_pos
            # pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
            # pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
            # norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
            # target_scores = target_scores * norm_align_metric

            # normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
            pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]

            # --- Safely calculate norm_align_metric ---
            # Create a mask where the denominator is safe (greater than eps)
            # Expand the mask to match the shape of align_metric for indexing
            safe_division_mask = (pos_align_metrics > self.eps).expand_as(align_metric) # MODIFIED: Expand mask

            # Initialize norm_align_metric_raw with zeros
            norm_align_metric_raw = torch.zeros_like(align_metric)

            # Perform division only where it's safe
            # Use the expanded mask for indexing both sides
            # Also expand pos_align_metrics for the division
            # Ensure pos_overlaps is also expanded correctly for multiplication
            pos_overlaps_expanded = pos_overlaps.expand_as(align_metric)
            pos_align_metrics_expanded = pos_align_metrics.expand_as(align_metric)

            # Check shapes before division if needed
            # print("align_metric shape:", align_metric.shape)
            # print("pos_overlaps_expanded shape:", pos_overlaps_expanded.shape)
            # print("pos_align_metrics_expanded shape:", pos_align_metrics_expanded.shape)
            # print("safe_division_mask shape:", safe_division_mask.shape)

            # Perform division using expanded tensors and mask
            numerator = (align_metric * pos_overlaps_expanded)[safe_division_mask]
            denominator = pos_align_metrics_expanded[safe_division_mask]
            norm_align_metric_raw[safe_division_mask] = numerator / denominator

            # Get the max and unsqueeze
            norm_align_metric = norm_align_metric_raw.max(-2)[0].unsqueeze(-1)

            # --- Clamp norm_align_metric ---
            # Ensure norm_align_metric is not NaN/Inf and clamp it to [0, 1]
            # This prevents multiplying target_scores by extreme values.
            norm_align_metric = torch.nan_to_num(norm_align_metric, nan=0.0, posinf=1.0, neginf=0.0)
            norm_align_metric = torch.clamp(norm_align_metric, 0.0, 1.0)
            # --- Add check for norm_align_metric ---
            # 檢查計算出的 norm_align_metric 是否有效
            if torch.isnan(norm_align_metric).any() or torch.isinf(norm_align_metric).any():
                print(f"DEBUG Assigner: norm_align_metric contains NaN/Inf!")
            if (norm_align_metric < 0.0).any():
                print(f"DEBUG Assigner: norm_align_metric contains negative values! Min: {norm_align_metric.min().item()}")
            # --- End check ---
            target_scores = target_scores * norm_align_metric # 歸一化目標分數

            # --- Add check after multiplication ---
            # 檢查乘法後 target_scores 是否有效
            if torch.isnan(target_scores).any() or torch.isinf(target_scores).any():
                print(f"DEBUG Assigner (Post-Norm): target_scores contains NaN/Inf!")
            min_ts_post_norm = target_scores.min().item()
            max_ts_post_norm = target_scores.max().item()
            if min_ts_post_norm < 0.0 or max_ts_post_norm > 1.0:
                print(f"DEBUG Assigner (Post-Norm): target_scores 超出 [0, 1]! Min: {min_ts_post_norm}, Max: {max_ts_post_norm}")
            # --- End check ---
            
            # append
            target_labels_lst.append(target_labels)
            target_bboxes_lst.append(target_bboxes)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)
        
        # concat
        target_labels = torch.cat(target_labels_lst, 0)
        target_bboxes = torch.cat(target_bboxes_lst, 0)
        target_scores = torch.cat(target_scores_lst, 0)
        fg_mask = torch.cat(fg_mask_lst, 0)
        
        return target_labels, target_bboxes, target_scores, fg_mask.bool()
    
    def get_pos_mask(self,
                     pd_scores,
                     pd_bboxes,
                     gt_labels,
                     gt_bboxes,
                     anc_points,
                     mask_gt):
        
        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask
        mask_topk = self.select_topk_candidates(
                align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask
        mask_pos = mask_topk * mask_in_gts * mask_gt
        
        return mask_pos, align_metric, overlaps
    
    def get_box_metrics(self,
                        pd_scores,
                        pd_bboxes,
                        gt_labels,
                        gt_bboxes):
        
        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pd_scores[ind[0], ind[1]]
        
        overlaps = iou_calculator(gt_bboxes, pd_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        return align_metric, overlaps
    
    def select_topk_candidates(self,
                               metrics,
                               largest=True,
                               topk_mask=None):
        
        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(
                metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile(
                    [1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1,
                                 torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.to(metrics.dtype)
    
    def get_targets(self,
                    gt_labels,
                    gt_bboxes,
                    target_gt_idx,
                    fg_mask):
        
        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        
        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]
        
        # assigned target scores
        target_labels[target_labels < 0] = 0
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores,
                                    torch.full_like(target_scores, 0))
        
        return target_labels, target_bboxes, target_scores
