# 2023.09.18-Changed for loss implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.assigners.tal_assigner import TaskAlignedAssigner
from yolov6.utils.figure_iou import IOUloss
from yolov6.utils.general import bbox2dist, box_iou, dist2bbox, xywh2xyxy


class ComputeLoss:
    '''Loss computation func.'''
    
    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=0,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                         'class': 1.0,
                         'iou': 2.5,
                         'dfl': 0.5}
                 ):
        
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size
        
        self.warmup_epoch = warmup_epoch
        self.formal_assigner = TaskAlignedAssigner(topk=26, num_classes=self.num_classes, alpha=1.0, beta=6.0)
        
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight
    
    def __call__(
            self,
            outputs,
            targets,
            epoch_num,
            step_num
    ):
        
        feats, pred_scores, pred_distri = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = \
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                             device=feats[0].device, is_eval=False, mode='ab')
        
        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.full((1, 4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]
        
        # targets
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        
        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_distri[..., :2] += anchor_points_s
        pred_bboxes = xywh2xyxy(pred_distri)
        
        try:
            target_labels, target_bboxes, target_scores, fg_mask = \
                self.formal_assigner(
                        pred_scores.detach(),
                        pred_bboxes.detach() * stride_tensor,
                        anchor_points,
                        gt_labels,
                        gt_bboxes,
                        mask_gt)
            
            # 立即限制 target_scores 範圍
            target_scores = torch.clamp(target_scores, 0.0, 1.0)
        
        except RuntimeError:
            print(
                    "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                        CPU mode is applied in this batch. If you want to avoid this issue, \
                        try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            
            _pred_scores = pred_scores.detach().cpu().float()
            _pred_bboxes = pred_bboxes.detach().cpu().float()
            _anchor_points = anchor_points.cpu().float()
            _gt_labels = gt_labels.cpu().float()
            _gt_bboxes = gt_bboxes.cpu().float()
            _mask_gt = mask_gt.cpu().float()
            _stride_tensor = stride_tensor.cpu().float()
            
            target_labels, target_bboxes, target_scores, fg_mask = \
                self.formal_assigner(
                        _pred_scores,
                        _pred_bboxes * _stride_tensor,
                        _anchor_points,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt)
            
            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()

            # 對 target_scores 進行更全面的檢查和清理
            # 檢查 NaN 和極值
            if torch.isnan(target_scores).any():
                print(f"警告: 在 epoch {epoch_num}, step {step_num} 的 target_scores 中檢測到 NaN 值")
                target_scores = torch.nan_to_num(target_scores, 0.0)
            
            # 統一確保所有標籤值都在合法範圍內
            target_scores = torch.clamp(target_scores, 0.0, 1.0)

            # 安全計算 target_scores_sum
            try:
                target_scores_sum = torch.clamp(target_scores.sum(), min=1e-6)  # 避免除以零
            except Exception as e:
                print(f'計算 target_scores_sum 時出錯: {e}')
                target_scores_sum = torch.tensor(1.0, device=target_scores.device)

            fg_mask = fg_mask.cuda()
        # Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()
        
        # rescale bbox
        target_bboxes /= stride_tensor
        
        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]

        # 在此處添加 clamp 操作 --> 確保目標值在預期的範圍 [0, 1]
        target_scores = torch.clamp(target_scores, 0.0, 1.0)  # 確保目標值在 [0, 1] 範圍內

        # print(f"Before loss_cls - target_scores min: {target_scores.min().item()}, max: {target_scores.max().item()}")
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)
        
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        try:
            target_scores_sum = target_scores.sum()
            if target_scores_sum > 0:
                loss_cls /= target_scores_sum
        except BaseException as e:
            print(f'Loss_fuse ERROR: {e}')
            if not torch.any(torch.isnan(target_scores)):
                target_scores_sum = target_scores.sum()
                if not torch.isnan(target_scores_sum) and target_scores_sum > 0:
                    loss_cls /= target_scores_sum
            else:
                target_scores_sum = 0
        
        # bbox loss
        # print(f"Before bbox_loss - target_scores min: {target_scores.min().item()}, max: {target_scores.max().item()}")
        target_scores = torch.clamp(target_scores, 0.0, 1.0)  

        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)
        
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        
        loss_items = torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                                (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                                (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()
        
        return loss, loss_items
    
    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(
                np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]).to(
                targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets
    
    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(
                    self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()
    
    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        # 添加 clamp 確保值域正確
        gt_score = torch.clamp(gt_score, 0.0, 1.0)
        pred_score = torch.clamp(pred_score, 1e-6, 1.0 - 1e-6)  # 防止 BCE 中的數值問題

        # 檢查是否有 NaN 值並處理
        if torch.isnan(gt_score).any() or torch.isnan(pred_score).any():
            print("警告：在計算損失前檢測到 NaN 值！")
            gt_score = torch.nan_to_num(gt_score, 0.0)
            pred_score = torch.nan_to_num(pred_score, 0.5)
        
        # 確保 weight 計算時不會出現極值
        with torch.no_grad():
            weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
            weight = torch.clamp(weight, min=0.0, max=1000.0)
            
        # 使用 float32 精度以提高數值穩定性
        with torch.amp.autocast(device_type='cuda', enabled=False):
            try:
                # 確保輸入 BCE 的值都在合法範圍
                gt_score_safe = torch.clamp(gt_score.float(), 0.0, 1.0)
                pred_score_safe = torch.clamp(pred_score.float(), 1e-6, 1.0 - 1e-6)
                loss = F.binary_cross_entropy(
                    pred_score_safe, 
                    gt_score_safe, 
                    reduction='none'
                )
                loss = (loss * weight).sum()
                
                # 避免返回無限值或 NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print("警告：檢測到無效損失值！重置為零")
                    return torch.tensor(0.0, device=pred_score.device, dtype=pred_score.dtype)
                return loss
            except Exception as e:
                print(f"損失計算錯誤: {e}")
                return torch.tensor(0.0, device=pred_score.device, dtype=pred_score.dtype)
            
        # weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        # with torch.cuda.amp.autocast(enabled=False):
        # with torch.amp.autocast(device_type='cuda', enabled=False):
        #     loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()
        
        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl
    
    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                    target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                    target_scores.sum(-1), fg_mask).unsqueeze(-1)
            
            # 確保權重數值合法
            bbox_weight = torch.clamp(bbox_weight, 0.0, 1000.0)  # 避免極端大值

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
            
            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                        [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                        pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                        target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         target_ltrb_pos) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
        
        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.
        
        return loss_iou, loss_dfl
    
    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
                pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
                target_left.shape) * weight_left
        loss_right = F.cross_entropy(
                pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
                target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
