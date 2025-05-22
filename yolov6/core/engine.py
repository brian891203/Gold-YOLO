# 2023.09.18-Changed for engine implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com
# !/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import os
import os.path as osp
import time
from ast import Pass
from copy import deepcopy

import cv2
import numpy as np
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import tools.eval as eval
from yolov6.data.data_load import create_dataloader
from yolov6.models.losses.loss import ComputeLoss as ComputeLoss
from yolov6.models.losses.loss_distill import \
    ComputeLoss as ComputeLoss_distill
from yolov6.models.losses.loss_distill_ns import \
    ComputeLoss as ComputeLoss_distill_ns
from yolov6.models.losses.loss_fuseab import ComputeLoss as ComputeLoss_ab
from yolov6.models.yolo import build_model
from yolov6.solver.build import build_lr_scheduler, build_optimizer
from yolov6.utils.checkpoint import (load_state_dict, save_checkpoint,
                                     strip_optimizer)
from yolov6.utils.ema import ModelEMA, de_parallel
from yolov6.utils.events import (LOGGER, NCOLS, load_yaml, write_tbimg,
                                 write_tblog)
from yolov6.utils.nms import xywh2xyxy
from yolov6.utils.RepOptimizer import RepVGGOptimizer, extract_scales


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device
        # 恢復訓練設置
        if args.resume:
            self.ckpt = torch.load(args.resume, map_location='cpu')

        # 分佈式訓練設置
        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir

        # get data loader
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict['nc']
        self.train_loader, self.val_loader = self.get_data_loader(args, cfg, self.data_dict)
        # get_data_loader 方法負責：建立數據加載器，處理批次大小、資料增強等

        # get model and optimizer
        self.distill_ns = True if self.args.distill and self.cfg.model.type in ['YOLOv6n', 'YOLOv6s', 'GoldYOLO-n', 'GoldYOLO-s'] else False
        model = self.get_model(args, cfg, self.num_classes, device) # 調用 self.get_model 創建主模型
        # get_model 方法負責：使用 build_model 創建模型，加載預訓練權重（若有）

        # 如果啟用蒸餾，調用 self.get_teacher_model 創建教師模型
        if self.args.distill:
            if self.args.fuse_ab:
                LOGGER.error('ERROR in: Distill models should turn off the fuse_ab.\n')
                exit()
            self.teacher_model = self.get_teacher_model(args, cfg, self.num_classes, device)
        # 量化設置
        if self.args.quant:
            self.quant_setup(model, cfg, device)
        
        # 優化器設置
        if cfg.training_mode == 'repopt':
            # 如果使用 repopt 訓練模式，創建 RepVGGOptimizer
            scales = self.load_scale_from_pretrained_models(cfg, device)
            reinit = False if cfg.model.pretrained is not None else True
            self.optimizer = RepVGGOptimizer(model, scales, args, cfg, reinit=reinit)
        else:
            # 否則調用 self.get_optimizer 創建標準優化器
            self.optimizer = self.get_optimizer(args, cfg, model)

        # 學習率調度器和模型平均
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
        self.ema = ModelEMA(model) if self.main_process else None

        # tensorboard TensorBoard 設置
        self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None
        self.start_epoch = 0

        # resume 檢查點恢復
        if hasattr(self, "ckpt"):
            resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(resume_state_dict, strict=True)  # load
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            if self.main_process:
                self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']
        # 模型並行化
        self.model = self.parallel_model(args, model, device)
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']
        
        # 訓練參數設置
        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.vis_imgs_list = []
        self.write_trainbatch_tb = args.write_trainbatch_tb
        # set color for classnames
        self.color = [tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)]
        
        # 損失設置
        self.loss_num = 3
        # 'dfl_loss'：分佈焦點損失 Distribution Focal Loss
        # DFL (Distribution Focal Loss) 將常規回歸問題轉換為一般分佈預測問題
        # 使用多個分類器輸出每個坐標的分佈，而不是直接預測坐標
        # 通過設置 cfg.model.head.use_dfl 和 cfg.model.head.reg_max 參數配置
        self.loss_info = ['Epoch', 'iou_loss', 'dfl_loss', 'cls_loss']

        if self.args.distill:  # 當開啟知識蒸餾功能（self.args.distill=True）時，會添加第四個損失
            self.loss_num += 1
            self.loss_info += ['cwd_loss']  #通道級蒸餾損失 (cwd_loss)
    
    # Training Process
    # Some notes
    # train() - 整個訓練過程（所有epochs）
    # train_before_loop() - 負責初始化整個訓練過程所需的各種組件和參數
    # train_in_loop() - 單個epoch的處理
    # train_in_steps() - 單個批次(batch)的處理
    # strip_model() - 這個方法負責清理和優化訓練完成後的模型文件，為部署做準備
    # train_after_loop() - 負責在訓練完全結束後執行必要的清理工作
    def train(self):
        try:
            self.train_before_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_in_loop(self.epoch)
            self.strip_model()
        
        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()
    
    # Training loop for each epoch
    # 1. 為這個 epoch 的訓練批次準備
    # 2. 迭代器遍歷該 epoch 的所有 batch 並開始 batch training
    # 3. 評估與保存階段 eval_and_save()
    def train_in_loop(self, epoch_num):
        try:
            self.prepare_for_steps() # 為這個epoch的訓練批次準備
            # self.pbar 是一個 tqdm 包裝的迭代器（在主進程中）或普通的 enumerate(self.train_loader) 迭代器
            # self.step 是當前批次索引
            # self.batch_data 包含當前批次的圖像和標籤
            for self.step, self.batch_data in self.pbar:  # 內層循環遍歷該epoch的所有batch
                try:
                    self.train_in_steps(epoch_num, self.step) # 處理單個批次 batch
                except Exception as e:
                    LOGGER.error(f'ERROR in training steps: {e}')
                self.print_details()   # 打印該批次的訓練細節
        except Exception as _:
            LOGGER.error('ERROR in training steps.')
            raise
        try:
            self.eval_and_save()  # 這個epoch結束後進行評估和模型保存
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise
    
    # Training loop for batchdata
    # 一個 Batch 的訓練步驟 
    # 1. 取出資料 → images, targets
    # 2. 歸一化與轉移至 GPU
    # 3. 前向傳播取得預測
    # 4. 計算損失（含蒸餾或分支融合）
    # 5. 多卡同步損失（可選）
    # 6. 反向傳播（計算梯度）
    # 7. 優化器更新參數（step + zero_grad）
    # 8. 可選視覺化至 TensorBoard
    # 9. 錯誤處理與跳過異常 batch
    def train_in_steps(self, epoch_num, step_num):
        try:
            # 預處理數據：歸一化圖像並移至目標設備
            # 從 0-255 歸一化到 0-1 使梯度計算更加穩定
            images, targets = self.prepro_data(self.batch_data, self.device)

            # plot train_batch and save to tensorboard once an epoch
            # 將訓練批次可視化到 TensorBoard（如果有配置）
            if self.write_trainbatch_tb and self.main_process and self.step == 0:
                self.plot_train_batch(images, targets)
                write_tbimg(self.tblogger, self.vis_train_batch, self.step + self.max_stepnum * self.epoch, type='train')
            
            # forward
            # with amp.autocast(enabled=self.device != 'cpu'):
            # 前向傳播（使用自動混合精度加速）
            with torch.amp.autocast(device_type='cuda', enabled=(self.device != 'cpu')):
                # 前向傳播，經過 Backbone → Neck → Head
                preds, s_featmaps = self.model(images)
                # preds（預測結果）可能有不同的格式，取決於模型的工作模式：
                # 1.標準模式
                # preds = (pred_cls, pred_box, pred_dfl)  # 元組形式
                #     pred_cls: 分類預測，形狀為 [batch_size, num_anchors, num_classes]
                #     pred_box: 邊界框預測，形狀為 [batch_size, num_anchors, 4]（xywh 格式）
                #     pred_dfl: 分佈焦點預測，用於更精確的邊界框定位
                # 2.分支融合模式（fuse_ab）
                # preds = (pred_ab_cls, pred_ab_box, pred_ab_dfl, pred_af_cls, pred_af_box, pred_af_dfl)
                #     前三個元素: ab 分支（輔助分支）的預測
                #     後三個元素: af 分支（主分支）的預測
                
                # s_featmaps（特徵圖）學生模型的多尺度特徵圖
                # s_featmaps = [feat_small, feat_medium, feat_large]  # 列表形式
                # 包含不同尺度的特徵圖，通常是 3 個
                # 主要用於知識蒸餾（當 self.args.distill=True 時）
                # 在知識蒸餾模式下，這些特徵圖與教師模型的特徵圖進行比較
                # 形狀通常為 [batch_size, channels, height, width]，其中高度和寬度與檢測的尺度相關

                # 計算損失 - 根據訓練模式選擇適當的損失函數
                if self.args.distill: # 知識蒸餾模式
                    with torch.no_grad():
                        t_preds, t_featmaps = self.teacher_model(images)
                    temperature = self.args.temperature
                    total_loss, loss_items = self.compute_loss_distill(preds, t_preds, s_featmaps, t_featmaps, targets, \
                                                                    epoch_num, self.max_epoch, temperature, step_num)
                
                elif self.args.fuse_ab: # 分支融合模式
                    total_loss, loss_items = self.compute_loss((preds[0], preds[3], preds[4]), targets, epoch_num,
                                                            step_num)  # YOLOv6_af
                    total_loss_ab, loss_items_ab = self.compute_loss_ab(preds[:3], targets, epoch_num,
                                                                        step_num)  # YOLOv6_ab
                    total_loss += total_loss_ab
                    loss_items += loss_items_ab
                else: # 標準模式
                    total_loss, loss_items = self.compute_loss(preds, targets, epoch_num, step_num)  # YOLOv6_af

                # 分佈式訓練中調整損失
                if self.rank != -1:
                    total_loss *= self.world_size

            # backward 反向傳播
            self.scaler.scale(total_loss).backward()
            self.loss_items = loss_items

            # 更新優化器
            self.update_optimizer()
        
        except RuntimeError as e:
            # 處理數值錯誤，允許跳過問題批次
            error_msg = str(e)
            if "device-side assert triggered" in error_msg or "value cannot be converted" in error_msg:
                print(f"警告: 在 epoch {epoch_num}, step {step_num} 發生數值錯誤。跳過此批次。")
                # 清空緩存並重置優化器
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                return
            else:
                raise  # 重新引發其他錯誤
    
    # 確定是否需要評估當前 epoch
    # 評估模型性能
    # 保存檢查點（最新、最佳、特定 epoch）
    # 記錄訓練指標到 TensorBoard
    # 可視化驗證結果
    def eval_and_save(self):
        # 計算剩餘輪次
        remaining_epochs = self.max_epoch - self.epoch
        # 確定評估間隔：在訓練末期縮短評估間隔
        eval_interval = self.args.eval_interval if remaining_epochs > self.args.heavy_eval_range else 3
        # 決定是否評估當前 epoch
        is_val_epoch = (not self.args.eval_final_only or (remaining_epochs == 1)) and (self.epoch % eval_interval == 0)
        if self.main_process:
            # 更新 EMA 模型屬性
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'])  # update attributes for ema model
            # 如果是評估輪次，執行模型評估
            if is_val_epoch:
                self.eval_model()
                self.ap = self.evaluate_results[1]  # mAP@0.5:0.95
                self.best_ap = max(self.ap, self.best_ap)
            # save ckpt
            # 創建檢查點字典
            ckpt = {
                    'model': deepcopy(de_parallel(self.model)).half(),
                    'ema': deepcopy(self.ema.ema).half(),
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
            }
            # 設置保存路徑
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            # 保存最新檢查點，如果是最佳性能則標記
            save_checkpoint(ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt')
            # 在訓練末期保存每個 epoch 的檢查點
            if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
                save_checkpoint(ckpt, False, save_ckpt_dir, model_name=f'{self.epoch}_ckpt')
            
            # default save best ap ckpt in stop strong aug epochs
            # 在停止強數據增強後的 epoch 中保存最佳模型
            if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
                if self.best_stop_strong_aug_ap < self.ap:
                    self.best_stop_strong_aug_ap = max(self.ap, self.best_stop_strong_aug_ap)
                    save_checkpoint(ckpt, False, save_ckpt_dir, model_name='best_stop_aug_ckpt')
            
            # 釋放內存
            del ckpt
            # log for learning rate
            # 記錄當前學習率
            lr = [x['lr'] for x in self.optimizer.param_groups]
            self.evaluate_results = list(self.evaluate_results) + lr
            
            # log for tensorboard
            # 記錄到 TensorBoard
            write_tblog(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss)
            # save validation predictions to tensorboard
            # 將驗證預測可視化保存到 TensorBoard
            write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')
    
    def eval_model(self):
        if not hasattr(self.cfg, "eval_params"):
            results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                       batch_size=self.batch_size // self.world_size * 2,
                                                       img_size=self.img_size,
                                                       model=self.ema.ema if self.args.calib is False else self.model,
                                                       conf_thres=0.03,
                                                       dataloader=self.val_loader,
                                                       save_dir=self.save_dir,
                                                       task='train')
        else:
            def get_cfg_value(cfg_dict, value_str, default_value):
                if value_str in cfg_dict:
                    if isinstance(cfg_dict[value_str], list):
                        return cfg_dict[value_str][0] if cfg_dict[value_str][0] is not None else default_value
                    else:
                        return cfg_dict[value_str] if cfg_dict[value_str] is not None else default_value
                else:
                    return default_value
            
            eval_img_size = get_cfg_value(self.cfg.eval_params, "img_size", self.img_size)
            results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                       batch_size=get_cfg_value(self.cfg.eval_params, "batch_size",
                                                                                self.batch_size // self.world_size * 2),
                                                       img_size=eval_img_size,
                                                       model=self.ema.ema if self.args.calib is False else self.model,
                                                       conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres",
                                                                                0.03),
                                                       dataloader=self.val_loader,
                                                       save_dir=self.save_dir,
                                                       task='train',
                                                       test_load_size=get_cfg_value(self.cfg.eval_params,
                                                                                    "test_load_size", eval_img_size),
                                                       letterbox_return_int=get_cfg_value(self.cfg.eval_params,
                                                                                          "letterbox_return_int",
                                                                                          False),
                                                       force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad",
                                                                                  False),
                                                       not_infer_on_rect=get_cfg_value(self.cfg.eval_params,
                                                                                       "not_infer_on_rect", False),
                                                       scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact",
                                                                                 False),
                                                       verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                                                       do_coco_metric=get_cfg_value(self.cfg.eval_params,
                                                                                    "do_coco_metric", True),
                                                       do_pr_metric=get_cfg_value(self.cfg.eval_params, "do_pr_metric",
                                                                                  False),
                                                       plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve",
                                                                                False),
                                                       plot_confusion_matrix=get_cfg_value(self.cfg.eval_params,
                                                                                           "plot_confusion_matrix",
                                                                                           False),
                                                       )
        
        LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
        self.evaluate_results = results[:2]
        # plot validation predictions
        self.plot_val_pred(vis_outputs, vis_paths)
    
    # 在每次訓練開始前執行，負責初始化整個訓練過程所需的各種組件和參數。
    # 這是整個訓練流程的第一步，確保所有必要的設置都已就緒
    def train_before_loop(self):
        # 記錄訓練開始時間和輸出開始日誌
        LOGGER.info('Training start...')
        self.start_time = time.time()

        # 計算學習率預熱的總步數
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum),
                                  1000) if self.args.quant is False else 0
        # 設置學習率調度器的起始點
        self.scheduler.last_epoch = self.start_epoch - 1

        # last_opt_step 用於追蹤最後執行的優化步驟（用於梯度累積）
        self.last_opt_step = -1
        # self.scaler = amp.GradScaler(enabled=self.device != 'cpu')
        # GradScaler 用於自動混合精度(AMP)訓練
        self.scaler = torch.amp.GradScaler(enabled=(self.device != 'cpu'))
        
        # 評估指標初始化
        # best_ap：追蹤整個訓練過程中的最佳 mAP 值
        # ap：當前輪次的 mAP 值
        # best_stop_strong_aug_ap：停止強數據增強後的最佳 mAP 值
        # evaluate_results：存儲評估結果的元組 (AP50, AP50_95)
        self.best_ap, self.ap = 0.0, 0.0
        self.best_stop_strong_aug_ap = 0.0
        self.evaluate_results = (0, 0)  # AP50, AP50_95
        
        # 標準損失計算器初始化
        # num_classes：分類數量
        # ori_img_size：原始圖像尺寸
        # warmup_epoch：ATSS（自適應訓練樣本選擇）算法的預熱輪次
        # use_dfl：是否使用分佈焦點損失(Distribution Focal Loss)
        # reg_max：DFL 的最大回歸值
        # iou_type：IoU 損失類型（如 giou, ciou 等）
        # fpn_strides：特徵金字塔網絡的步長
        self.compute_loss = ComputeLoss(num_classes=self.data_dict['nc'],
                                        ori_img_size=self.img_size,
                                        warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                        use_dfl=self.cfg.model.head.use_dfl,
                                        reg_max=self.cfg.model.head.reg_max,
                                        iou_type=self.cfg.model.head.iou_type,
                                        fpn_strides=self.cfg.model.head.strides)
        
        # 分支融合損失計算器初始化
        # 在啟用分支融合模式時初始化 ComputeLoss_ab
        # 參數類似標準損失計算器，但固定了部分參數：
        # warmup_epoch=0：不使用預熱
        # use_dfl=False：不使用分佈焦點損失
        # reg_max=0：不使用回歸最大值
        # 分支融合模式同時訓練輔助分支(ab)和主分支(af)
        if self.args.fuse_ab:
            self.compute_loss_ab = ComputeLoss_ab(num_classes=self.data_dict['nc'],
                                                  ori_img_size=self.img_size,
                                                  warmup_epoch=0,
                                                  use_dfl=False,
                                                  reg_max=0,
                                                  iou_type=self.cfg.model.head.iou_type,
                                                  fpn_strides=self.cfg.model.head.strides)
        # 知識蒸餾損失計算器初始化
        # 根據模型類型選擇不同的蒸餾損失函數：
        # 小型模型使用 ComputeLoss_distill_ns（針對 nano/small 模型優化）
        # 其他模型使用 ComputeLoss_distill
        # 額外參數：
        # distill_weight：知識蒸餾各部分損失的權重
        # distill_feat：是否啟用特徵圖蒸餾（通過 distill_feat 參數控制）
        if self.args.distill:
            if self.cfg.model.type in ['YOLOv6n', 'YOLOv6s', 'GoldYOLO-n', 'GoldYOLO-s']:
                Loss_distill_func = ComputeLoss_distill_ns
            else:
                Loss_distill_func = ComputeLoss_distill
            
            self.compute_loss_distill = Loss_distill_func(num_classes=self.data_dict['nc'],
                                                          ori_img_size=self.img_size,
                                                          fpn_strides=self.cfg.model.head.strides,
                                                          warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                                          use_dfl=self.cfg.model.head.use_dfl,
                                                          reg_max=self.cfg.model.head.reg_max,
                                                          iou_type=self.cfg.model.head.iou_type,
                                                          distill_weight=self.cfg.model.head.distill_weight,
                                                          distill_feat=self.args.distill_feat,
                                                          )
    
    def prepare_for_steps(self):
        # 如果不是第一個 epoch，更新學習率調度器
        if self.epoch > self.start_epoch:
            self.scheduler.step()
        # 在最後 n 個 epoch 停止使用強數據增強（馬賽克和混合)
        # stop strong aug like mosaic and mixup from last n epoch by recreate dataloader
        if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
            self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)
        
        # 設置模型為訓練模式    
        self.model.train()

        # 分佈式訓練設置
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        # 初始化損失平均值追蹤器
        self.mean_loss = torch.zeros(self.loss_num, device=self.device)
        # 清空優化器梯度
        self.optimizer.zero_grad()
        # 打印損失表頭
        # self.loss_num = 3 --> 'iou_loss', 'dfl_loss', 'cls_loss'
        LOGGER.info(('\n' + '%10s' * (self.loss_num + 1)) % (*self.loss_info,))
        # 準備數據迭代器
        self.pbar = enumerate(self.train_loader)
        # 在主進程中設置進度條
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    # Print loss after each steps
    # 每個批次後更新並顯示訓練統計信息
    def print_details(self):
        if self.main_process:
            # 更新平均損失值
            self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)
            # 更新進度條描述，顯示當前 epoch 和各項損失
            # Epoch: 10/299   IoU_loss: 1.234   DFL_loss: 0.567   CLS_loss: 0.789 |████████████████████| 100% [320/320] ETA: 00:00
            # 各項損失值（IoU損失、DFL損失、分類損失等）
            self.pbar.set_description(('%10s' + '%10.4g' * self.loss_num) % (f'{self.epoch}/{self.max_epoch - 1}', \
                                                                             *(self.mean_loss)))
    
    # 訓練過程的最後一步，在所有訓練輪次完成後執行。
    # 這個方法負責清理和優化訓練完成後的模型文件，為部署做準備
    def strip_model(self):
        if self.main_process:
            # 計算並記錄總訓練時間
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')
            # 確定模型檢查點保存位置
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            # 調用 strip_optimizer 函數精簡模型檢查點
            strip_optimizer(save_ckpt_dir, self.epoch)  # strip optimizers for saved pt model
    
    # Empty cache if training finished
    # 負責在訓練完全結束後執行必要的清理工作。
    def train_after_loop(self):
        if self.device != 'cpu':
            # 釋放 GPU 內存：通過 torch.cuda.empty_cache() 釋放所有未使用的 CUDA 緩存
            torch.cuda.empty_cache()
    
    # 根據訓練進度調整學習率和動量（熱身階段）
    # 在累積足夠的梯度後更新模型參數
    # 更新 EMA 模型
    def update_optimizer(self):
        # 計算全局步數
        curr_step = self.step + self.max_stepnum * self.epoch
        # 設置梯度累積數量
        self.accumulate = max(1, round(64 / self.batch_size))

        # 在熱身階段動態調整學習率和動量
        if curr_step <= self.warmup_stepnum:
            # 在熱身階段根據當前步數調整梯度累積數量
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            # 為每個參數組更新學習率和動量
            for k, param in enumerate(self.optimizer.param_groups):
                # 偏置學習率特殊處理
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                # 線性插值計算當前學習率
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                        [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                # 如果有動量參數，也進行插值
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                                  [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])
        # 判斷是否需要執行優化器步驟
        if curr_step - self.last_opt_step >= self.accumulate:
            # 執行縮放的優化器步驟
            self.scaler.step(self.optimizer)
            # 更新梯度縮放器
            self.scaler.update()
            # 清空梯度
            self.optimizer.zero_grad()
            # 更新 EMA 模型
            if self.ema:
                self.ema.update(self.model)
            # 記錄最後優化步驟
            self.last_opt_step = curr_step
    
    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        train_path, val_path = data_dict['train'], data_dict['val']
        # check data
        nc = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # create train dataloader
        train_loader = create_dataloader(train_path, args.img_size, args.batch_size // args.world_size, grid_size,
                                         hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                         workers=args.workers, shuffle=True, check_images=args.check_images,
                                         check_labels=args.check_labels, data_dict=data_dict, task='train')[0]
        # create val dataloader
        val_loader = None
        if args.rank in [-1, 0]:
            val_loader = create_dataloader(val_path, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                                           hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                                           workers=args.workers, check_images=args.check_images,
                                           check_labels=args.check_labels, data_dict=data_dict, task='val')[0]
        
        return train_loader, val_loader
    
    # staticmethod: 函數與實例或類別都無關，只是一個工具函數，純粹是一個「放在類別裡的普通函式」
    # 在 Gold-YOLO 的訓練迴圈中，prepro_data 方法在每個批次處理開始時被調用
    # 接收批次數據 batch_data（來自數據加載器）和目標設備 device（通常是 GPU）
    # 提取並處理圖像數據：batch_data[0]
    # 提取並處理標籤數據：batch_data[1]
    # 返回處理後的圖像和標籤，供模型訓練使用
    @staticmethod
    def prepro_data(batch_data, device):
        # 將圖像數據歸一化到 0-1 範圍並移至指定設備
        # 梯度穩定性：神經網絡對輸入值的大小敏感。0-255 範圍的輸入可能導致梯度爆炸或消失，歸一化到 0-1 範圍使梯度計算更加穩定
        images = batch_data[0].to(device, non_blocking=True).float() / 255
        # 將標籤數據移至指定設備
        targets = batch_data[1].to(device)
        return images, targets
    
    def get_model(self, args, cfg, nc, device):
        model = build_model(cfg, nc, device, fuse_ab=self.args.fuse_ab, distill_ns=self.distill_ns)
        weights = cfg.model.pretrained
        if weights:  # finetune if pretrained model is set
            LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
            model = load_state_dict(weights, model, map_location=device)
        
        if args.use_syncbn and self.rank != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        LOGGER.info('Model: {}'.format(model))
        return model
    
    def get_teacher_model(self, args, cfg, nc, device):
        teacher_fuse_ab = False if cfg.model.head.num_layers != 3 else True
        model = build_model(cfg, nc, device, fuse_ab=teacher_fuse_ab)
        weights = args.teacher_model_path
        if weights:  # finetune if pretrained model is set
            LOGGER.info(f'Loading state_dict from {weights} for teacher')
            model = load_state_dict(weights, model, map_location=device)
        LOGGER.info('Model: {}'.format(model))
        # Do not update running means and running vars
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
        return model
    
    @staticmethod
    def load_scale_from_pretrained_models(cfg, device):
        weights = cfg.model.scales
        scales = None
        if not weights:
            LOGGER.error("ERROR: No scales provided to init RepOptimizer!")
        else:
            ckpt = torch.load(weights, map_location=device)
            scales = extract_scales(ckpt)
        return scales
    
    # 負責配置模型的並行化策略，對於高效地利用多 GPU 資源進行訓練至關重要
    # 這個方法支持兩種 PyTorch 並行化策略：1. DP (DataParallel) 2. DDP (DistributedDataParallel) 
    @staticmethod
    def parallel_model(args, model, device):
        # If DP mode
        # 觸發條件:
        # 使用 GPU 訓練 (device.type != 'cpu')
        # 未設置分佈式環境 (args.rank == -1)
        # 存在多個 GPU (torch.cuda.device_count() > 1)
        dp_mode = device.type != 'cpu' and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)
        
        # If DDP mode
        # 觸發條件:
        # 使用 GPU 訓練 (device.type != 'cpu')
        # 已設置分佈式環境 (args.rank != -1)
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        
        return model
    
    def get_optimizer(self, args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        cfg.solver.lr0 *= args.batch_size / (self.world_size * args.bs_per_gpu)  # rescale lr0 related to batchsize
        optimizer = build_optimizer(cfg, model)
        return optimizer
    
    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf
    
    def plot_train_batch(self, images, targets, max_size=1920, max_subplots=16):
        # Plot train_batch with labels
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)
        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)
        paths = self.batch_data[2]  # image paths
        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y:y + h, x:x + w, :] = im
        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)  # borders
            cv2.putText(mosaic, f"{os.path.basename(paths[i])[:40]}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(220, 220, 220), thickness=1)  # filename
            if len(targets) > 0:
                ti = targets[targets[:, 0] == i]  # image targets
                boxes = xywh2xyxy(ti[:, 2:6]).T
                classes = ti[:, 1].astype('int')
                labels = ti.shape[1] == 6  # labels if no conf column
                if boxes.shape[1]:
                    if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                        boxes[[0, 2]] *= w  # scale to pixels
                        boxes[[1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes *= scale
                boxes[[0, 2]] += x
                boxes[[1, 3]] += y
                for j, box in enumerate(boxes.T.tolist()):
                    box = [int(k) for k in box]
                    cls = classes[j]
                    color = tuple([int(x) for x in self.color[cls]])
                    cls = self.data_dict['names'][cls] if self.data_dict['names'] else cls
                    if labels:
                        label = f'{cls}'
                        cv2.rectangle(mosaic, (box[0], box[1]), (box[2], box[3]), color, thickness=1)
                        cv2.putText(mosaic, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color,
                                    thickness=1)
        self.vis_train_batch = mosaic.copy()
    
    def plot_val_pred(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=5):
        # plot validation predictions
        self.vis_imgs_list = []
        for (vis_output, vis_path) in zip(vis_outputs, vis_paths):
            vis_output_array = vis_output.cpu().numpy()  # xyxy
            ori_img = cv2.imread(vis_path)
            for bbox_idx, vis_bbox in enumerate(vis_output_array):
                x_tl = int(vis_bbox[0])
                y_tl = int(vis_bbox[1])
                x_br = int(vis_bbox[2])
                y_br = int(vis_bbox[3])
                box_score = vis_bbox[4]
                cls_id = int(vis_bbox[5])
                # draw top n bbox
                if box_score < vis_conf or bbox_idx > vis_max_box_num:
                    break
                cv2.rectangle(ori_img, (x_tl, y_tl), (x_br, y_br), tuple([int(x) for x in self.color[cls_id]]),
                              thickness=1)
                cv2.putText(ori_img, f"{self.data_dict['names'][cls_id]}: {box_score:.2f}", (x_tl, y_tl - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple([int(x) for x in self.color[cls_id]]), thickness=1)
            self.vis_imgs_list.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))
    
    # PTQ
    def calibrate(self, cfg):
        def save_calib_model(model, cfg):
            # Save calibrated checkpoint
            output_model_path = os.path.join(cfg.ptq.calib_output_path, '{}_calib_{}.pt'.
                                             format(os.path.splitext(os.path.basename(cfg.model.pretrained))[0],
                                                    cfg.ptq.calib_method))
            if cfg.ptq.sensitive_layers_skip is True:
                output_model_path = output_model_path.replace('.pt', '_partial.pt')
            LOGGER.info('Saving calibrated model to {}... '.format(output_model_path))
            if not os.path.exists(cfg.ptq.calib_output_path):
                os.mkdir(cfg.ptq.calib_output_path)
            torch.save({'model': deepcopy(de_parallel(model)).half()}, output_model_path)
        
        assert self.args.quant is True and self.args.calib is True
        if self.main_process:
            from tools.qat.qat_utils import ptq_calibrate
            ptq_calibrate(self.model, self.train_loader, cfg)
            self.epoch = 0
            self.eval_model()
            save_calib_model(self.model, cfg)
    
    # QAT
    def quant_setup(self, model, cfg, device):
        if self.args.quant:
            from tools.qat.qat_utils import (qat_init_model_manu,
                                             skip_sensitive_layers)
            qat_init_model_manu(model, cfg, self.args)
            # workaround
            model.neck.upsample_enable_quant(cfg.ptq.num_bits, cfg.ptq.calib_method)
            # if self.main_process:
            #     print(model)
            # QAT
            if self.args.calib is False:
                if cfg.qat.sensitive_layers_skip:
                    skip_sensitive_layers(model, cfg.qat.sensitive_layers_list)
                # QAT flow load calibrated model
                assert cfg.qat.calib_pt is not None, 'Please provide calibrated model'
                model.load_state_dict(torch.load(cfg.qat.calib_pt)['model'].float().state_dict())
            model.to(device)
