# 2023.09.18-Changed for checkpoint load implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import os.path as osp
import shutil

import torch

from yolov6.utils.events import LOGGER
from yolov6.utils.torch_utils import fuse_model


def load_state_dict(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt['model'].float().state_dict()
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def load_checkpoint_2(model, weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)['model']
    # model = ckpt
    model.load_state_dict(ckpt)
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """ Save checkpoint to the disk."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + '.pt')
    torch.save(ckpt, filename)
    if is_best:
        best_filename = osp.join(save_dir, 'best_ckpt.pt')
        shutil.copyfile(filename, best_filename)


def strip_optimizer(ckpt_dir, epoch):
    """Delete optimizer from saved checkpoint file"""
    for s in ['best', 'last']:
        ckpt_path = osp.join(ckpt_dir, '{}_ckpt.pt'.format(s))
        if not osp.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if ckpt.get('ema'):
            ckpt['model'] = ckpt['ema']  # replace model with ema
        for k in ['optimizer', 'ema', 'updates']:  # keys
            ckpt[k] = None
        ckpt['epoch'] = epoch
        ckpt['model'].half()  # to FP16
        for p in ckpt['model'].parameters():
            p.requires_grad = False
        torch.save(ckpt, ckpt_path)

        # 遍歷模型檢查點：對 'last' 和 'best' 兩種檢查點進行處理
        # 檢查路徑格式：{dir_path}/{sm}_ckpt.pt，其中 sm 是 'last' 或 'best'
        # 如果檢查點不存在則跳過
        # 加載檢查點：使用 torch.load 加載檢查點文件到 CPU 內存

        # map_location=torch.device('cpu') 確保模型能在任何設備上加載
        # 選擇模型權重：優先使用 EMA 模型（如果存在）

        # EMA (Exponential Moving Average) 模型通常提供更好的泛化性能
        # 使用 .float() 確保模型參數為單精度浮點格式
        # 保存精簡模型：只保存輪次信息和模型權重

        # 移除優化器狀態、學習率調度器和其他訓練相關參數
        # 使用與原檢查點相同的文件名覆蓋保存
        # 記錄日誌：輸出成功精簡的信息
