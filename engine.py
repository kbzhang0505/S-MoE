# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2
from torchsummary import summary

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name),
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name),
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)

# the training routine
# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 0):
#     model.train()
#     criterion.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#
#     for samples, targets in data_loader:
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         # Forward pass
#         outputs = model(samples)
#
#         # Calculate all losses
#         loss_dict = criterion(outputs, targets)
#         total_loss = loss_dict['total_loss']
#
#         # Loss reduction and logging preparation
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#
#         # 分离缩放前后的损失用于日志记录
#         loss_dict_reduced_unscaled = {
#             f'{k}_unscaled': v.detach()
#             for k, v in loss_dict_reduced.items()
#             if k != 'total_loss'
#         }
#
#         # 获取缩放后的损失值（仅用于日志显示）
#         weight_dict = criterion.weight_dict
#         loss_dict_reduced_scaled = {
#             k: v.detach() * weight_dict.get(k, 1.0)
#             for k, v in loss_dict_reduced.items()
#             if k in weight_dict
#         }
#
#         # 检查损失有效性
#         loss_value = total_loss.item()
#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping training")
#             print(loss_dict_reduced)
#             sys.exit(1)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         total_loss.backward()
#         if max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#         optimizer.step()
#
#         # 更新日志记录器
#         metric_logger.update(
#             loss=total_loss.item(),
#             **loss_dict_reduced_scaled,
#             **loss_dict_reduced_unscaled
#         )
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#
#     # 跨进程同步统计信息
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {
#         k: meter.global_avg
#         for k, meter in metric_logger.meters.items()
#     }


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        summary(model, (3, 128, 128))
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    for samples, targets in data_loader:
        samples = samples.to(device)

        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # if specified, save the visualized images
        if vis_dir is not None:
            vis(samples, targets, [points], vis_dir)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse

# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """
# 训练和评估函数，主要用于main.py
# 大部分代码来自DETR (https://github.com/facebookresearch/detr)。
# """
#
# import math  # 导入数学库
# import os  # 导入操作系统库
# import sys  # 导入系统库
# from typing import Iterable  # 导入Iterable类型
#
# import torch  # 导入PyTorch库
# from filterpy.kalman import predict
# from torch import nn
# import util.misc as utils  # 导入自定义的工具库
# from util.misc import NestedTensor  # 导入NestedTensor类型
# import numpy as np  # 导入NumPy库
# import time  # 导入时间库
# import torchvision.transforms as standard_transforms  # 导入标准的图像变换模块
# import cv2  # 导入OpenCV库
#
#
# class DeNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean  # 存储均值
#         self.std = std  # 存储标准差
#
#     def __call__(self, tensor):
#         # 对输入的张量进行反归一化操作
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)  # 对张量t进行反归一化
#         return tensor  # 返回处理后的张量
#
#
# def vis(samples, targets, pred, vis_dir, des=None):
#     '''
#     samples -> tensor: [batch, 3, H, W]  # 输入样本张量，包含多个图像
#     targets -> list of dict: [{'points':[], 'image_id': str}]  # 目标数据，包含每个图像的地面真实点
#     pred -> list: [num_preds, 2]  # 模型预测结果
#     '''
#     gts = [t['point'].tolist() for t in targets]  # 提取目标中的地面真实点列表
#
#     pil_to_tensor = standard_transforms.ToTensor()  # 定义从PIL图像到张量的转换
#
#     # 定义图像反归一化和转换为PIL格式的组合操作
#     restore_transform = standard_transforms.Compose([
#         DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 反归一化操作
#         standard_transforms.ToPILImage()  # 转换为PIL图像
#     ])
#
#     # 逐个绘制图像
#     for idx in range(samples.shape[0]):
#         sample = restore_transform(samples[idx])  # 反归一化和转换当前样本
#         sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255  # 转换为RGB并归一化到255
#         sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()  # 复制并转换为BGR格式的真实图像
#         sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()  # 复制并转换为BGR格式的预测图像
#
#         max_len = np.max(sample_gt.shape)  # 获取图像的最大边长
#
#         size = 2  # 绘制点的大小
#         # 绘制真实点
#         for t in gts[idx]:
#             sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)  # 用绿色绘制真实点
#         # 绘制预测点
#         for p in pred[idx]:
#             sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)  # 用红色绘制预测点
#
#         name = targets[idx]['image_id']  # 获取图像ID
#         # 保存可视化的图像
#         if des is not None:
#             cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name),
#                                                                                   des, len(gts[idx]), len(pred[idx]))), sample_gt)  # 保存真实点图
#             cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name),
#                                                                                     des, len(gts[idx]), len(pred[idx]))), sample_pred)  # 保存预测点图
#         else:
#             cv2.imwrite(
#                 os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
#                 sample_gt)  # 保存真实点图
#             cv2.imwrite(
#                 os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
#                 sample_pred)  # 保存预测点图
#
#
# # 训练过程
# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 0):
#     model.train()  # 设置模型为训练模式
#     criterion.train()  # 设置损失函数为训练模式
#     metric_logger = utils.MetricLogger(delimiter="  ")  # 初始化日志记录器
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率
#     # 迭代所有训练样本
#     for samples, targets in data_loader:
#         samples = samples.to(device)  # 将样本移到指定设备
#         # 将 targets 中的所有项移动到设备上，并同时提取 density
#         for t in targets:
#             t['point'] = t['point'].to(device)
#             t['image_id'] = t['image_id'].to(device)
#             t['labels'] = t['labels'].to(device)
#             density = t.get('density')  # 直接从 target 中取出 density
#             if density is not None:
#                 density = density.to(device)  # 将 density 移动到设备上
#
#         # 前向传播
#         outputs = model(samples)  # 获取模型输出
#         # 计算损失
#         loss_dict = criterion(outputs, targets)  # 根据输出和目标计算损失字典
#         # Calculate density loss
#         if 'pred_density' in outputs:  # Check if the output has predicted density
#             density_loss = nn.MSELoss(reduction='mean')(outputs['pred_density'], density)
#             loss_dict['density_loss'] = density_loss
#
#         weight_dict = criterion.weight_dict  # 获取损失权重字典
#         # 计算加权损失
#         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
#
#         # 减少所有损失
#         loss_dict_reduced = utils.reduce_dict(loss_dict)  # 对损失进行归约
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v  # 不加权损失
#                                       for k, v in loss_dict_reduced.items()}
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]  # 加权损失
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())  # 加权损失总和
#
#         loss_value = losses_reduced_scaled.item()  # 获取损失值
#
#         if not math.isfinite(loss_value):  # 检查损失值是否有限
#             print("Loss is {}, stopping training".format(loss_value))  # 打印损失
#             print(loss_dict_reduced)  # 打印损失字典
#             sys.exit(1)  # 退出程序
#
#         # 反向传播
#         optimizer.zero_grad()  # 清空梯度
#         losses.backward()  # 反向传播计算梯度
#         if max_norm > 0:  # 如果设置了最大梯度范数
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 进行梯度裁剪
#         optimizer.step()  # 更新参数
#         # 更新日志
#         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)  # 更新日志记录器
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # 更新学习率
#
#     # 收集所有进程的统计信息
#     metric_logger.synchronize_between_processes()  # 同步进程
#     print("Averaged stats:", metric_logger)  # 打印平均统计信息
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  # 返回日志信息
#
#
# # 推理过程
# @torch.no_grad()
# def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
#     model.eval()  # 设置模型为评估模式
#
#     metric_logger = utils.MetricLogger(delimiter="  ")  # 初始化日志记录器
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))  # 记录分类错误率
#     # 在所有图像上进行推理以计算MAE
#     maes = []  # 存储绝对误差
#     mses = []  # 存储均方误差
#     density_losses = []
#     for samples, targets, density in data_loader:
#         samples = samples.to(device)  # 将样本移到指定设备
#
#         outputs, pre_den = model(samples)  # 获取模型输出
#         predict = pre_den[0]
#         outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]  # 获取预测概率
#         outputs_points = outputs['pred_points'][0]  # 获取预测的点
#
#         # 计算密度图损失
#         if 'pred_density' in outputs:
#             density_loss = nn.MSELoss(reduction='mean')(outputs['pred_density'], density.to(device))
#             density_losses.append(density_loss.item())
#
#         gt_cnt = targets[0]['point'].shape[0]  # 获取真实点的数量
#         threshold = 0.5  # 设定阈值
#
#         points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()  # 获取阈值以上的预测点
#         predict_cnt = int((outputs_scores > threshold).sum())  # 计算预测点的数量
#         # 如果指定，保存可视化图像
#         if vis_dir is not None:
#             vis(samples, targets, [points], vis_dir)  # 可视化并保存图像
#         # 计算MAE和MSE
#         mae = abs(predict_cnt - gt_cnt)  # 计算绝对误差
#         # mae += torch.abs(predict.sum() - density.sum()).item()
#         mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)  # 计算均方误差
#         # mse += ((predict.sum() - density.sum()) ** 2).item()
#         maes.append(float(mae))  # 存储MAE
#         mses.append(float(mse))  # 存储MSE
#     # 计算MAE和MSE的平均值
#     mae = np.mean(maes)  # 计算MAE的平均值
#     mse = np.sqrt(np.mean(mses))  # 计算MSE的平方根作为RMSE
#
#     return mae, mse  # 返回MAE和RMSE
