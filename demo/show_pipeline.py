#!/usr/bin/env python
# 基于MMDetection 3.x与mmengine的CLRerNet的pipeline可视化工具
import argparse
import os
import cv2
import numpy as np
import torch
from mmengine.config import Config
from libs.datasets.pipelines import Compose

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='输入图像文件路径')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('out_dir', help='输出目录路径')
    parser.add_argument('--denormalize', action='store_true', 
                      help='是否对归一化的图像进行反归一化处理')
    parser.add_argument('--device', default='cuda:0', help='推理使用的设备')
    args = parser.parse_args()
    return args

def save_image(img, filename, denormalize=False):
    """保存图像到文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 如果是tensor，转换为numpy
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    # 确保图像格式正确
    if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW格式
        img = np.transpose(img, (1, 2, 0))
    
    # 如果是单通道图像，转为三通道
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 根据需要进行反归一化处理
    if denormalize:
        # 假设归一化使用的是[0, 0, 0]均值和[255, 255, 255]方差
        # 这里根据configs中看到的归一化配置设置
        img = img * np.array([255.0, 255.0, 255.0]) + np.array([0.0, 0.0, 0.0])
    
    # 确保像素值在合理范围内
    if img.max() <= 1.0 and img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 保存图像
    cv2.imwrite(filename, img)
    print(f"已保存图像到: {filename}")

def visualize_pipeline_result(img_path, config_path, out_dir, denormalize=False):
    """可视化pipeline处理后的图像"""
    # 加载配置文件
    cfg = Config.fromfile(config_path)
    
    # 读取原始图像
    ori_img = cv2.imread(img_path)
    if ori_img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    
    # 保存原始图像
    ori_out_path = os.path.join(out_dir, "original.png")
    save_image(ori_img, ori_out_path)
    
    # 准备数据
    ori_shape = ori_img.shape
    data = dict(
        filename=img_path,
        sub_img_name=None,
        img=ori_img,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=ori_shape,
        ori_shape=ori_shape,
    )
    
    # 获取test pipeline
    test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    
    # 执行pipeline处理
    result = test_pipeline(data)
    
    # 保存pipeline处理后的图像
    if result is not None and 'inputs' in result:
        # 处理后的图像可能在inputs中(tensor格式)
        processed_img = result['inputs']
        processed_out_path = os.path.join(out_dir, "after_pipeline.png")
        save_image(processed_img, processed_out_path, denormalize)
    
    # 保存data_samples中的信息，如果有的话
    if result is not None and 'data_samples' in result:
        # 这里根据需要提取data_samples中的信息并可视化
        data_sample = result['data_samples']
        if hasattr(data_sample, 'metainfo'):
            # 打印一些元信息，帮助理解处理过程
            print("处理后的元信息:")
            for key, value in data_sample.metainfo.items():
                print(f"  {key}: {value}")
    
    return result

def main():
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化pipeline处理结果
    result = visualize_pipeline_result(
        args.img, 
        args.config, 
        args.out_dir,
        args.denormalize
    )
    
    print(f"Pipeline处理完成，结果已保存到: {args.out_dir}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    args = parse_args()
    main()
