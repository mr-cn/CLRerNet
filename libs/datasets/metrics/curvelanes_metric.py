"""
用于Curvelanes数据集的评估指标，独立实现版
"""

import os
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Sequence
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
import cv2
from tqdm import tqdm

from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log


@METRICS.register_module()
class CurvelanesMetric(BaseMetric):
    def __init__(self, 
                 data_root, 
                 data_list, 
                 y_step=2,
                 iou_thresholds=[0.5], 
                 eval_width=1640,
                 eval_height=590,
                 lane_width=30,
                 interpolation_points=50):
        super().__init__()
        self.img_prefix = data_root
        self.list_path = data_list
        self.result_dir = tempfile.TemporaryDirectory().name
        self.ori_w, self.ori_h = 1640, 590  # Curvelanes原始尺寸
        self.y_step = y_step
        
        # 评估参数
        self.iou_thresholds = iou_thresholds
        self.eval_width = eval_width
        self.eval_height = eval_height
        self.lane_width = lane_width
        self.interpolation_points = interpolation_points

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for result in data_samples:
            self.results.append(result)

    def compute_metrics(self, results) -> dict:
        # 保存预测结果到临时目录
        self.save_predictions(results)
        
        # 加载所有预测和标注数据
        pred_data = self.load_prediction_data()
        gt_data = self.load_gt_data()
        
        # 计算指标
        metrics = {}
        for thr in self.iou_thresholds:
            tp, fp, fn = self.evaluate_all_images(pred_data, gt_data, thr)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.update({
                f'TP_{thr}': tp,
                f'FP_{thr}': fp,
                f'FN_{thr}': fn,
                f'Precision_{thr}': precision,
                f'Recall_{thr}': recall,
                f'F1_{thr}': f1
            })
        
        return metrics

    def save_predictions(self, results):
        """将预测结果保存为JSON文件"""
        os.makedirs(self.result_dir, exist_ok=True)
        for result in tqdm(results, desc='Saving predictions'):
            lanes = result["lanes"]
            img_name = result["metainfo"]["sub_img_name"]
            basename = Path(img_name).stem
            
            # 转换车道线格式
            lanes_json = {"Lines": []}
            for lane in lanes:
                ys = np.linspace(0, self.ori_h-1, num=20)
                xs = lane(ys/self.ori_h) * self.ori_w
                valid = (xs >= 0) & (xs < self.ori_w)
                points = [{"x": float(x), "y": float(y)} 
                         for x, y in zip(xs[valid], ys[valid]) if valid]
                if len(points) >= 2:
                    lanes_json["Lines"].append(points)
            
            # 写入文件
            with open(Path(self.result_dir)/f"{basename}.lines.json", 'w') as f:
                json.dump(lanes_json, f)

    def load_prediction_data(self):
        """加载所有预测结果"""
        with open(self.list_path) as f:
            img_names = [l.strip() for l in f]
        
        data = []
        for name in tqdm(img_names, desc='Loading predictions'):
            base = Path(name).stem
            path = Path(self.result_dir)/f"{base}.lines.json"
            data.append(self._load_lane_file(path))
        return data

    def load_gt_data(self):
        """加载所有标注数据"""
        with open(self.list_path) as f:
            img_names = [l.strip() for l in f]
        
        data = []
        for name in tqdm(img_names, desc='Loading GT'):
            base = Path(name).stem
            path = Path(self.img_prefix)/"labels"/f"{base}.lines.json"
            data.append(self._load_lane_file(path))
        return data

    def _load_lane_file(self, path):
        """加载单个车道文件"""
        try:
            with open(path) as f:
                data = json.load(f)
                return [
                    [{"x": float(p["x"]), "y": float(p["y"])} 
                    for p in lane if len(p) >= 2]  # 过滤无效车道
                    for lane in data.get("Lines", [])
                ]
        except:
            return []

    def evaluate_all_images(self, pred_data, gt_data, iou_thresh):
        """在所有图像上计算指标"""
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for pred_lanes, gt_lanes in tqdm(zip(pred_data, gt_data), 
                                       desc=f'Eval @{iou_thresh}'):
            # 插值处理
            interp_pred = [self.interpolate_lane(l) for l in pred_lanes]
            interp_gt = [self.interpolate_lane(l) for l in gt_lanes]
            
            # 计算IOU矩阵
            iou_matrix = np.zeros((len(interp_gt), len(interp_pred)))
            for i, gt in enumerate(interp_gt):
                for j, pred in enumerate(interp_pred):
                    iou_matrix[i,j] = self.calc_iou(gt, pred)
            
            # 匈牙利匹配
            if iou_matrix.size == 0:
                tp, fp, fn = 0, len(pred_lanes), len(gt_lanes)
            else:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                tp = (iou_matrix[row_ind, col_ind] >= iou_thresh).sum()
                fp = len(pred_lanes) - tp
                fn = len(gt_lanes) - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        return total_tp, total_fp, total_fn

    def interpolate_lane(self, lane_points):
        """三次样条插值车道线"""
        if len(lane_points) < 2:
            return []
        
        try:
            # 转换为numpy数组
            points = np.array([[p['x'], p['y']] for p in lane_points])
            points = points[np.argsort(points[:,1])]  # 按y坐标排序
            
            # 参数化插值
            tck, u = splprep(points.T, s=0, k=min(3, len(points)-1))
            u_new = np.linspace(0, 1, self.interpolation_points)
            x, y = splev(u_new, tck)
            
            return [{'x': float(xi), 'y': float(yi)} 
                   for xi, yi in zip(x, y)]
        except:
            return lane_points  # 插值失败时返回原始点

    def calc_iou(self, lane1, lane2):
        """计算两条车道线的IOU"""
        # 创建空图像
        h, w = self.eval_height, self.eval_width
        img1 = np.zeros((h, w), dtype=np.uint8)
        img2 = np.zeros((h, w), dtype=np.uint8)
        
        # 绘制车道线
        def draw_lane(img, lane):
            pts = []
            for p in lane:
                x = int(np.clip(p['x'], 0, w-1))
                y = int(np.clip(p['y'], 0, h-1))
                pts.append((x, y))
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts)], False, 255, self.lane_width)
        
        draw_lane(img1, lane1)
        draw_lane(img2, lane2)
        
        # 计算IOU
        intersection = np.logical_and(img1, img2)
        union = np.logical_or(img1, img2)
        iou = intersection.sum() / (union.sum() + 1e-9)
        return iou