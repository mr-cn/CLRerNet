"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/curvelanes_dataset.py
"""
from pathlib import Path
import json
import os

import cv2
import numpy as np
from mmdet.registry import DATASETS
from tqdm import tqdm
from libs.datasets.metrics.culane_metric import CULaneMetric

from .culane_dataset import CulaneDataset


@DATASETS.register_module()
class CurvelanesDataset(CulaneDataset):
    def __init__(
        self,
        data_root,
        data_list,
        pipeline,
        diff_file=None,
        diff_thr=15,
        test_mode=True,
        y_step=2,
        img_prefix=None,
        **kwargs
    ):
        """
        Args:
            data_root (str): Dataset root path.
            data_list (str): Dataset list file path.
            pipeline (List[mmcv.utils.config.ConfigDict]):
                Data transformation pipeline configs.
            test_mode (bool): Test flag.
            y_step (int): Row interval (in the original image's y scale) to sample
                the predicted lanes for evaluation.
            img_prefix (str, optional): The prefix of image path. If provided, it will override data_root.
        """
        # 首先调用父类的初始化方法
        super(CurvelanesDataset, self).__init__(
            data_root=data_root,
            data_list=data_list,
            pipeline=pipeline,
            diff_file=diff_file,
            diff_thr=diff_thr,
            test_mode=test_mode,
            y_step=y_step,
            **kwargs
        )
        
        # 如果提供了img_prefix，则覆盖原来的img_prefix
        if img_prefix is not None:
            self.img_prefix = img_prefix
            print(f"Using custom img_prefix: {self.img_prefix}")

    def prepare_train_img(self, idx):
        """
        Read and process the image through the transform pipeline for training.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        img_info = self.img_infos[idx]
        imgname = str(Path(self.img_prefix) / img_info)
        sub_img_name = img_info
        img_tmp = cv2.imread(imgname)
        ori_shape = img_tmp.shape
        if ori_shape == (1440, 2560, 3):
            img = np.zeros((800, 2560, 3), np.uint8)
            img[:800, :, :] = img_tmp[640:, ...]
            crop_shape = (800, 2560, 3)
            offset_y = -640
        elif ori_shape == (660, 1570, 3):
            img = np.zeros((480, 1570, 3), np.uint8)
            img[:480, :, :] = img_tmp[180:, ...]
            crop_shape = (480, 1570, 3)
            offset_y = -180
        elif ori_shape == (720, 1280, 3):
            img = np.zeros((352, 1280, 3), np.uint8)
            img[:352, :, :] = img_tmp[368:, ...]
            crop_shape = (352, 1280, 3)
            offset_y = -368
        else:
            return None
        img_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(idx, offset_y)
        eval_shape = (
            crop_shape[0] / ori_shape[0] * 224,
            224,
        )  # Used for LaneIoU calculation.
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=kps,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=img_shape,
            ori_shape=ori_shape,
            eval_shape=eval_shape,
            crop_shape=crop_shape,
        )
        if self.mask_paths[0]:
            mask = self.load_mask(idx)
            mask = mask[-offset_y:, :, 0]
            mask = np.clip(mask, 0, 1)
            assert mask.shape[:2] == crop_shape[:2]
            results["gt_masks"] = mask

        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """
        Read and process the image through the transform pipeline for test.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        # 处理图像路径，确保正确加载图像
        img_info = self.img_infos[idx]
        imgname = str(Path(self.img_prefix) / img_info)
        sub_img_name = img_info
        
        # 尝试加载图像
        img_tmp = cv2.imread(imgname)
        
        # 如果加载失败，打印错误信息并返回None
        if img_tmp is None:
            print(f"错误: 无法加载图像 {imgname}")
            return None
            
        ori_shape = img_tmp.shape

        if ori_shape == (1440, 2560, 3):
            img = np.zeros((800, 2560, 3), np.uint8)
            img[:800, :, :] = img_tmp[640:, ...]
            crop_shape = (800, 2560, 3)
            crop_offset = [0, 640]
        elif ori_shape == (660, 1570, 3):
            img = np.zeros((480, 1570, 3), np.uint8)
            crop_shape = (480, 1570, 3)
            img[:480, :, :] = img_tmp[180:, ...]
            crop_offset = [0, 180]
        elif ori_shape == (720, 1280, 3):
            img = np.zeros((352, 1280, 3), np.uint8)
            img[:352, :, :] = img_tmp[368:, ...]
            crop_shape = (352, 1280, 3)
            crop_offset = [0, 368]
        else:
            # 打印未知形状信息便于调试
            print(f"警告: 未知图像形状 {ori_shape} for {imgname}")
            return None

        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=crop_shape,
            ori_shape=ori_shape,
            crop_offset=crop_offset,
            crop_shape=crop_shape,
        )
        return self.pipeline(results)

    @staticmethod
    def convert_coords_formal(lanes):
        res = []
        for lane in lanes:
            lane_coords = []
            for coord in lane:
                lane_coords.append({"x": coord[0], "y": coord[1]})
            res.append(lane_coords)
        return res

    def parse_anno(self, filename, formal=True):
        anno_dir = filename.replace(".jpg", ".lines.json")
        annos = []
        try:
            with open(anno_dir, "r") as anno_f:
                json_data = json.load(anno_f)
                lanes = json_data.get("Lines", [])
                
                for lane in lanes:
                    coords = []
                    for point in lane:
                        x = float(point["x"])
                        y = float(point["y"])
                        coords.append((x, y))
                    
                    if len(coords) >= 2:  # 至少需要两个点才是有效车道
                        annos.append(coords)
            
            if formal:
                annos = self.convert_coords_formal(annos)
            return annos
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"读取标注文件错误 {anno_dir}: {e}")
            return []

    def evaluate(
        self,
        results,
        metric="F1",
        logger=None,
        eval_width=224,
        eval_height=224,
        iou_thresh=0.5,
        lane_width=5,
    ):
        """
        Evaluate the test results.
        Args:
            results (List[dict]): All inference results containing:
                result (dict): contains 'lanes' and 'scores'.
                meta (dict): contains meta information.
            metric (str): Metric type to evaluate. (not used)
            evel_width (int): image width for IoU calculation.
            evel_height (int): image height for IoU calculation.
            iou_thresh (float): IoU threshold for evaluation.
            lane_width (int): lane virtual width to calculate IoUs.
        Returns:
            dict: Evaluation result dict containing
                F1, precision, recall, etc. on the specified IoU thresholds.

        """
        from libs.datasets.metrics.curvelanes_metric import CurvelanesMetric
        
        # 使用正确的CurvelanesMetric类和现有属性
        metric_core = CurvelanesMetric(
            data_root=self.img_prefix, 
            data_list='',  # 使用空字符串，CurvelanesMetric会处理这种情况
            y_step=5
        )
        
        # 初始化结果列表
        metric_core.results = []
        
        for result in tqdm(results):
            ori_shape = result["meta"]["ori_shape"]
            filename = result["meta"]["filename"]
            sub_img_name = filename.split('/')[-1]
            
            # 获取预测车道
            pred = self.convert_coords_laneatt(result["result"], ori_shape)
            
            # 转换预测结果为需要的格式
            pred_lanes = [[[coord['x'], coord['y']] for coord in lane] for lane in pred]
            
            # 创建数据样本
            data_sample = {
                'lanes': pred_lanes,
                'metainfo': {'sub_img_name': sub_img_name}
            }
            
            # 处理结果
            metric_core.process({}, [data_sample])

        # 计算指标
        result_dict = metric_core.compute_metrics(metric_core.results)
        print(result_dict)
        return result_dict

    def convert_coords_laneatt(self, result, ori_shape):
        lanes = result["lanes"]
        scores = result["scores"]
        ys = np.arange(0, ori_shape[0], 8) / ori_shape[0]
        out = []
        for lane, score in zip(lanes, scores):
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * ori_shape[1]
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * ori_shape[0]
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_coords = []
            for x, y in zip(lane_xs, lane_ys):
                lane_coords.append({"x": x, "y": y})
            out.append(lane_coords)
        return out

    def load_labels(self, idx, offset_y=0):
        """
        重写load_labels方法以处理偏移和图像路径问题
        Args:
            idx (int): Data index.
            offset_y (int): Y轴偏移量.
        Returns:
            List[list]: list of lane point lists.
            list: class id (=1) for lane instances.
            list: instance id (start from 1) for lane instances.
        """
        if not self.test_mode and len(self.annotations) > idx:
            anno_dir = str(Path(self.img_prefix).joinpath(self.annotations[idx]))
            shapes = []
            with open(anno_dir, "r") as anno_f:
                lines = anno_f.readlines()
                for line in lines:
                    coords = []
                    coords_str = line.strip().split(" ")
                    for i in range(len(coords_str) // 2):
                        coord_x = float(coords_str[2 * i])
                        coord_y = float(coords_str[2 * i + 1]) + offset_y
                        coords.append(coord_x)
                        coords.append(coord_y)
                    if len(coords) > 3:
                        shapes.append(coords)
            id_classes = [1 for i in range(len(shapes))]
            id_instances = [i + 1 for i in range(len(shapes))]
            return shapes, id_classes, id_instances
        else:
            return [], [], []
