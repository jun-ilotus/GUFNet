import os

import torch
import yaml
import numpy as np

from PIL import Image
from torch.utils import data
from pathlib import Path

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))

@register_dataset
class WCS2D3D(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r', encoding='utf-8') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)

        seg_num_per_class = config['dataset_params']['seg_labelweights']  # 权重
        seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = torch.load(self.im_idx[index])
        if isinstance(raw_data, torch.Tensor):
            raw_data = raw_data.numpy()
        elif isinstance(raw_data, np.ndarray):
            pass  # 已经是 numpy 数组，无需转换

        origin_len = len(raw_data)
        world_points = raw_data[:, :3].astype(np.float32)  # 保留原始世界坐标
        
        # 在减中心点之前，计算图像投影所需的参数
        world_x_min, world_y_min = world_points[:, 0].min(), world_points[:, 1].min()
        world_x_max, world_y_max = world_points[:, 0].max(), world_points[:, 1].max()
        point_center = world_points.mean(axis=0)
        
        # 转换为相对坐标（减去当前帧的中心点）
        points = world_points - point_center
        
        rgb = raw_data[:, 3:6].astype(np.float32)

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 6], dtype=int), axis=1)
        else:
            annotated_data = torch.load(self.im_idx[index])
            if isinstance(annotated_data, torch.Tensor):
                annotated_data = annotated_data.numpy()
            data_np = annotated_data
            annotated_data = data_np[:, 6:7]
            #annotated_data[annotated_data == 15] = self.config['dataset_params']['ignore_label']
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

            if self.config['dataset_params']['ignore_label'] != 0:
                annotated_data -= 1
                annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        image_file = self.im_idx[index].replace('velodyne', 'image').replace('.pth', '.png')
        image = Image.open(image_file)

        # 统一转成 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 构建投影矩阵：从世界坐标(相对中心点)投影到图像像素坐标
        # 假设图像是按点云范围裁剪的外切矩形（带10%边距）
        margin = 0
        world_width = world_x_max - world_x_min
        world_height = world_y_max - world_y_min
        world_x_min_img = world_x_min - world_width * margin  # 图像对应的世界坐标范围
        world_y_min_img = world_y_min - world_height * margin
        world_width_img = world_width * (1 + 2 * margin)
        world_height_img = world_height * (1 + 2 * margin)
        
        # 计算像素分辨率（米/像素）
        img_width, img_height = image.size
        pixel_resolution_x = world_width_img / img_width
        pixel_resolution_y = world_height_img / img_height
        
        # 投影公式: img_coord = (world_coord + point_center - world_min_img) / pixel_resolution
        # 合并为矩阵形式: [scale, 0, 0, offset] @ [x, y, z, 1]^T
        scale_x = 1.0 / pixel_resolution_x
        scale_y = 1.0 / pixel_resolution_y
        offset_x = -(world_x_min_img - point_center[0]) / pixel_resolution_x
        offset_y = -(world_y_min_img - point_center[1]) / pixel_resolution_y
        
        proj_matrix = np.array([
            [scale_x, 0, 0, offset_x],
            [0, scale_y, 0, offset_y],
            [0, 0, 1, 0]
        ], dtype=np.float32)

        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = annotated_data.astype(np.uint8)
        data_dict['rgb'] = rgb
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['proj_matrix'] = proj_matrix

        return data_dict, self.im_idx[index]

from dataloader import santaclara_dataset  # 注册 SantaClara 数据集