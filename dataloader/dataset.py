"""
Task-specific Datasets
"""
import random
import torch
import numpy as np
from scipy.spatial import cKDTree

from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from pyquaternion import Quaternion


# ---------------------------------------------------------------------------
# G3D 几何图生成工具函数
# ---------------------------------------------------------------------------

def _estimate_normals_and_curvature(xyz, k=20):
    """对点云估计逐点法向量 (nx, ny, nz) 和曲率 (curvature).

    使用 KNN + PCA (最小特征值对应法向量).

    Args:
        xyz: (N, 3) ndarray, 点云坐标.
        k: KNN 邻域大小.

    Returns:
        normals:   (N, 3) ndarray, 单位法向量.
        curvature: (N,)   ndarray, 曲率 ∈ [0, 1].
    """
    N = xyz.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)
    curvature = np.zeros(N, dtype=np.float32)

    if N < 4:
        normals[:, 2] = 1.0  # fallback: z-up
        return normals, curvature

    k = min(k, N)
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k, workers=-1)  # (N, k)

    for i in range(N):
        neighbors = xyz[idx[i]]  # (k, 3)
        cov = np.cov(neighbors, rowvar=False)  # (3, 3)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        except np.linalg.LinAlgError:
            normals[i] = [0., 0., 1.]
            continue
        normals[i] = eigvecs[:, 0]  # 最小特征值对应法向
        eig_sum = eigvals.sum()
        curvature[i] = eigvals[0] / (eig_sum + 1e-8)

    # 一致化法向方向：朝 z 正方向
    flip = normals[:, 2] < 0
    normals[flip] *= -1

    return normals, curvature


def _rasterize_per_pixel(values, img_pts, img_h, img_w, method='median'):
    """将逐点属性栅格化到图像网格.

    Args:
        values:  (M, C) 或 (M,) ndarray — 每个投影点的属性值.
        img_pts: (M, 2) int64 — [row, col] 格式的像素坐标.
        img_h, img_w: 图像尺寸.
        method:  'median' | 'mean'.

    Returns:
        raster: (H, W, C) 或 (H, W) ndarray float32.
    """
    if values.ndim == 1:
        values = values[:, np.newaxis]
    C = values.shape[1]
    raster = np.zeros((img_h, img_w, C), dtype=np.float32)
    count = np.zeros((img_h, img_w), dtype=np.int32)

    # 按像素分组
    pixel_data = {}  # {(row, col): list of value arrays}
    for i in range(len(img_pts)):
        r, c = int(img_pts[i, 0]), int(img_pts[i, 1])
        if 0 <= r < img_h and 0 <= c < img_w:
            key = (r, c)
            if key not in pixel_data:
                pixel_data[key] = []
            pixel_data[key].append(values[i])

    agg_fn = np.median if method == 'median' else np.mean
    for (r, c), vals in pixel_data.items():
        arr = np.stack(vals, axis=0)  # (n, C)
        raster[r, c] = agg_fn(arr, axis=0)

    if C == 1:
        raster = raster[:, :, 0]
    return raster


def compute_g3d_map(xyz_original, keep_idx, points_img, img_h, img_w,
                    knn_k=20):
    """从点云计算 G3D 稠密几何图，栅格化到图像网格.

    G3D 通道 (7):
      [0-2] nx, ny, nz  — 法向量
      [3]   curvature    — 曲率
      [4]   z_norm       — 归一化高度
      [5]   grad_z_x     — 图像空间 z 梯度 (水平)
      [6]   grad_z_y     — 图像空间 z 梯度 (垂直)

    Args:
        xyz_original: (N_all, 3) 原始点云坐标 (未经 3D 增强).
        keep_idx:     (N_all,) bool mask — 哪些点投影到了图像内.
        points_img:   (M, 2) int-castable — [row, col] 投影像素坐标.
        img_h, img_w: 图像尺寸.
        knn_k:        KNN 邻域大小 (默认 20).

    Returns:
        g3d: (H, W, 7) float32 ndarray.
    """
    pts = xyz_original[keep_idx]  # (M, 3)
    img_pts = points_img.astype(np.int64)

    # 若无有效投影点，返回全零 G3D
    if pts.shape[0] == 0:
        return np.zeros((img_h, img_w, 7), dtype=np.float32)

    # 1) 估计法向 + 曲率（在 3D 空间做）
    normals, curvature = _estimate_normals_and_curvature(pts, k=knn_k)

    # 2) 归一化高度
    z = pts[:, 2].copy()
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-6)

    # 3) 组合每点属性 (M, 5): [nx, ny, nz, curv, z_norm]
    per_point = np.concatenate([
        normals,                       # (M, 3)
        curvature[:, np.newaxis],      # (M, 1)
        z_norm[:, np.newaxis],         # (M, 1)
    ], axis=1)  # (M, 5)

    # 4) 栅格化到图像
    raster_5ch = _rasterize_per_pixel(per_point, img_pts, img_h, img_w,
                                      method='median')  # (H, W, 5)

    # 5) 图像空间 z 梯度（从栅格化后的 z_norm 通道用有限差分计算）
    z_raster = raster_5ch[:, :, 4]  # (H, W)
    grad_z_y = np.zeros_like(z_raster)
    grad_z_x = np.zeros_like(z_raster)
    grad_z_y[1:-1, :] = (z_raster[2:, :] - z_raster[:-2, :]) / 2.0
    grad_z_x[:, 1:-1] = (z_raster[:, 2:] - z_raster[:, :-2]) / 2.0

    # 6) 拼接为 7 通道 G3D
    g3d = np.concatenate([
        raster_5ch,                          # (H, W, 5)
        grad_z_x[:, :, np.newaxis],          # (H, W, 1)
        grad_z_y[:, :, np.newaxis],          # (H, W, 1)
    ], axis=2)  # (H, W, 7)

    return g3d.astype(np.float32)

REGISTERED_DATASET_CLASSES = {}
REGISTERED_COLATE_CLASSES = {}

try:
    from torchsparse import SparseTensor
    from torchsparse.utils.collate import sparse_collate_fn
    from torchsparse.utils.quantize import sparse_quantize
except:
    print('please install torchsparse if you want to run spvcnn/minkowskinet!')


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def register_collate_fn(cls, name=None):
    global REGISTERED_COLATE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_COLATE_CLASSES, f"exist class: {REGISTERED_COLATE_CLASSES}"
    REGISTERED_COLATE_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


def get_collate_class(name):
    global REGISTERED_COLATE_CLASSES
    assert name in REGISTERED_COLATE_CLASSES, f"available class: {REGISTERED_COLATE_CLASSES}"
    return REGISTERED_COLATE_CLASSES[name]


@register_dataset
class point_image_dataset_wcs2d3d(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.bottom_crop = config['dataset_params']['bottom_crop']
        self.resize = config['dataset_params'].get('resize', False)
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params']['image_normalizer']
        self.use_zmap = config['model_params'].get('use_zmap', True)
        # Cap the number of 3D→2D projections per sample to bound fusion VRAM.
        # 0 = disabled. Default: 250000 (covers ~97% of tiles at 300×300 crop)
        self.max_img_pts = config['dataset_params'].get('max_img_pts', 250000)

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        rgb = data['rgb']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        # WCS2D3D 数据已经在加载时转换成相对坐标，不需要再过滤空间范围
        mask = np.ones(xyz.shape[0], dtype=bool)
        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        rgb = rgb[mask]
        point_num = len(xyz)

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx] = labels[0]
                rgb[drop_idx, :] = rgb[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]

        # load 2D data
        image = data['img']
        proj_matrix = data['proj_matrix']

        # project points into image
        # WCS2D3D 不需要前向过滤（xyz[:,0]>0），保留所有点
        keep_idx = np.ones(xyz.shape[0], dtype=bool)
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image.size)
        keep_idx[keep_idx] = keep_idx_img_pts

        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]
        
        ### 生成 Z-map（高度栅格化 2.5D 辅助分支）###
        # Z-map(u,v) = median{zi | (ui,vi)=(u,v)}
        # 使用原始点云的高度信息（z坐标）生成与图像同分辨率的高度图
        img_w, img_h = image.size[0], image.size[1]
        z_map = np.zeros((img_h, img_w), dtype=np.float32)
        
        # 获取投影到图像内的点的z坐标（使用原始xyz，未经3D增强）
        valid_z = data['xyz'][keep_idx, 2]  # 使用原始高度，不受3D增强影响
        valid_img_pts = points_img.astype(np.int64)  # [row, col] 格式
        
        # 为了计算中值，需要收集每个像素位置的所有z值
        z_lists = {}  # {(row, col): [z1, z2, ...]}
        for i in range(len(valid_img_pts)):
            row, col = valid_img_pts[i, 0], valid_img_pts[i, 1]
            if 0 <= row < img_h and 0 <= col < img_w:
                key = (row, col)
                if key not in z_lists:
                    z_lists[key] = []
                z_lists[key].append(valid_z[i])
        
        # 计算每个像素的中值高度
        for (row, col), z_values in z_lists.items():
            z_map[row, col] = np.median(z_values)
        
        # 对没有点投影的像素保持0，让网络学习处理稀疏高度图

        ### 生成 G3D 几何图 (法向 + 曲率 + z_norm + 梯度, 7 通道) ###
        g3d = compute_g3d_map(
            xyz_original=data['xyz'], keep_idx=keep_idx,
            points_img=points_img, img_h=img_h, img_w=img_w,
        )  # (H, W, 7)

        ### 3D Augmentation ###
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        img_label = labels[keep_idx]
        point2img_index = np.arange(len(labels))[keep_idx]
        feat = np.concatenate((xyz, rgb), axis=1)

        ### 2D Augmentation ###
        if self.bottom_crop:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            # 使用投影点中位数中心裁剪，越界时贴齐图像边界
            crop_w, crop_h = int(self.bottom_crop[0]), int(self.bottom_crop[1])
            img_w, img_h = image.size[0], image.size[1]

            if len(points_img) > 0:
                center_x = int(np.median(points_img[:, 1]))  # col = x
                center_y = int(np.median(points_img[:, 0]))  # row = y
            else:
                center_x = img_w // 2
                center_y = img_h // 2

            max_left = max(0, img_w - crop_w)
            max_top = max(0, img_h - crop_h)

            left = int(np.clip(center_x - crop_w // 2, 0, max_left))
            top = int(np.clip(center_y - crop_h // 2, 0, max_top))
            right = int(min(img_w, left + crop_w))
            bottom = int(min(img_h, top + crop_h))

            # 若仍因边界截断，回推 left/top 保证窗口尺寸一致
            left = max(0, right - crop_w)
            top = max(0, bottom - crop_h)

            # update image points (points_img: [row(y), col(x)] after fliplr)
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image, z_map, g3d
            image = image.crop((left, top, right, bottom))
            z_map = z_map[top:bottom, left:right]  # 同步裁剪 Z-map
            g3d = g3d[top:bottom, left:right]  # 同步裁剪 G3D
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            img_label = img_label[keep_idx]
            point2img_index = point2img_index[keep_idx]

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.

        # 2D augmentation - 同步翻转 image, z_map, g3d
        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            z_map = np.ascontiguousarray(np.fliplr(z_map))  # 同步翻转 Z-map
            g3d = np.ascontiguousarray(np.fliplr(g3d))  # 同步翻转 G3D
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std
        
        # Z-map 归一化：使用高度的统计信息进行归一化
        # 计算有效高度值（非零）的统计量
        valid_z = z_map[z_map != 0]
        if len(valid_z) > 0:
            z_mean = np.mean(valid_z)
            z_std = np.std(valid_z) + 1e-6  # 避免除零
            z_map_normalized = (z_map - z_mean) / z_std
            # 无点投影的像素保持为0（表示无高度信息）
            z_map_normalized[z_map == 0] = 0
        else:
            z_map_normalized = z_map
        
        # 将 Z-map 扩展维度并与 RGB 拼接：[H, W, 3] + [H, W, 1] -> [H, W, 4]
        z_map_expanded = z_map_normalized[:, :, np.newaxis]
        image_with_zmap = np.concatenate([image, z_map_expanded], axis=2)

        data_dict = {}
        data_dict['coord'] = xyz
        data_dict['feat'] = rgb
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        if self.use_zmap:
            data_dict['img'] = image_with_zmap  # 4通道：RGB + Z-map
        else:
            data_dict['img'] = image  # 3通道：仅RGB

        # Cap projected-point count to avoid VRAM spikes from large tiles
        if self.max_img_pts > 0 and len(img_indices) > self.max_img_pts:
            sel = np.random.choice(len(img_indices), self.max_img_pts, replace=False)
            sel.sort()
            img_indices = img_indices[sel]
            img_label = img_label[sel]
            point2img_index = point2img_index[sel]

        data_dict['img_indices'] = img_indices
        data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index
        data_dict['z_map'] = z_map_normalized  # 单独保存归一化后的Z-map用于调试
        data_dict['g3d'] = g3d  # G3D 几何图 (H, W, 7)

        return data_dict


@register_dataset
class point_image_dataset_wcs2d3d_cachedgeom(point_image_dataset_wcs2d3d):
    """
    WCS2D3D/SantaClara dataset variant that prefers precomputed z_map/g3d and
    projected-point correspondences from sidecar geometry-cache NPZ files.

    If the cache is absent for a sample, it falls back to the original online
    computation path implemented in `point_image_dataset_wcs2d3d`.
    """

    def __getitem__(self, index):
        data, root = self.point_cloud_dataset[index]

        cache_available = all(
            key in data for key in ('z_map', 'g3d', 'img_indices', 'img_label', 'point2img_index')
        )
        if not cache_available:
            return super().__getitem__(index)

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label'].reshape(-1)
        rgb = data['rgb']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask = np.ones(xyz.shape[0], dtype=bool)
        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        instance_label = instance_label[mask]
        ref_index = ref_index[mask]
        rgb = rgb[mask]
        point_num = len(xyz)

        if self.dropout and self.point_cloud_dataset.imageset == 'train':
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx] = labels[0]
                rgb[drop_idx, :] = rgb[0, :]
                instance_label[drop_idx] = instance_label[0]
                ref_index[drop_idx] = ref_index[0]

        image = data['img']
        z_map = data['z_map'].astype(np.float32)
        g3d = data['g3d'].astype(np.float32)
        points_img = data['img_indices'].astype(np.float32)
        img_label = data['img_label'].astype(np.uint8)
        point2img_index = data['point2img_index'].astype(np.int64)

        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            xyz[:, 0:3] += noise_translate

        feat = np.concatenate((xyz, rgb), axis=1)

        if self.bottom_crop:
            crop_w, crop_h = int(self.bottom_crop[0]), int(self.bottom_crop[1])
            img_w, img_h = image.size[0], image.size[1]

            if len(points_img) > 0:
                center_x = int(np.median(points_img[:, 1]))
                center_y = int(np.median(points_img[:, 0]))
            else:
                center_x = img_w // 2
                center_y = img_h // 2

            max_left = max(0, img_w - crop_w)
            max_top = max(0, img_h - crop_h)

            left = int(np.clip(center_x - crop_w // 2, 0, max_left))
            top = int(np.clip(center_y - crop_h // 2, 0, max_top))
            right = int(min(img_w, left + crop_w))
            bottom = int(min(img_h, top + crop_h))

            left = max(0, right - crop_w)
            top = max(0, bottom - crop_h)

            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            image = image.crop((left, top, right, bottom))
            z_map = z_map[top:bottom, left:right]
            g3d = g3d[top:bottom, left:right]
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            img_label = img_label[keep_idx]
            point2img_index = point2img_index[keep_idx]

        img_indices = points_img.astype(np.int64)

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image = np.array(image, dtype=np.float32, copy=False) / 255.

        if np.random.rand() < self.flip2d:
            image = np.ascontiguousarray(np.fliplr(image))
            z_map = np.ascontiguousarray(np.fliplr(z_map))
            g3d = np.ascontiguousarray(np.fliplr(g3d))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        valid_z = z_map[z_map != 0]
        if len(valid_z) > 0:
            z_mean = np.mean(valid_z)
            z_std = np.std(valid_z) + 1e-6
            z_map_normalized = (z_map - z_mean) / z_std
            z_map_normalized[z_map == 0] = 0
        else:
            z_map_normalized = z_map

        z_map_expanded = z_map_normalized[:, :, np.newaxis]
        image_with_zmap = np.concatenate([image, z_map_expanded], axis=2)

        data_dict = {}
        data_dict['coord'] = xyz
        data_dict['feat'] = rgb
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        if self.use_zmap:
            data_dict['img'] = image_with_zmap
        else:
            data_dict['img'] = image

        if self.max_img_pts > 0 and len(img_indices) > self.max_img_pts:
            sel = np.random.choice(len(img_indices), self.max_img_pts, replace=False)
            sel.sort()
            img_indices = img_indices[sel]
            img_label = img_label[sel]
            point2img_index = point2img_index[sel]

        data_dict['img_indices'] = img_indices
        data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index
        data_dict['z_map'] = z_map_normalized
        data_dict['g3d'] = g3d

        return data_dict


@register_collate_fn
def collate_fn_default(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    ref_labels = data[0]['ref_label']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    point2img_index = [torch.from_numpy(d['point2img_index']).long() for d in data]
    path = [d['root'] for d in data]

    img = [torch.from_numpy(d['img']) for d in data]
    img_indices = [d['img_indices'] for d in data]
    img_label = [torch.from_numpy(d['img_label']) for d in data]
    z_map = [torch.from_numpy(d['z_map']) for d in data]
    img_shape = [(d['img'].shape[0], d['img'].shape[1]) for d in data]

    # 统一图像尺寸：找到batch中的最大尺寸，其他图像padding到该尺寸
    img_shapes = [im.shape for im in img]  # [(H1, W1, C), (H2, W2, C), ...]
    max_h = max([s[0] for s in img_shapes])
    max_w = max([s[1] for s in img_shapes])

    # 确保尺寸能被16整除（ResNet等网络需要）
    max_h = ((max_h + 15) // 16) * 16
    max_w = ((max_w + 15) // 16) * 16

    # Pad所有图像到统一尺寸
    img_padded = []
    for i, im in enumerate(img):
        h, w, c = im.shape
        if h != max_h or w != max_w:
            # 创建padding后的图像（填充0，对应归一化后的黑色）
            padded = torch.zeros(max_h, max_w, c, dtype=im.dtype)
            padded[:h, :w, :] = im
            img_padded.append(padded)
        else:
            img_padded.append(im)
    img = img_padded

    # Pad所有z_map到统一尺寸
    z_map_padded = []
    for i, z in enumerate(z_map):
        h, w = z.shape
        if h != max_h or w != max_w:
            padded = torch.zeros(max_h, max_w, dtype=z.dtype)
            padded[:h, :w] = z
            z_map_padded.append(padded)
        else:
            z_map_padded.append(z)
    z_map = z_map_padded

    # Pad G3D 几何图（如果数据集提供，用于 GATA G3D 流）
    has_g3d = 'g3d' in data[0]
    g3d_batch = None
    if has_g3d:
        g3d_list = [torch.from_numpy(d['g3d']) for d in data]  # (H, W, dg)
        g3d_padded = []
        for g in g3d_list:
            h, w = g.shape[0], g.shape[1]
            if h != max_h or w != max_w:
                if g.dim() == 2:
                    padded = torch.zeros(max_h, max_w, dtype=g.dtype)
                    padded[:h, :w] = g
                else:
                    padded = torch.zeros(max_h, max_w, g.shape[2], dtype=g.dtype)
                    padded[:h, :w, :] = g
                g3d_padded.append(padded)
            else:
                g3d_padded.append(g)
        # -> (B, dg, H, W)
        stacked = torch.stack(g3d_padded, 0)
        if stacked.dim() == 3:
            stacked = stacked.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        else:
            stacked = stacked.permute(0, 3, 1, 2)  # (B, H, W, dg) -> (B, dg, H, W)
        g3d_batch = stacked

    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
    points = [torch.from_numpy(d['point_feat']) for d in data]
    # rgb = [torch.from_numpy(d['feat']) for d in data]
    coord = [torch.from_numpy(d['ref_xyz']) for d in data]
    ref_xyz = [torch.from_numpy(d['ref_xyz']) for d in data]
    labels = [torch.from_numpy(d['point_label']) for d in data]

    result = {
        'points': torch.cat(points).float(),
        'ref_xyz': torch.cat(ref_xyz).float(),
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).long().squeeze(1),
        'raw_labels': torch.from_numpy(ref_labels).long(),
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(),
        'point2img_index': point2img_index,
        'img': torch.stack(img, 0).permute(0, 3, 1, 2),
        'img_indices': img_indices,
        'img_label': torch.cat(img_label, 0).squeeze(1).long(),
        'z_map': torch.stack(z_map, 0),
        'img_shape': img_shape,
        'path': path,
        'point_xyz': torch.cat(coord).float(),
        # 'point_colors': torch.cat(rgb).float()
    }
    if g3d_batch is not None:
        result['g3d'] = g3d_batch
    return result


@register_collate_fn
def collate_fn_voxel(inputs):
    return sparse_collate_fn(inputs)
