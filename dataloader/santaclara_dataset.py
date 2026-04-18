"""
SantaClara / N3C-California dataset loader.

Supported modes:
1. Preprocessed NPZ tiles created by the local pipeline.
2. Raw IKDNet-style `.laz + _img.tif` samples.

Raw-label remapping is controlled by `dataset_params.label_mapping`.
"""
import glob
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils import data

from dataloader.pc_dataset import register_dataset

try:
    import laspy
except ImportError:
    laspy = None

try:
    import tifffile
except ImportError:
    tifffile = None


def _is_missing_laz_backend_error(exc):
    message = str(exc).lower()
    return (
        "no lazbackend selected" in message
        or "cannot decompress data" in message
        or ("laz" in message and "backend" in message)
    )


def _is_missing_imagecodecs_error(exc):
    message = str(exc).lower()
    return "requires the 'imagecodecs' package" in message or "requires the imagecodecs package" in message


def _unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _to_rgb_uint8(image_np):
    image_np = np.asarray(image_np)

    if image_np.ndim == 2:
        image_np = np.repeat(image_np[:, :, None], 3, axis=2)
    elif image_np.ndim == 3:
        if image_np.shape[0] in (3, 4) and image_np.shape[-1] not in (3, 4):
            image_np = np.transpose(image_np, (1, 2, 0))
        if image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=2)
        elif image_np.shape[-1] > 3:
            image_np = image_np[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape: {image_np.shape}")

    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.floating):
            image_np = np.clip(image_np, 0.0, 1.0) * 255.0
        else:
            image_np = np.clip(image_np, 0, 255)
        image_np = image_np.astype(np.uint8)

    return np.ascontiguousarray(image_np)


def _project_points_to_image(xyz_centered, proj_matrix, image_size):
    img_width, img_height = image_size
    if xyz_centered.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    points_hcoords = np.concatenate(
        [xyz_centered.astype(np.float64), np.ones((xyz_centered.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    projected = (proj_matrix @ points_hcoords.T).T
    denom = projected[:, 2]
    valid = np.abs(denom) > 1e-6
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float64)

    img_points = projected[valid, :2] / denom[valid, None]
    keep = (
        (img_points[:, 0] > 0)
        & (img_points[:, 1] > 0)
        & (img_points[:, 0] < img_width)
        & (img_points[:, 1] < img_height)
    )
    return np.fliplr(img_points[keep])


@register_dataset
class SantaClara(data.Dataset):
    """
    SantaClara / N3C-California dataset loader.

    Raw-data layouts supported:
    - `data_path/train|val|test/**/*.laz`
    - `data_path/lidar/train|val|test/**/*.laz`
    - `data_path/laz/train|val|test/**/*.laz`

    Images can either share the same root or be placed in a separate
    `dataset_params.image_data_path` directory with mirrored relative paths.
    """

    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r', encoding='utf-8') as stream:
            label_yaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = label_yaml['learning_map']
        self.imageset = imageset
        self.data_path = os.path.abspath(data_path)
        self.dataset_params = config['dataset_params']

        self.preprocess_cfg = self.dataset_params.get('preprocessed_data', {})
        self.use_preprocessed = bool(self.preprocess_cfg.get('enabled', False))
        self.preprocessed_root = self.preprocess_cfg.get('path', '')
        self.geometry_cache_cfg = self.dataset_params.get('precomputed_geometry', {})
        self.use_geometry_cache = bool(self.geometry_cache_cfg.get('enabled', False))
        self.geometry_cache_root = self.geometry_cache_cfg.get('path', '')
        self.geometry_cache_root = os.path.abspath(self.geometry_cache_root) if self.geometry_cache_root else ''
        self._geometry_cache_warned = False

        self.image_data_path = self.dataset_params.get('image_data_path', '')
        self.image_data_path = os.path.abspath(self.image_data_path) if self.image_data_path else ''
        self.image_suffix = self.dataset_params.get('image_suffix', '_img.tif')
        self.sample_base_root = {}

        self._build_label_lut()

        split_dir = self._resolve_split_dir(imageset)
        self.im_idx = []

        if self.use_preprocessed and self.preprocessed_root:
            split_root = os.path.join(self.preprocessed_root, split_dir)
            npz_files = sorted(glob.glob(os.path.join(split_root, '**', '*.npz'), recursive=True))
            if len(npz_files) == 0:
                npz_files = sorted(glob.glob(os.path.join(split_root, '*.npz')))

            if len(npz_files) > 0:
                for f in npz_files:
                    for _ in range(num_vote):
                        self.im_idx.append(f)

                print(
                    f"[SantaClara-{imageset}] Using preprocessed NPZ, collected {len(self.im_idx)} samples "
                    f"({len(npz_files)} files x {num_vote} votes)"
                )
            else:
                print(
                    f"[SantaClara-{imageset}] WARN: preprocessed_data enabled but no npz found in {split_root}, "
                    f"fallback to raw LAZ"
                )

        if len(self.im_idx) == 0:
            raw_files = self._collect_raw_files(split_dir)
            for f in raw_files:
                for _ in range(num_vote):
                    self.im_idx.append(f)

            print(
                f"[SantaClara-{imageset}] Using raw LAZ + TIF, collected {len(self.im_idx)} samples "
                f"({len(raw_files)} files x {num_vote} votes)"
            )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = np.array(config['dataset_params']['seg_labelweights'], dtype=np.float64)
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            self.seg_labelweights = np.power(
                np.amax(seg_labelweights) / (seg_labelweights + 1e-8), 1 / 3.0
            )
        else:
            self.seg_labelweights = np.ones(config['model_params']['num_classes'], dtype=np.float32)

    @staticmethod
    def _resolve_split_dir(imageset):
        if imageset == 'train':
            return 'train'
        if imageset == 'val':
            return 'val'
        if imageset == 'test':
            return 'test'
        raise ValueError(f"Invalid imageset: {imageset}. Must be train/val/test.")

    def _build_label_lut(self):
        self.label_lut = np.zeros(256, dtype=np.int32)
        for orig_label, mapped_label in self.learning_map.items():
            orig_label = int(orig_label)
            if 0 <= orig_label < 256:
                self.label_lut[orig_label] = int(mapped_label)

    def _collect_raw_files(self, split_dir):
        candidate_dirs = [
            os.path.join(self.data_path, split_dir),
            os.path.join(self.data_path, 'lidar', split_dir),
            os.path.join(self.data_path, 'laz', split_dir),
        ]

        files = []
        for split_root in candidate_dirs:
            patterns = [
                os.path.join(split_root, '**', '*.laz'),
                os.path.join(split_root, '**', '*.las'),
                os.path.join(split_root, '*.laz'),
                os.path.join(split_root, '*.las'),
            ]

            matched = []
            for pattern in patterns:
                matched.extend(glob.glob(pattern, recursive='**' in pattern))

            matched = sorted({os.path.abspath(path) for path in matched})
            base_root = os.path.abspath(os.path.dirname(split_root))
            for path in matched:
                self.sample_base_root[path] = base_root
            files.extend(matched)

        files = sorted(set(files))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No raw SantaClara files found for split '{split_dir}' under {self.data_path}. "
                f"Expected one of: {candidate_dirs}"
            )
        return files

    def _resolve_image_path(self, sample_path):
        sample_path = os.path.abspath(sample_path)
        sample_base_root = self.sample_base_root.get(sample_path, self.data_path)

        try:
            rel_path = os.path.relpath(sample_path, sample_base_root)
        except ValueError:
            rel_path = os.path.basename(sample_path)

        rel_stem = os.path.splitext(rel_path)[0]
        file_stem = Path(rel_stem).name

        rel_candidates = [
            rel_stem + self.image_suffix,
            rel_stem + '.tif',
            rel_stem + '.tiff',
            file_stem + self.image_suffix,
            file_stem + '.tif',
            file_stem + '.tiff',
        ]

        sample_path_png = sample_path.replace('.laz', '.png').replace('.las', '.png')
        direct_candidates = [
            sample_path.replace('.laz', self.image_suffix).replace('.las', self.image_suffix),
            sample_path.replace('.laz', '.tif').replace('.las', '.tif'),
            sample_path.replace('.laz', '.tiff').replace('.las', '.tiff'),
            sample_path_png,
            sample_path.replace(f'{os.sep}lidar{os.sep}', f'{os.sep}image{os.sep}').replace('.laz', self.image_suffix),
            sample_path.replace(f'{os.sep}laz{os.sep}', f'{os.sep}image{os.sep}').replace('.laz', self.image_suffix),
        ]

        roots = _unique_preserve_order([
            self.image_data_path,
            os.path.join(self.data_path, 'image'),
            self.data_path,
            os.path.join(sample_base_root, 'image'),
            os.path.dirname(sample_path),
        ])

        for candidate in direct_candidates:
            if os.path.exists(candidate):
                return candidate

        for root in roots:
            for rel_candidate in rel_candidates:
                candidate = os.path.abspath(os.path.join(root, rel_candidate))
                if os.path.exists(candidate):
                    return candidate

        raise FileNotFoundError(
            f"Unable to find image for {sample_path}. "
            f"Tried roots={roots} with relative paths={rel_candidates}"
        )

    @staticmethod
    def _extract_point_features(las):
        intensity = np.asarray(las.intensity, dtype=np.float32).reshape(-1, 1)
        if hasattr(las, 'return_num'):
            return_num = np.asarray(las.return_num, dtype=np.float32).reshape(-1, 1)
        elif hasattr(las, 'return_number'):
            return_num = np.asarray(las.return_number, dtype=np.float32).reshape(-1, 1)
        else:
            return_num = np.ones_like(intensity, dtype=np.float32)
        return np.concatenate([intensity, return_num], axis=1)

    @staticmethod
    def _build_model_to_raster(tags):
        if 34264 in tags:
            transform = np.asarray(tags[34264].value, dtype=np.float64).reshape(4, 4)
            raster_to_model = np.array([
                [transform[0, 0], transform[0, 1], transform[0, 3]],
                [transform[1, 0], transform[1, 1], transform[1, 3]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            return np.linalg.inv(raster_to_model)

        if 33550 in tags and 33922 in tags:
            pixel_scale = np.asarray(tags[33550].value, dtype=np.float64)
            tiepoints = np.asarray(tags[33922].value, dtype=np.float64)
            if pixel_scale.size >= 2 and tiepoints.size >= 6:
                scale_x = abs(float(pixel_scale[0]))
                scale_y = abs(float(pixel_scale[1]))
                raster_x, raster_y = float(tiepoints[0]), float(tiepoints[1])
                model_x, model_y = float(tiepoints[3]), float(tiepoints[4])
                return np.array([
                    [1.0 / scale_x, 0.0, raster_x - model_x / scale_x],
                    [0.0, -1.0 / scale_y, raster_y + model_y / scale_y],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

        return None

    @staticmethod
    def _build_projection_from_affine(model_to_raster, point_center):
        point_center = np.asarray(point_center, dtype=np.float64)
        offset_x = (
            model_to_raster[0, 0] * point_center[0]
            + model_to_raster[0, 1] * point_center[1]
            + model_to_raster[0, 2]
        )
        offset_y = (
            model_to_raster[1, 0] * point_center[0]
            + model_to_raster[1, 1] * point_center[1]
            + model_to_raster[1, 2]
        )
        return np.array([
            [model_to_raster[0, 0], model_to_raster[0, 1], 0.0, offset_x],
            [model_to_raster[1, 0], model_to_raster[1, 1], 0.0, offset_y],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    @staticmethod
    def _build_fallback_projection(points_world, image_size, point_center):
        img_width, img_height = image_size
        world_x_min, world_y_min = points_world[:, 0].min(), points_world[:, 1].min()
        world_x_max, world_y_max = points_world[:, 0].max(), points_world[:, 1].max()
        world_width = max(world_x_max - world_x_min, 1e-6)
        world_height = max(world_y_max - world_y_min, 1e-6)

        scale_x = img_width / world_width
        scale_y = img_height / world_height
        offset_x = (point_center[0] - world_x_min) * scale_x
        offset_y = (world_y_max - point_center[1]) * scale_y

        return np.array([
            [scale_x, 0.0, 0.0, offset_x],
            [0.0, -scale_y, 0.0, offset_y],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def _select_projection_matrix(self, points_world, centered_points, image_size, point_center, model_to_raster):
        proj_matrix = None
        if model_to_raster is not None:
            affine_proj = self._build_projection_from_affine(model_to_raster, point_center)
            affine_points = _project_points_to_image(centered_points, affine_proj, image_size)
            min_valid_points = max(8, int(0.001 * max(len(centered_points), 1)))
            if len(affine_points) >= min_valid_points:
                proj_matrix = affine_proj

        if proj_matrix is None:
            proj_matrix = self._build_fallback_projection(points_world, image_size, point_center)

        return proj_matrix

    def _load_preprocessed_sample(self, sample_path):
        with np.load(sample_path, allow_pickle=True) as data:
            centered_points = data['xyz'].astype(np.float32)
            labels = data['labels'].astype(np.uint8)
            feat_key = 'point_feat' if 'point_feat' in data else 'rgb'
            feat = data[feat_key].astype(np.float32)
            image_np = data['img'].astype(np.uint8)
            proj_matrix = data['proj_matrix'].astype(np.float32)
            origin_len = int(data['origin_len'])
            if 'root' in data:
                pc_path = str(data['root'].item())
            else:
                pc_path = sample_path

        image = Image.fromarray(image_np, mode='RGB')
        data_dict = {
            'xyz': centered_points,
            'labels': labels,
            'instance_label': labels.copy(),
            'rgb': feat,
            'origin_len': origin_len,
            'img': image,
            'proj_matrix': proj_matrix,
        }

        if self.use_geometry_cache and self.geometry_cache_root:
            geometry_path = self._resolve_geometry_cache_path(sample_path)
            if geometry_path and os.path.exists(geometry_path):
                with np.load(geometry_path, allow_pickle=True) as geometry:
                    for key in ('z_map', 'g3d', 'img_indices', 'img_label', 'point2img_index'):
                        if key in geometry:
                            data_dict[key] = geometry[key]
            elif not self._geometry_cache_warned:
                print(
                    f"[SantaClara-{self.imageset}] WARN: geometry cache enabled but file not found: "
                    f"{geometry_path}. Fallback to online z_map/g3d computation."
                )
                self._geometry_cache_warned = True

        return data_dict, pc_path

    def _resolve_geometry_cache_path(self, sample_path):
        if not self.geometry_cache_root:
            return None
        try:
            rel_path = os.path.relpath(sample_path, self.preprocessed_root)
        except ValueError:
            rel_path = os.path.basename(sample_path)
        return os.path.abspath(os.path.join(self.geometry_cache_root, rel_path))

    def _load_raw_sample(self, sample_path):
        if laspy is None:
            raise ImportError(
                "Raw SantaClara loading requires `laspy[lazrs]`. "
                "Please install it before using `.laz` files."
            )
        if tifffile is None:
            raise ImportError(
                "Raw SantaClara loading requires `tifffile`. "
                "Please install it before using GeoTIFF images."
            )

        try:
            las = laspy.read(sample_path)
        except Exception as exc:
            if _is_missing_laz_backend_error(exc):
                raise RuntimeError(
                    f"Failed to read LAZ file: {sample_path}\n"
                    f"Original error: {exc}\n\n"
                    f"The environment has `laspy` but no LAZ decompression backend.\n"
                    f"Install one of these and rerun:\n"
                    f"  pip install \"laspy[lazrs]\"\n"
                    f"or:\n"
                    f"  pip install lazrs\n"
                ) from exc
            raise
        points_world = np.stack([
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ], axis=1)

        point_center = points_world.mean(axis=0)
        centered_points = (points_world - point_center).astype(np.float32)
        origin_len = int(points_world.shape[0])

        raw_labels = np.asarray(las.classification, dtype=np.int32)
        raw_labels = np.clip(raw_labels, 0, len(self.label_lut) - 1)
        labels = self.label_lut[raw_labels].astype(np.uint8).reshape(-1, 1)
        point_feat = self._extract_point_features(las)

        image_path = self._resolve_image_path(sample_path)
        try:
            with tifffile.TiffFile(image_path) as tif:
                image_np = _to_rgb_uint8(tif.asarray())
                model_to_raster = self._build_model_to_raster(tif.pages[0].tags)
        except Exception as exc:
            if _is_missing_imagecodecs_error(exc):
                raise RuntimeError(
                    f"Failed to read TIFF image: {image_path}\n"
                    f"Original error: {exc}\n\n"
                    f"This image appears to use a codec handled by `imagecodecs` "
                    f"(for example LZW-compressed TIFF).\n"
                    f"Install it and rerun:\n"
                    f"  pip install imagecodecs\n"
                ) from exc
            raise

        image = Image.fromarray(image_np)
        proj_matrix = self._select_projection_matrix(
            points_world=points_world,
            centered_points=centered_points,
            image_size=image.size,
            point_center=point_center,
            model_to_raster=model_to_raster,
        )

        data_dict = {
            'xyz': centered_points,
            'labels': labels,
            'instance_label': labels.copy(),
            'rgb': point_feat,
            'origin_len': origin_len,
            'img': image,
            'proj_matrix': proj_matrix,
        }
        return data_dict, sample_path

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        sample_path = self.im_idx[index]
        if sample_path.lower().endswith('.npz'):
            return self._load_preprocessed_sample(sample_path)
        return self._load_raw_sample(sample_path)
