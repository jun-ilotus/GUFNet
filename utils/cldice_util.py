import numpy as np
from pytorch_lightning.metrics import Metric

from dataloader.pc_dataset import get_SemKITTI_label_name

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None


DEFAULT_CLDICE_TARGET_CLASSES = [
    'Ditch',
    'Dam',
    'Slope',
    'Cement Road',
    'Dirt Road',
]

CLDICE_CLASS_ALIASES = {
    'Ditch': ['ditch'],
    'Dam': ['dam'],
    'Slope': ['slope'],
    'Cement Road': ['cement road', 'cementroad'],
    'Dirt Road': ['dirt road', 'dirtroad'],
}


def _normalize_class_name(name):
    if name is None:
        return ''
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def _coerce_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    try:
        return dict(config)
    except Exception:
        return {}


def _parse_target_classes(raw_value):
    if raw_value is None:
        return list(DEFAULT_CLDICE_TARGET_CLASSES)
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(',') if item.strip()]
    if isinstance(raw_value, (list, tuple)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return list(DEFAULT_CLDICE_TARGET_CLASSES)


def _build_available_name_to_id(label_mapping):
    label_name_map = get_SemKITTI_label_name(label_mapping)
    available = {}
    for class_id, class_name in label_name_map.items():
        class_id = int(class_id)
        if class_id == 0:
            continue
        available[_normalize_class_name(class_name)] = class_id
    return available


def resolve_cldice_classes(label_mapping, target_classes=None):
    available = _build_available_name_to_id(label_mapping)
    resolved = []
    seen_ids = set()

    for display_name in _parse_target_classes(target_classes):
        aliases = CLDICE_CLASS_ALIASES.get(display_name, [display_name])
        class_id = None
        for alias in aliases:
            class_id = available.get(_normalize_class_name(alias))
            if class_id is not None:
                break
        if class_id is None or class_id in seen_ids:
            continue
        seen_ids.add(class_id)
        resolved.append((display_name, class_id))
    return resolved


def _compress_axis(sorted_values):
    if len(sorted_values) == 0:
        return np.zeros((0,), dtype=np.int64)

    compressed = np.zeros(len(sorted_values), dtype=np.int64)
    for idx in range(1, len(sorted_values)):
        gap = int(sorted_values[idx] - sorted_values[idx - 1])
        compressed[idx] = compressed[idx - 1] + (1 if gap == 1 else 2)
    return compressed


def _voxelize_xy(xy, voxel_size):
    if len(xy) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    voxels = np.floor(np.asarray(xy, dtype=np.float64) / float(voxel_size)).astype(np.int64)
    return np.unique(voxels, axis=0)


def rasterize_binary_masks(pred_xy, gt_xy, voxel_size):
    pred_voxels = _voxelize_xy(pred_xy, voxel_size)
    gt_voxels = _voxelize_xy(gt_xy, voxel_size)

    if len(pred_voxels) == 0 and len(gt_voxels) == 0:
        return np.zeros((0, 0), dtype=bool), np.zeros((0, 0), dtype=bool)

    if len(pred_voxels) == 0:
        union_voxels = gt_voxels
    elif len(gt_voxels) == 0:
        union_voxels = pred_voxels
    else:
        union_voxels = np.vstack([pred_voxels, gt_voxels])

    unique_x = np.unique(union_voxels[:, 0])
    unique_y = np.unique(union_voxels[:, 1])
    compressed_x = _compress_axis(unique_x)
    compressed_y = _compress_axis(unique_y)

    width = int(compressed_x[-1] + 3) if len(compressed_x) > 0 else 0
    height = int(compressed_y[-1] + 3) if len(compressed_y) > 0 else 0
    pred_mask = np.zeros((height, width), dtype=bool)
    gt_mask = np.zeros((height, width), dtype=bool)

    if len(pred_voxels) > 0:
        pred_x = compressed_x[np.searchsorted(unique_x, pred_voxels[:, 0])] + 1
        pred_y = compressed_y[np.searchsorted(unique_y, pred_voxels[:, 1])] + 1
        pred_mask[pred_y, pred_x] = True

    if len(gt_voxels) > 0:
        gt_x = compressed_x[np.searchsorted(unique_x, gt_voxels[:, 0])] + 1
        gt_y = compressed_y[np.searchsorted(unique_y, gt_voxels[:, 1])] + 1
        gt_mask[gt_y, gt_x] = True

    return pred_mask, gt_mask


def skeletonize_binary_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    if ndimage is None:  # pragma: no cover
        raise ImportError('scipy is required for clDice evaluation but is not installed.')

    structure = ndimage.generate_binary_structure(2, 2)
    skeleton = np.zeros_like(mask, dtype=bool)
    current = mask.copy()

    while current.any():
        eroded = ndimage.binary_erosion(current, structure=structure)
        opened = ndimage.binary_dilation(eroded, structure=structure)
        skeleton |= current & ~opened
        current = eroded

    return skeleton


def compute_binary_cldice_stats(pred_mask, gt_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    pred_skeleton = skeletonize_binary_mask(pred_mask)
    gt_skeleton = skeletonize_binary_mask(gt_mask)

    return {
        'pred_skeleton_hit': float(np.count_nonzero(pred_skeleton & gt_mask)),
        'pred_skeleton_size': float(np.count_nonzero(pred_skeleton)),
        'gt_skeleton_hit': float(np.count_nonzero(gt_skeleton & pred_mask)),
        'gt_skeleton_size': float(np.count_nonzero(gt_skeleton)),
    }


def compute_pointwise_cldice_stats(xyz, pred_mask, gt_mask, voxel_size):
    xyz = np.asarray(xyz)
    pred_xy = xyz[pred_mask, :2] if np.any(pred_mask) else np.zeros((0, 2), dtype=np.float64)
    gt_xy = xyz[gt_mask, :2] if np.any(gt_mask) else np.zeros((0, 2), dtype=np.float64)

    pred_grid, gt_grid = rasterize_binary_masks(pred_xy, gt_xy, voxel_size)
    if pred_grid.size == 0 and gt_grid.size == 0:
        return {
            'pred_skeleton_hit': 0.0,
            'pred_skeleton_size': 0.0,
            'gt_skeleton_hit': 0.0,
            'gt_skeleton_size': 0.0,
        }
    return compute_binary_cldice_stats(pred_grid, gt_grid)


def cldice_from_stats(pred_skeleton_hit, pred_skeleton_size, gt_skeleton_hit, gt_skeleton_size):
    if pred_skeleton_size <= 0 and gt_skeleton_size <= 0:
        return np.nan

    topology_precision = pred_skeleton_hit / pred_skeleton_size if pred_skeleton_size > 0 else 0.0
    topology_sensitivity = gt_skeleton_hit / gt_skeleton_size if gt_skeleton_size > 0 else 0.0
    denom = topology_precision + topology_sensitivity
    if denom <= 0:
        return 0.0
    return 2.0 * topology_precision * topology_sensitivity / denom


def format_cldice_report(class_names, scores, overall, best_overall, title='Validation'):
    lines = [f'{title} per class clDice:']
    for class_name, class_score in zip(class_names, scores):
        if np.isfinite(class_score):
            lines.append(f'{class_name} : {class_score * 100:.2f}%')
        else:
            lines.append(f'{class_name} : N/A')
    lines.append(
        f'Current val clDice is {overall * 100:.3f} while the best val clDice is {best_overall * 100:.3f}'
    )
    return '\n'.join(lines)


class ClDice(Metric):
    def __init__(self, dataset_config, eval_config=None, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)

        eval_config = _coerce_dict(eval_config)
        cldice_config = _coerce_dict(eval_config.get('cldice', {}))

        self.ignore_label = int(dataset_config.get('ignore_label', 0))
        self.enabled = bool(cldice_config.get('enabled', False))
        self.voxel_size = float(cldice_config.get('voxel_size', 0.25))
        self.target_class_names = _parse_target_classes(cldice_config.get('target_classes'))
        self.class_items = resolve_cldice_classes(dataset_config['label_mapping'], self.target_class_names)
        self.class_display_names = [item[0] for item in self.class_items]
        self.class_ids = np.asarray([item[1] for item in self.class_items], dtype=np.int64)
        self.best_cldice = 0.0

        self.pred_skeleton_hits = np.zeros(len(self.class_ids), dtype=np.float64)
        self.pred_skeleton_sizes = np.zeros(len(self.class_ids), dtype=np.float64)
        self.gt_skeleton_hits = np.zeros(len(self.class_ids), dtype=np.float64)
        self.gt_skeleton_sizes = np.zeros(len(self.class_ids), dtype=np.float64)

        if len(self.class_ids) == 0:
            self.enabled = False

    def reset_state(self):
        self.pred_skeleton_hits.fill(0.0)
        self.pred_skeleton_sizes.fill(0.0)
        self.gt_skeleton_hits.fill(0.0)
        self.gt_skeleton_sizes.fill(0.0)

    def update(self, predict_labels, val_pt_labs, xyz) -> None:
        if not self.enabled:
            return

        predict_labels = np.asarray(predict_labels).reshape(-1)
        val_pt_labs = np.asarray(val_pt_labs).reshape(-1)
        xyz = np.asarray(xyz)

        if xyz.ndim == 1:
            xyz = xyz.reshape(-1, 3)
        if xyz.ndim != 2 or xyz.shape[1] < 2:
            raise ValueError(f'clDice expects xyz with shape [N, >=2], got {xyz.shape}')

        length = min(len(predict_labels), len(val_pt_labs), len(xyz))
        if length == 0:
            return

        predict_labels = predict_labels[:length]
        val_pt_labs = val_pt_labs[:length]
        xyz = xyz[:length]

        valid = val_pt_labs != self.ignore_label
        if not np.any(valid):
            return

        predict_labels = predict_labels[valid]
        val_pt_labs = val_pt_labs[valid]
        xyz = xyz[valid]

        if self.ignore_label != 0:
            predict_labels = predict_labels + 1
            val_pt_labs = val_pt_labs + 1

        for idx, class_id in enumerate(self.class_ids):
            pred_mask = predict_labels == class_id
            gt_mask = val_pt_labs == class_id
            if not np.any(pred_mask) and not np.any(gt_mask):
                continue

            stats = compute_pointwise_cldice_stats(xyz, pred_mask, gt_mask, self.voxel_size)
            self.pred_skeleton_hits[idx] += stats['pred_skeleton_hit']
            self.pred_skeleton_sizes[idx] += stats['pred_skeleton_size']
            self.gt_skeleton_hits[idx] += stats['gt_skeleton_hit']
            self.gt_skeleton_sizes[idx] += stats['gt_skeleton_size']

    def compute(self):
        if not self.enabled:
            return np.zeros((0,), dtype=np.float64), 0.0, self.best_cldice

        scores = np.full(len(self.class_ids), np.nan, dtype=np.float64)
        for idx in range(len(self.class_ids)):
            scores[idx] = cldice_from_stats(
                self.pred_skeleton_hits[idx],
                self.pred_skeleton_sizes[idx],
                self.gt_skeleton_hits[idx],
                self.gt_skeleton_sizes[idx],
            )

        finite_scores = scores[np.isfinite(scores)]
        overall = float(np.mean(finite_scores)) if finite_scores.size > 0 else 0.0
        if overall > self.best_cldice:
            self.best_cldice = overall

        self.reset_state()
        return scores, overall, self.best_cldice
