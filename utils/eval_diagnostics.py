import numpy as np
import torch

from dataloader.pc_dataset import get_SemKITTI_label_name


WCS2D3D_CLDICE_CLASSES = (
    ("Ditch", "Ditch"),
    ("Dam", "Dam"),
    ("Slope", "Slope"),
    ("Cementroad", "Cement Road"),
    ("Dirtroad", "Dirt Road"),
)


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def has_eval_gate_uncertainty(model):
    fusion = getattr(model, "fusion", None)
    if fusion is None:
        return False

    fusion_modules = getattr(fusion, "fusion_modules", None)
    if not fusion_modules:
        return False

    for module in reversed(fusion_modules):
        if hasattr(module, "ugaf"):
            return True
    return False


def extract_eval_gate_uncertainty(model):
    fusion = getattr(model, "fusion", None)
    if fusion is None:
        return None

    fusion_modules = getattr(fusion, "fusion_modules", None)
    if not fusion_modules:
        return None

    unc = None
    for module in reversed(fusion_modules):
        ugaf = getattr(module, "ugaf", None)
        if ugaf is None:
            continue
        unc = getattr(ugaf, "_last_uncertainty", None)
        if unc:
            break
    if not unc:
        return None

    # Match the current CMF reliability gating path in eval/test:
    # r = exp(-min(u_img, u_pts)), where eval uses the aleatoric terms.
    u_img = unc.get("u_img_aleatoric", None)
    u_pts = unc.get("u_pts_aleatoric", None)
    if u_img is None:
        u_img = unc.get("u_img", None)
    if u_pts is None:
        u_pts = unc.get("u_pts", None)
    if u_img is None or u_pts is None:
        return None

    return torch.min(u_img, u_pts).reshape(-1)


def split_projected_scores(flat_scores, projected_lengths):
    flat_scores = _to_numpy(flat_scores)
    if flat_scores is None:
        return None

    flat_scores = flat_scores.reshape(-1)
    projected_lengths = [int(length) for length in projected_lengths]
    total_length = int(sum(projected_lengths))
    if flat_scores.size < total_length:
        return None

    score_slices = []
    cursor = 0
    for length in projected_lengths:
        score_slices.append(flat_scores[cursor:cursor + length])
        cursor += length
    return score_slices


def build_pointwise_uncertainty(num_points, projected_indices_list, score_slices):
    point_scores_sum = np.zeros(int(num_points), dtype=np.float64)
    point_scores_count = np.zeros(int(num_points), dtype=np.int64)

    for projected_indices, scores in zip(projected_indices_list, score_slices):
        projected_indices = _to_numpy(projected_indices).reshape(-1).astype(np.int64, copy=False)
        scores = _to_numpy(scores).reshape(-1).astype(np.float64, copy=False)

        length = min(projected_indices.size, scores.size)
        if length == 0:
            continue

        projected_indices = projected_indices[:length]
        scores = scores[:length]
        valid = (
            (projected_indices >= 0)
            & (projected_indices < num_points)
            & np.isfinite(scores)
        )
        if not np.any(valid):
            continue

        np.add.at(point_scores_sum, projected_indices[valid], scores[valid])
        np.add.at(point_scores_count, projected_indices[valid], 1)

    pointwise_uncertainty = np.full(int(num_points), np.nan, dtype=np.float64)
    valid_points = point_scores_count > 0
    pointwise_uncertainty[valid_points] = (
        point_scores_sum[valid_points] / point_scores_count[valid_points]
    )
    return pointwise_uncertainty


def _zhang_suen_skeletonize(mask):
    image = (_to_numpy(mask) > 0).astype(np.uint8)
    if image.ndim != 2:
        raise ValueError("clDice expects a 2D binary mask.")
    if image.size == 0 or image.sum() == 0:
        return image.astype(bool)

    while True:
        changed = False
        for first_sub_iter in (True, False):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1))
                + ((p3 == 0) & (p4 == 1))
                + ((p4 == 0) & (p5 == 1))
                + ((p5 == 0) & (p6 == 1))
                + ((p6 == 0) & (p7 == 1))
                + ((p7 == 0) & (p8 == 1))
                + ((p8 == 0) & (p9 == 1))
                + ((p9 == 0) & (p2 == 1))
            )

            if first_sub_iter:
                cond_a = (p2 * p4 * p6) == 0
                cond_b = (p4 * p6 * p8) == 0
            else:
                cond_a = (p2 * p4 * p8) == 0
                cond_b = (p2 * p6 * p8) == 0

            removable = (
                (image == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & cond_a
                & cond_b
            )
            if np.any(removable):
                image[removable] = 0
                changed = True

        if not changed:
            break

    return image.astype(bool)


def compute_mask_cldice(pred_mask, gt_mask, eps=1e-8):
    pred_mask = (_to_numpy(pred_mask) > 0)
    gt_mask = (_to_numpy(gt_mask) > 0)

    if not np.any(gt_mask):
        return None
    if not np.any(pred_mask):
        return 0.0

    pred_skeleton = _zhang_suen_skeletonize(pred_mask)
    gt_skeleton = _zhang_suen_skeletonize(gt_mask)

    pred_skeleton_sum = float(pred_skeleton.sum())
    gt_skeleton_sum = float(gt_skeleton.sum())
    if pred_skeleton_sum <= 0.0 or gt_skeleton_sum <= 0.0:
        return 0.0

    topology_precision = float(np.logical_and(pred_skeleton, gt_mask).sum()) / (pred_skeleton_sum + eps)
    topology_sensitivity = float(np.logical_and(gt_skeleton, pred_mask).sum()) / (gt_skeleton_sum + eps)

    denom = topology_precision + topology_sensitivity
    if denom <= 0.0:
        return 0.0
    return (2.0 * topology_precision * topology_sensitivity) / denom


def rasterize_projected_binary_mask(img_indices, point_mask, image_shape):
    image_shape = tuple(int(x) for x in image_shape[:2])
    height, width = image_shape

    binary_mask = np.zeros((height, width), dtype=bool)
    img_indices = _to_numpy(img_indices).reshape(-1, 2).astype(np.int64, copy=False)
    point_mask = (_to_numpy(point_mask).reshape(-1) > 0)

    length = min(img_indices.shape[0], point_mask.size)
    if length == 0:
        return binary_mask

    rows = img_indices[:length, 0]
    cols = img_indices[:length, 1]
    valid = (
        point_mask[:length]
        & (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )
    if np.any(valid):
        binary_mask[rows[valid], cols[valid]] = True
    return binary_mask


def binary_auroc(error_targets, uncertainty_scores):
    error_targets = _to_numpy(error_targets).reshape(-1).astype(np.int64, copy=False)
    uncertainty_scores = _to_numpy(uncertainty_scores).reshape(-1).astype(np.float64, copy=False)

    valid = np.isfinite(uncertainty_scores)
    error_targets = error_targets[valid]
    uncertainty_scores = uncertainty_scores[valid]
    if error_targets.size == 0:
        return np.nan

    positive_count = int(error_targets.sum())
    negative_count = int(error_targets.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return np.nan

    sort_order = np.argsort(uncertainty_scores, kind="mergesort")
    sorted_scores = uncertainty_scores[sort_order]

    ranks = np.zeros(sorted_scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = average_rank
        start = end

    inverse_order = np.empty_like(sort_order)
    inverse_order[sort_order] = np.arange(sort_order.size)
    ranks = ranks[inverse_order]

    positive_rank_sum = float(ranks[error_targets == 1].sum())
    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


class EvalDiagnosticTracker:
    def __init__(self, dataset_config):
        self.dataset_name = str(dataset_config.get("pc_dataset_type", ""))
        self.ignore_label = int(dataset_config.get("ignore_label", 0))

        label_names = get_SemKITTI_label_name(dataset_config["label_mapping"])
        self.label_name_to_id = {
            str(class_name): int(class_id)
            for class_id, class_name in label_names.items()
        }

        self.cldice_classes = []
        if self.dataset_name == "WCS2D3D":
            for internal_name, display_name in WCS2D3D_CLDICE_CLASSES:
                class_id = self.label_name_to_id.get(internal_name, None)
                if class_id is not None:
                    self.cldice_classes.append((display_name, class_id))

        self.reset()

    def reset(self):
        self.cldice_records = {
            display_name: []
            for display_name, _ in self.cldice_classes
        }
        self.error_targets = []
        self.error_scores = []
        self.auroc_debug = {
            "uncertainty_supported": 1,
            "uncertainty_batches_requested": 0,
            "uncertainty_batches_available": 0,
            "uncertainty_batches_missing": 0,
            "uncertainty_length_mismatch_batches": 0,
            "uncertainty_points_expected": 0,
            "uncertainty_points_received": 0,
            "points_total": 0,
            "points_with_uncertainty": 0,
            "points_after_ignore": 0,
            "points_after_filter": 0,
            "error_points": 0,
            "correct_points": 0,
        }

    def note_uncertainty_not_supported(self):
        self.auroc_debug["uncertainty_supported"] = 0

    def note_uncertainty_requested(self, expected_points):
        self.auroc_debug["uncertainty_batches_requested"] += 1
        self.auroc_debug["uncertainty_points_expected"] += int(expected_points)

    def note_uncertainty_missing(self):
        self.auroc_debug["uncertainty_batches_missing"] += 1

    def note_uncertainty_available(self, received_points):
        self.auroc_debug["uncertainty_batches_available"] += 1
        self.auroc_debug["uncertainty_points_received"] += int(received_points)

    def note_uncertainty_length_mismatch(self, received_points):
        self.auroc_debug["uncertainty_length_mismatch_batches"] += 1
        self.auroc_debug["uncertainty_points_received"] += int(received_points)

    def update_projected_cldice(
        self,
        point_pred_labels,
        point_gt_labels,
        projected_point_indices,
        img_indices,
        image_shape,
    ):
        if not self.cldice_classes:
            return

        point_pred_labels = _to_numpy(point_pred_labels).reshape(-1)
        point_gt_labels = _to_numpy(point_gt_labels).reshape(-1)
        projected_point_indices = _to_numpy(projected_point_indices).reshape(-1).astype(np.int64, copy=False)
        img_indices = _to_numpy(img_indices).reshape(-1, 2).astype(np.int64, copy=False)

        length = min(projected_point_indices.size, img_indices.shape[0])
        if length == 0:
            return

        projected_point_indices = projected_point_indices[:length]
        img_indices = img_indices[:length]

        valid_point_indices = (
            (projected_point_indices >= 0)
            & (projected_point_indices < point_gt_labels.size)
        )
        if not np.any(valid_point_indices):
            return

        projected_point_indices = projected_point_indices[valid_point_indices]
        img_indices = img_indices[valid_point_indices]
        projected_pred = point_pred_labels[projected_point_indices]
        projected_gt = point_gt_labels[projected_point_indices]
        valid_gt = projected_gt != self.ignore_label

        for display_name, class_id in self.cldice_classes:
            gt_mask = rasterize_projected_binary_mask(
                img_indices,
                valid_gt & (projected_gt == class_id),
                image_shape,
            )
            if not np.any(gt_mask):
                continue

            pred_mask = rasterize_projected_binary_mask(
                img_indices,
                valid_gt & (projected_pred == class_id),
                image_shape,
            )
            score = compute_mask_cldice(pred_mask, gt_mask)
            if score is not None:
                self.cldice_records[display_name].append(float(score))

    def update_error_detection(self, point_pred_labels, point_gt_labels, pointwise_uncertainty):
        point_pred_labels = _to_numpy(point_pred_labels).reshape(-1)
        point_gt_labels = _to_numpy(point_gt_labels).reshape(-1)
        pointwise_uncertainty = _to_numpy(pointwise_uncertainty).reshape(-1).astype(np.float64, copy=False)

        length = min(point_pred_labels.size, point_gt_labels.size, pointwise_uncertainty.size)
        if length == 0:
            return

        point_pred_labels = point_pred_labels[:length]
        point_gt_labels = point_gt_labels[:length]
        pointwise_uncertainty = pointwise_uncertainty[:length]
        finite_uncertainty = np.isfinite(pointwise_uncertainty)
        non_ignore = point_gt_labels != self.ignore_label
        valid = non_ignore & finite_uncertainty

        self.auroc_debug["points_total"] += int(length)
        self.auroc_debug["points_with_uncertainty"] += int(finite_uncertainty.sum())
        self.auroc_debug["points_after_ignore"] += int(non_ignore.sum())
        self.auroc_debug["points_after_filter"] += int(valid.sum())
        if not np.any(valid):
            return

        error_targets = (point_pred_labels[valid] != point_gt_labels[valid]).astype(np.int64, copy=False)
        error_count = int(error_targets.sum())
        valid_count = int(error_targets.size)
        self.auroc_debug["error_points"] += error_count
        self.auroc_debug["correct_points"] += (valid_count - error_count)
        self.error_targets.append(error_targets)
        self.error_scores.append(pointwise_uncertainty[valid])

    def compute(self):
        cldice_scores = {}
        for display_name, _ in self.cldice_classes:
            values = self.cldice_records.get(display_name, [])
            cldice_scores[display_name] = float(np.mean(values)) if values else np.nan

        finite_cldice = [score for score in cldice_scores.values() if np.isfinite(score)]
        cldice_avg = float(np.mean(finite_cldice)) if finite_cldice else np.nan

        if self.error_targets and self.error_scores:
            error_targets = np.concatenate(self.error_targets, axis=0)
            error_scores = np.concatenate(self.error_scores, axis=0)
            auroc = binary_auroc(error_targets, error_scores)
        else:
            auroc = np.nan

        auroc_reason = None
        if not np.isfinite(auroc):
            if self.auroc_debug["uncertainty_supported"] == 0:
                auroc_reason = "the current single-modality 3D evaluation path did not expose a final gating uncertainty tensor"
            elif self.auroc_debug["uncertainty_batches_available"] == 0:
                if self.auroc_debug["uncertainty_batches_missing"] > 0:
                    auroc_reason = "final gating uncertainty was not collected from the last UGAF module"
                elif self.auroc_debug["uncertainty_length_mismatch_batches"] > 0:
                    auroc_reason = "uncertainty length mismatched the projected points"
            elif self.auroc_debug["points_after_filter"] == 0:
                if self.auroc_debug["points_with_uncertainty"] == 0:
                    auroc_reason = "no point received a finite uncertainty score after point mapping"
                elif self.auroc_debug["points_after_ignore"] == 0:
                    auroc_reason = "all candidate points were ignore-label points"
                else:
                    auroc_reason = "no valid points remained after ignore-label and finite-score filtering"
            elif self.auroc_debug["error_points"] == 0:
                auroc_reason = "all evaluated points were predicted correctly"
            elif self.auroc_debug["correct_points"] == 0:
                auroc_reason = "all evaluated points were predicted incorrectly"
            else:
                auroc_reason = "AUROC could not be computed from the collected scores"

        summary = {
            "cldice": cldice_scores,
            "cldice_avg": cldice_avg,
            "auroc_error_detection": auroc,
            "auroc_error_detection_reason": auroc_reason,
            "auroc_debug": dict(self.auroc_debug),
        }
        self.reset()
        return summary
