#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import yaml
import json
import time
import numpy as np
import pytorch_lightning as pl

from datetime import datetime
from pytorch_lightning.metrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from utils.metric_util import IoU, mAcc
from utils.cldice_util import ClDice, format_cldice_report
from utils.eval_diagnostics import (
    EvalDiagnosticTracker,
    build_pointwise_uncertainty,
    extract_eval_gate_uncertainty,
    has_eval_gate_uncertainty,
    split_projected_scores,
)
from utils.schedulers import cosine_schedule_with_warmup
from utils.puaf_vis import maybe_visualize_puaf

try:
    import laspy
except ImportError:
    laspy = None


class LightningBaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._last_step_end_time = time.perf_counter()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.val_iou = IoU(self.args['dataset_params'], compute_on_step=False)
        self.val_macc = mAcc(self.args['dataset_params'], compute_on_step=False)
        self.val_cldice = ClDice(
            self.args['dataset_params'],
            self.args.get('eval_metrics', None),
            compute_on_step=False,
        )
        self._test_metric_batches = 0
        self._save_test_predictions = bool(self.args.get('save_test_predictions', False))

        if self._should_save_test_predictions():
            ckpt_path = self.args.get('checkpoint', None)
            base_dir = os.path.dirname(ckpt_path) if ckpt_path else '.'
            self.submit_dir = os.path.join(base_dir, 'submit_' + datetime.now().strftime('%Y_%m_%d'))
            with open(self.args['dataset_params']['label_mapping'], 'r') as stream:
                self.mapfile = yaml.safe_load(stream)

        self.ignore_label = self.args['dataset_params']['ignore_label']
        self.eval_diagnostics = EvalDiagnosticTracker(self.args['dataset_params'])

    def configure_optimizers(self):
        if self.args['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.args['train_params']["learning_rate"])
        elif self.args['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.args['train_params']["learning_rate"],
                                        momentum=self.args['train_params']["momentum"],
                                        weight_decay=self.args['train_params']["weight_decay"],
                                        nesterov=self.args['train_params']["nesterov"])
        else:
            raise NotImplementedError

        if self.args['train_params']["lr_scheduler"] == 'StepLR':
            lr_scheduler = StepLR(
                optimizer,
                step_size=self.args['train_params']["decay_step"],
                gamma=self.args['train_params']["decay_rate"]
            )
        elif self.args['train_params']["lr_scheduler"] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.args['train_params']["decay_rate"],
                patience=self.args['train_params']["decay_step"],
                verbose=True
            )
        elif self.args['train_params']["lr_scheduler"] == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.args['train_params']['max_num_epochs'] - 4,
                eta_min=1e-5,
            )
        elif self.args['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts':
            from functools import partial
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=partial(
                    cosine_schedule_with_warmup,
                    num_epochs=self.args['train_params']['max_num_epochs'],
                    batch_size=self.args['dataset_params']['train_data_loader']['batch_size'],
                    dataset_size=self.args['dataset_params']['training_size'],
                    num_gpu=len(self.args.gpu)
                ),
            )
        else:
            raise NotImplementedError

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step' if self.args['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts' else 'epoch',
            'frequency': 1
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.args.monitor,
        }

    def forward(self, data):
        pass

    def _should_collect_eval_diagnostics(self):
        return not self._should_save_test_predictions()

    def _should_save_test_predictions(self):
        return bool(self.args.get('submit_to_server', False) and self._save_test_predictions)

    def _get_eval_diagnostic_cfg(self):
        cfg = self.args.get('eval_diagnostics', None)
        if cfg is None:
            return {}
        return cfg

    def _should_collect_eval_uncertainty(self):
        diag_cfg = self._get_eval_diagnostic_cfg()
        return (
            self._should_collect_eval_diagnostics()
            and bool(diag_cfg.get('aux_fusion_uncertainty', False))
            and hasattr(self, 'fusion')
        )

    def _prepare_eval_diagnostic_forward(self, data_dict):
        if self._should_collect_eval_uncertainty():
            data_dict['_run_puaf_vis'] = True
        return data_dict

    def _split_projected_uncertainty(self, data_dict):
        if not self._should_collect_eval_uncertainty():
            self.eval_diagnostics.note_uncertainty_not_supported()
            return None

        projected_lengths = [len(item) for item in data_dict.get('img_indices', [])]
        expected_points = int(sum(projected_lengths))
        self.eval_diagnostics.note_uncertainty_requested(expected_points)

        if not has_eval_gate_uncertainty(self):
            self.eval_diagnostics.note_uncertainty_not_supported()
            return None

        flat_uncertainty = extract_eval_gate_uncertainty(self)
        if flat_uncertainty is None:
            self.eval_diagnostics.note_uncertainty_missing()
            return None

        if torch.is_tensor(flat_uncertainty):
            received_points = int(flat_uncertainty.numel())
        else:
            received_points = int(np.asarray(flat_uncertainty).size)

        if received_points < expected_points:
            self.eval_diagnostics.note_uncertainty_length_mismatch(received_points)
            return None

        self.eval_diagnostics.note_uncertainty_available(received_points)
        return split_projected_scores(flat_uncertainty, projected_lengths)

    def _collect_validation_diagnostics(self, data_dict):
        if not self._should_collect_eval_diagnostics():
            return
        if 'batch_idx' not in data_dict or 'point2img_index' not in data_dict or 'img_indices' not in data_dict:
            return

        point_prediction = data_dict['logits'].argmax(1).detach().cpu().numpy()
        point_gt = data_dict['labels'].detach().cpu().numpy()
        batch_size = len(data_dict['point2img_index'])
        point_counts = torch.bincount(
            data_dict['batch_idx'].detach().cpu(),
            minlength=batch_size,
        ).tolist()
        uncertainty_slices = self._split_projected_uncertainty(data_dict)

        point_cursor = 0
        for sample_idx, point_count in enumerate(point_counts):
            point_count = int(point_count)
            point_slice = slice(point_cursor, point_cursor + point_count)
            point_cursor += point_count

            projected_indices = data_dict['point2img_index'][sample_idx]
            img_indices = data_dict['img_indices'][sample_idx]
            img_shape = data_dict['img_shape'][sample_idx]

            self.eval_diagnostics.update_projected_cldice(
                point_prediction[point_slice],
                point_gt[point_slice],
                projected_indices,
                img_indices,
                img_shape,
            )

            if uncertainty_slices is None or sample_idx >= len(uncertainty_slices):
                continue

            pointwise_uncertainty = build_pointwise_uncertainty(
                num_points=point_count,
                projected_indices_list=[projected_indices],
                score_slices=[uncertainty_slices[sample_idx]],
            )
            self.eval_diagnostics.update_error_detection(
                point_prediction[point_slice],
                point_gt[point_slice],
                pointwise_uncertainty,
            )

    def _collect_test_diagnostics(self, data_dict, point_prediction, point_gt):
        if not self._should_collect_eval_diagnostics():
            return
        if 'point2img_index' not in data_dict or not data_dict['point2img_index']:
            return

        self.eval_diagnostics.update_projected_cldice(
            point_prediction,
            point_gt,
            data_dict['point2img_index'][0],
            data_dict['img_indices'][0],
            data_dict['img_shape'][0],
        )

        uncertainty_slices = self._split_projected_uncertainty(data_dict)
        if uncertainty_slices is None:
            return

        pointwise_uncertainty = build_pointwise_uncertainty(
            num_points=len(point_prediction),
            projected_indices_list=list(data_dict['point2img_index']),
            score_slices=uncertainty_slices,
        )
        self.eval_diagnostics.update_error_detection(
            point_prediction,
            point_gt,
            pointwise_uncertainty,
        )

    @staticmethod
    def _format_ratio(value):
        return 'N/A' if not np.isfinite(value) else f'{value * 100:.2f}%'

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _normalize_origin_len(origin_len):
        if torch.is_tensor(origin_len):
            return int(origin_len.reshape(-1)[0].item())
        if isinstance(origin_len, np.ndarray):
            return int(origin_len.reshape(-1)[0])
        if isinstance(origin_len, (list, tuple)):
            return int(origin_len[0])
        return int(origin_len)

    @staticmethod
    def _split_path_components(path):
        normalized = str(path).replace('\\', '/')
        return [item for item in normalized.split('/') if item]

    def _build_text_prediction_path(self, path):
        components = self._split_path_components(path)
        if components:
            filename = os.path.splitext(components[-1])[0] + '.txt'
        else:
            filename = 'prediction.txt'

        parent = components[-2] if len(components) >= 2 else 'default'
        grandparent = components[-3] if len(components) >= 3 else 'default'
        technical_dirs = {'velodyne', 'velodyne_reduced', 'image', 'image_2', 'lidar', 'laz', 'las'}

        if parent in technical_dirs:
            rel_parts = [grandparent]
        elif grandparent in {'train', 'val', 'test'}:
            rel_parts = [grandparent, parent]
        else:
            rel_parts = [grandparent]

        return os.path.join(self.submit_dir, 'predictions', *rel_parts, filename)

    def _mapped_to_original_labels(self, mapped_prediction):
        mapped_prediction = np.asarray(mapped_prediction, dtype=np.int32).reshape(-1)
        inv_map = self.mapfile.get('learning_map_inv', {})
        if not inv_map:
            return mapped_prediction

        inv_map = {int(k): int(v) for k, v in inv_map.items()}
        return np.vectorize(lambda x: inv_map.get(int(x), int(x)), otypes=[np.int32])(mapped_prediction)

    def _resolve_export_labels_and_rgb(self, mapped_prediction, use_bgr_color_map=False):
        mapped_prediction = np.asarray(mapped_prediction, dtype=np.int32).reshape(-1)
        original_prediction = self._mapped_to_original_labels(mapped_prediction)

        color_map = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in self.mapfile.get('color_map', {}).items()
        }
        color_keys = set(color_map.keys())
        mapped_keys = set(int(v) for v in np.unique(mapped_prediction))
        original_keys = set(int(v) for v in np.unique(original_prediction))

        mapped_color_available = mapped_keys.issubset(color_keys)
        original_color_available = original_keys.issubset(color_keys)

        if mapped_color_available and not original_color_available:
            export_label = mapped_prediction
            color_lookup = mapped_prediction
        elif original_color_available:
            export_label = original_prediction
            color_lookup = original_prediction
        elif mapped_color_available:
            export_label = mapped_prediction
            color_lookup = mapped_prediction
        else:
            export_label = original_prediction
            color_lookup = original_prediction

        rgb = np.zeros((len(color_lookup), 3), dtype=np.float32)
        for idx, label in enumerate(color_lookup):
            color = color_map.get(int(label))
            if color is None:
                continue
            if use_bgr_color_map and len(color) >= 3:
                rgb[idx] = [color[2], color[1], color[0]]
            else:
                rgb[idx] = color[:3]

        return export_label.astype(np.int32, copy=False), rgb

    @staticmethod
    def _is_missing_laz_backend_error(exc):
        message = str(exc).lower()
        return (
            "no lazbackend selected" in message
            or "cannot decompress data" in message
            or ("laz" in message and "backend" in message)
        )

    def _warn_laz_backend_missing(self, path, exc):
        warned = getattr(self, '_laz_backend_warned', False)
        if warned:
            return
        self._laz_backend_warned = True
        print(
            "Warning: failed to read LAZ/LAS with laspy while exporting predictions.\n"
            f"Path: {path}\n"
            f"Reason: {exc}\n"
            "Falling back to available in-memory/NPZ coordinates. "
            "For exact world coordinates, install a LAZ backend:\n"
            "  pip install \"laspy[lazrs]\""
        )

    def _load_original_xyz_from_path(self, path, origin_len):
        origin_len = self._normalize_origin_len(origin_len)
        if not path:
            return None

        lower_path = str(path).lower()
        xyz = None
        if lower_path.endswith('.pth'):
            raw_data = torch.load(path)
            if isinstance(raw_data, torch.Tensor):
                raw_data = raw_data.detach().cpu().numpy()
            else:
                raw_data = np.asarray(raw_data)
            xyz = raw_data[:origin_len, :3].astype(np.float32, copy=False)
        elif lower_path.endswith('.bin'):
            raw_data = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
            xyz = raw_data[:origin_len, :3].astype(np.float32, copy=False)
        elif lower_path.endswith(('.las', '.laz')) and laspy is not None:
            try:
                las = laspy.read(path)
            except Exception as exc:
                if self._is_missing_laz_backend_error(exc):
                    self._warn_laz_backend_missing(path, exc)
                    return None
                raise
            xyz = np.stack([
                np.asarray(las.x, dtype=np.float32),
                np.asarray(las.y, dtype=np.float32),
                np.asarray(las.z, dtype=np.float32),
            ], axis=1)[:origin_len]
        elif lower_path.endswith('.npz'):
            with np.load(path, allow_pickle=True) as data:
                raw_root = data['root'].item() if 'root' in data else None
                if raw_root and os.path.abspath(str(raw_root)) != os.path.abspath(str(path)):
                    xyz = self._load_original_xyz_from_path(str(raw_root), origin_len)
                if xyz is None and 'xyz' in data:
                    xyz = data['xyz'][:origin_len, :3].astype(np.float32, copy=False)

        return xyz

    def _get_export_xyz(self, data_dict, path, origin_len):
        origin_len = self._normalize_origin_len(origin_len)
        xyz = self._load_original_xyz_from_path(path, origin_len)
        if xyz is not None and len(xyz) >= origin_len:
            return xyz[:origin_len, :3]

        for key in ('ref_xyz', 'point_xyz', 'coord'):
            value = data_dict.get(key, None)
            if torch.is_tensor(value) and value.shape[0] >= origin_len:
                return value[:origin_len, :3].detach().cpu().numpy().astype(np.float32, copy=False)
            value = self._to_numpy(value)
            if value is not None and value.ndim >= 2 and value.shape[0] >= origin_len:
                return value[:origin_len, :3].astype(np.float32, copy=False)

        return np.zeros((origin_len, 3), dtype=np.float32)

    def _load_centered_xyz_from_path(self, path, origin_len):
        if not path:
            return None

        lower_path = str(path).lower()
        xyz = None
        if lower_path.endswith('.pth'):
            raw_data = torch.load(path)
            if isinstance(raw_data, torch.Tensor):
                raw_data = raw_data.detach().cpu().numpy()
            else:
                raw_data = np.asarray(raw_data)
            xyz = raw_data[:origin_len, :3].astype(np.float32, copy=False)
        elif lower_path.endswith('.bin'):
            raw_data = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
            xyz = raw_data[:origin_len, :3].astype(np.float32, copy=False)

        if xyz is None or xyz.size == 0:
            return xyz

        if str(self.args['dataset_params'].get('pc_dataset_type', '')) == 'WCS2D3D':
            xyz = xyz - np.mean(xyz, axis=0, keepdims=True)
        return xyz

    def _extract_validation_cldice_xyz(self, data_dict):
        xyz = self._to_numpy(data_dict.get('ref_xyz', None))
        if xyz is None:
            xyz = self._to_numpy(data_dict.get('point_xyz', None))
        if xyz is None:
            xyz = self._to_numpy(data_dict.get('coord', None))
        if xyz is None:
            return None
        if xyz.ndim == 1:
            xyz = xyz.reshape(-1, 3)
        return xyz[:, :3]

    def _extract_test_cldice_xyz(self, data_dict, path, origin_len):
        xyz = self._load_centered_xyz_from_path(path, origin_len)
        if xyz is not None and len(xyz) >= int(origin_len):
            return xyz[:origin_len, :3]

        ref_xyz = self._to_numpy(data_dict.get('ref_xyz', None))
        if ref_xyz is None:
            ref_xyz = self._to_numpy(data_dict.get('point_xyz', None))
        if ref_xyz is None:
            return None
        if ref_xyz.ndim == 1:
            ref_xyz = ref_xyz.reshape(-1, 3)
        return ref_xyz[:, :3]

    def _update_cldice_metric(self, prediction, labels, xyz):
        if not getattr(self.val_cldice, 'enabled', False):
            return
        if xyz is None:
            return

        prediction = self._to_numpy(prediction)
        labels = self._to_numpy(labels)
        xyz = self._to_numpy(xyz)
        if prediction is None or labels is None or xyz is None:
            return

        prediction = prediction.reshape(-1)
        labels = labels.reshape(-1)
        if xyz.ndim == 1:
            xyz = xyz.reshape(-1, 3)

        length = min(prediction.size, labels.size, xyz.shape[0])
        if length <= 0:
            return

        self.val_cldice.update(
            prediction[:length],
            labels[:length],
            xyz[:length, :3],
        )

    def _consume_eval_diagnostic_lines(self, mIoU):
        summary = self.eval_diagnostics.compute()
        has_projected_cldice = bool(summary['cldice']) and (not getattr(self.val_cldice, 'enabled', False))
        auroc_value = summary['auroc_error_detection']
        has_auroc = np.isfinite(auroc_value)
        auroc_reason = summary.get('auroc_error_detection_reason', None)
        auroc_debug = summary.get('auroc_debug', {})

        if not has_projected_cldice and not has_auroc:
            return []

        lines = ['Diagnostic version : eval_diagnostics_v2']
        lines.append(f'mIoU : {self._format_ratio(mIoU)}')
        if has_projected_cldice:
            for display_name, score in summary['cldice'].items():
                lines.append(f'{display_name} projected clDice : {self._format_ratio(score)}')
            lines.append(f'Avg projected clDice : {self._format_ratio(summary["cldice_avg"])}')
        lines.append(f'AUROC(error detection) : {self._format_ratio(auroc_value)}')
        if not has_auroc and auroc_reason:
            lines.append(f'AUROC note : {auroc_reason}')
            lines.append(
                'AUROC debug : '
                f'uncertainty_batches={auroc_debug.get("uncertainty_batches_available", 0)}/'
                f'{auroc_debug.get("uncertainty_batches_requested", 0)}, '
                f'missing={auroc_debug.get("uncertainty_batches_missing", 0)}, '
                f'len_mismatch={auroc_debug.get("uncertainty_length_mismatch_batches", 0)}, '
                f'valid_points={auroc_debug.get("points_after_filter", 0)}, '
                f'error_points={auroc_debug.get("error_points", 0)}, '
                f'correct_points={auroc_debug.get("correct_points", 0)}'
            )
        return lines

    def training_step(self, data_dict, batch_idx):
        debug_cfg = self.args.get('train_params', {}).get('debug_profile', {})
        debug_enabled = bool(debug_cfg.get('enabled', False))
        debug_interval = int(debug_cfg.get('interval', 50))
        should_debug = debug_enabled and (self.global_step % max(debug_interval, 1) == 0)

        t_step_start = time.perf_counter()
        data_time = t_step_start - self._last_step_end_time

        if should_debug:
            data_dict['_debug_profile'] = True
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

        t_forward_start = time.perf_counter()
        data_dict = self.forward(data_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_forward_end = time.perf_counter()
        forward_time = t_forward_end - t_forward_start

        self.train_acc(data_dict['logits'].argmax(1)[data_dict['labels'] != self.ignore_label],
                       data_dict['labels'][data_dict['labels'] != self.ignore_label])
        self.log('train/acc', self.train_acc, on_epoch=True)
        self.log('train/loss_main_ce', data_dict['loss_main_ce'])
        self.log('train/loss_main_lovasz', data_dict['loss_main_lovasz'])

        if should_debug and (not hasattr(self, 'global_rank') or self.global_rank == 0):
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                mem_str = f"alloc={mem_alloc:.2f}G reserved={mem_reserved:.2f}G peak={mem_peak:.2f}G"
            else:
                mem_str = "cpu-only"

            prof = data_dict.get('_profile', {})
            prof_3d = float(prof.get('t_3d', -1.0))
            prof_2d = float(prof.get('t_2d', -1.0))
            prof_fusion = float(prof.get('t_fusion', -1.0))

            num_pts = -1
            if 'coord' in data_dict:
                coord_obj = data_dict['coord']
                if isinstance(coord_obj, torch.Tensor):
                    num_pts = int(coord_obj.shape[0])
                elif isinstance(coord_obj, (list, tuple)):
                    num_pts = int(sum(x.shape[0] for x in coord_obj if hasattr(x, 'shape')))

            num_img_pts = -1
            if 'img_indices' in data_dict:
                idx_obj = data_dict['img_indices']
                if isinstance(idx_obj, torch.Tensor):
                    num_img_pts = int(idx_obj.shape[0])
                elif isinstance(idx_obj, (list, tuple)):
                    num_img_pts = int(sum(x.shape[0] for x in idx_obj if hasattr(x, 'shape')))

            print(
                f"[PROFILE] step={self.global_step} "
                f"data={data_time:.3f}s forward={forward_time:.3f}s "
                f"(3d={prof_3d:.3f}s 2d={prof_2d:.3f}s fusion={prof_fusion:.3f}s) "
                f"pts={num_pts} img_pts={num_img_pts} {mem_str}"
            )

        self._last_step_end_time = time.perf_counter()

        return data_dict['loss']


    def validation_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        origin_len = data_dict['origin_len']
        vote_logits = torch.zeros((len(raw_labels), self.num_classes))

        # 标记当前 batch 是否需要跑 2D+融合路径以收集 PUAF 不确定性
        vis_cfg = self.args.get('train_params', {}).get('puaf_vis', None)
        if vis_cfg and vis_cfg.get('enabled', False):
            interval = vis_cfg.get('vis_interval', 10)
            max_frames = vis_cfg.get('max_frames', 8)
            if self.current_epoch % interval == 0 and batch_idx < max_frames:
                data_dict['_run_puaf_vis'] = True

        data_dict = self._prepare_eval_diagnostic_forward(data_dict)
        data_dict = self.forward(data_dict)

        if self.args['test']:
            # test 模式 batch_size=1，可以安全地用 index_add_ 映射回原始点
            vote_logits.index_add_(0, indices.cpu(), data_dict['logits'].cpu())
            if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
                vote_logits = vote_logits[:origin_len]
                raw_labels = raw_labels[:origin_len]
        else:
            # 训练时 val batch_size>1，raw_labels 只含第一个样本，
            # 无法用 index_add_ 跨 batch，直接用体素级 logits 评估
            vote_logits = data_dict['logits'].cpu()
            raw_labels = data_dict['labels'].squeeze(0).cpu()

        prediction = vote_logits.argmax(1)
        self._collect_validation_diagnostics(data_dict)
        self._update_cldice_metric(
            prediction,
            raw_labels,
            self._extract_validation_cldice_xyz(data_dict),
        )

        if self.ignore_label != 0:
            prediction = prediction[raw_labels != self.ignore_label]
            raw_labels = raw_labels[raw_labels != self.ignore_label]
            prediction += 1
            raw_labels += 1

        self.val_acc(prediction, raw_labels)
        self.log('val/acc', self.val_acc, on_epoch=True)
        self.val_iou(
            prediction.cpu().detach().numpy(),
            raw_labels.cpu().detach().numpy(),
         )

        # PUAF 不确定性可视化
        _logger = self.logger
        if _logger is not None and hasattr(_logger, '__iter__'):
            _logger = list(_logger)[0]  # LoggerCollection → 取第一个
        log_dir = getattr(_logger, 'log_dir', None) or getattr(_logger, 'save_dir', None) or '.'
        maybe_visualize_puaf(self, data_dict, batch_idx, self.current_epoch, log_dir, self.args)

        return data_dict['loss']

    def test_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        origin_len = data_dict['origin_len']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        path = data_dict['path'][0]

        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        data_dict = self._prepare_eval_diagnostic_forward(data_dict)
        data_dict = self.forward(data_dict)
        vote_logits.index_add_(0, indices.cpu(), data_dict['logits'].cpu())

        if self.args['dataset_params']['pc_dataset_type'] == 'SemanticKITTI_multiscan':
            vote_logits = vote_logits[:origin_len]
            raw_labels = raw_labels[:origin_len]

        prediction = vote_logits.argmax(1)
        self._collect_test_diagnostics(
            data_dict,
            prediction.cpu().numpy(),
            raw_labels.cpu().numpy(),
        )
        self._update_cldice_metric(
            prediction,
            raw_labels,
            self._extract_test_cldice_xyz(data_dict, path, origin_len),
        )

        if self.ignore_label != 0:
            prediction = prediction[raw_labels != self.ignore_label]
            raw_labels = raw_labels[raw_labels != self.ignore_label]
            prediction += 1
            raw_labels += 1

        if prediction.numel() > 0 and raw_labels.numel() > 0:
            self.val_acc(prediction, raw_labels)
            self.log('val/acc', self.val_acc, on_epoch=True)
            self.val_iou(
                prediction.cpu().detach().numpy(),
                raw_labels.cpu().detach().numpy(),
             )
            self.val_macc(
                prediction.cpu().detach().numpy(),
                raw_labels.cpu().detach().numpy(),
             )
            self._test_metric_batches += 1

        if self._should_save_test_predictions():
            origin_len_int = self._normalize_origin_len(origin_len)
            dataset_type = self.args['dataset_params']['pc_dataset_type']

            if dataset_type in {'WCS2D3D', 'SemanticKITTI'}:
                full_label_name = self._build_text_prediction_path(path)
                os.makedirs(os.path.dirname(full_label_name), exist_ok=True)

                if os.path.exists(full_label_name):
                    print('%s already exsist...' % (os.path.basename(full_label_name)))
                else:
                    xyz = self._get_export_xyz(data_dict, path, origin_len_int)
                    mapped_prediction = vote_logits[:origin_len_int].argmax(1).cpu().numpy().astype(np.int32)
                    export_label, rgb = self._resolve_export_labels_and_rgb(
                        mapped_prediction,
                        use_bgr_color_map=(dataset_type == 'SemanticKITTI'),
                    )
                    valid_len = min(len(xyz), len(export_label))
                    output_data = np.column_stack([xyz[:valid_len], rgb[:valid_len], export_label[:valid_len]])
                    np.savetxt(full_label_name, output_data, fmt='%.6f %.6f %.6f %.0f %.0f %.0f %d')
            elif dataset_type != 'nuScenes':
                full_label_name = self._build_text_prediction_path(path)
                os.makedirs(os.path.dirname(full_label_name), exist_ok=True)

                if os.path.exists(full_label_name):
                    print('%s already exsist...' % (os.path.basename(full_label_name)))
                else:
                    xyz = self._get_export_xyz(data_dict, path, origin_len_int)
                    mapped_prediction = vote_logits[:origin_len_int].argmax(1).cpu().numpy().astype(np.int32)
                    export_label, rgb = self._resolve_export_labels_and_rgb(mapped_prediction)
                    valid_len = min(len(xyz), len(export_label))
                    output_data = np.column_stack([xyz[:valid_len], rgb[:valid_len], export_label[:valid_len]])
                    np.savetxt(full_label_name, output_data, fmt='%.6f %.6f %.6f %.0f %.0f %.0f %d')
            else:
                meta_dict = {
                    "meta": {
                        "use_camera": False,
                        "use_lidar": True,
                        "use_map": False,
                        "use_radar": False,
                        "use_external": False,
                    }
                }
                os.makedirs(os.path.join(self.submit_dir, 'test'), exist_ok=True)
                with open(os.path.join(self.submit_dir, 'test', 'submission.json'), 'w', encoding='utf-8') as f:
                    json.dump(meta_dict, f)
                original_label = prediction.cpu().numpy().astype(np.uint8)

                assert all((original_label > 0) & (original_label < 17)), \
                    "Error: Array for predictions must be between 1 and 16 (inclusive)."

                full_save_dir = os.path.join(self.submit_dir, 'lidarseg/test')
                full_label_name = os.path.join(full_save_dir, path + '_lidarseg.bin')
                os.makedirs(full_save_dir, exist_ok=True)

                if os.path.exists(full_label_name):
                    print('%s already exsist...' % (full_label_name))
                else:
                    original_label.tofile(full_label_name)

        return data_dict['loss']

    def validation_epoch_end(self, outputs):
        iou, best_miou = self.val_iou.compute()
        finite_iou = iou[np.isfinite(iou)]
        mIoU = float(np.mean(finite_iou)) if finite_iou.size > 0 else 0.0
        str_print = ''
        self.log('val/mIoU', mIoU, on_epoch=True)
        self.log('val/best_miou', best_miou, on_epoch=True)
        str_print += 'Validation per class iou: '

        for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

        if finite_iou.size == 0:
            str_print += '\n[Warn] All class IoUs are NaN in this validation epoch. Check split/label_mapping/ignore settings.'

        str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
        if self.val_cldice.enabled:
            cldice_scores, cldice_overall, best_cldice = self.val_cldice.compute()
            self.log('val/clDice', cldice_overall, on_epoch=True)
            self.log('val/best_cldice', best_cldice, on_epoch=True)
            str_print += '\n\n' + format_cldice_report(
                self.val_cldice.class_display_names,
                cldice_scores,
                cldice_overall,
                best_cldice,
                title='Validation',
            )
        diagnostic_lines = self._consume_eval_diagnostic_lines(mIoU)
        if diagnostic_lines:
            str_print += '\n\nDiagnostic summary:'
            for line in diagnostic_lines:
                str_print += '\n' + line
        self.print(str_print)

        # 追加写入文件(如果指定了log_path参数)
        log_path = self.args.get('iou_log_path', None)
        if log_path is not None:
            # 如果是epoch 0且文件不存在或为空,先写入表头
            write_header = False
            if self.current_epoch == 0:
                if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                    write_header = True
            
            if self.trainer.global_rank == 0:
                with open(log_path, 'a', encoding='utf-8') as f:
                    # 写入表头
                    if write_header:
                        header = ""
                        for class_name in self.val_iou.unique_label_str:
                            header += f'{class_name}\t'
                        header += 'mIoU'
                        f.write(header + '\n')
                    
                    # 写入数据
                    str_print1 = ""
                    for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
                        str_print1 += '%.2f%%\t' % (class_iou * 100)
                    str_print1 += '%.3f' % (mIoU * 100)
                    f.write(str_print1 + '\n')

    def test_epoch_end(self, outputs):
        if self._test_metric_batches > 0:
            # Compute IoU
            iou, best_miou = self.val_iou.compute()
            finite_iou = iou[np.isfinite(iou)]
            mIoU = float(np.mean(finite_iou)) if finite_iou.size > 0 else 0.0
            self.log('val/mIoU', mIoU, on_epoch=True)
            self.log('val/best_miou', best_miou, on_epoch=True)
            
            # Compute mAcc
            acc, best_macc = self.val_macc.compute()
            mAcc = np.nanmean(acc)
            self.log('val/mAcc', mAcc, on_epoch=True)
            self.log('val/best_macc', best_macc, on_epoch=True)
            
            # Print IoU results
            str_print = ''
            str_print += 'Validation per class iou: '
            for class_name, class_iou in zip(self.val_iou.unique_label_str, iou):
                str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)
            if finite_iou.size == 0:
                str_print += '\n[Warn] All class IoUs are NaN in this validation epoch. Check split/label_mapping/ignore settings.'
            str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (mIoU * 100, best_miou * 100)
            
            # Print Accuracy results
            str_print += '\n\nValidation per class accuracy: '
            for class_name, class_acc in zip(self.val_macc.unique_label_str, acc):
                str_print += '\n%s : %.2f%%' % (class_name, class_acc * 100)
            str_print += '\nCurrent val mAcc is %.3f while the best val mAcc is %.3f' % (mAcc * 100, best_macc * 100)
            if self.val_cldice.enabled:
                cldice_scores, cldice_overall, best_cldice = self.val_cldice.compute()
                self.log('val/clDice', cldice_overall, on_epoch=True)
                self.log('val/best_cldice', best_cldice, on_epoch=True)
                str_print += '\n\n' + format_cldice_report(
                    self.val_cldice.class_display_names,
                    cldice_scores,
                    cldice_overall,
                    best_cldice,
                    title='Validation',
                )
            diagnostic_lines = self._consume_eval_diagnostic_lines(mIoU)
            if diagnostic_lines:
                str_print += '\n\nDiagnostic summary:'
                for line in diagnostic_lines:
                    str_print += '\n' + line
            
            self.print(str_print)
        else:
            self.print('No valid labels found on evaluated test split; skipped metric aggregation.')

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
