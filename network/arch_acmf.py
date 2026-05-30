import torch
import torch_scatter
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.SPVCNN_LGC import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.fast_scnn import FastSCNN as FastSCNN_4
from network.basic_block import ResNetFCN
from network.puaf import (
    ACMF_PointAdapter,
)
from utils.puaf_vis import instrument_ugaf_modules
from network.gata import GATAModule
from network.cross_mamba import CrossMambaInteraction


class xModalKD(nn.Module):
    def __init__(self, config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)
        self.cmf_geo_mode = 'coord'

        # 根据backbone类型设置各尺度输出通道数
        backbone_2d = config['model_params'].get('backbone_2d', 'FastSCNN')
        if backbone_2d == 'FastSCNN':
            self.img_feat_dims = [32, 48, 64, 128]
        else:
            # ResNet等backbone输出通道数均为hiden_size
            self.img_feat_dims = [self.hiden_size] * self.num_scales

        self.img_proj = nn.ModuleList([
            nn.Linear(self.img_feat_dims[i], self.hiden_size) for i in range(self.num_scales)
        ])

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        # ===== PUAF / ACMF epistemic 超参 =====
        self.puaf_mc_samples = config['model_params'].get('puaf_mc_samples', 4)
        self.puaf_dropout_p = config['model_params'].get('puaf_dropout_p', 0.2)
        self.puaf_lambda_e = config['model_params'].get('puaf_lambda_e', 1.0)
        self.puaf_enable_epistemic = config['model_params'].get('puaf_enable_epistemic', True)
        self.puaf_epistemic_in_eval = config['model_params'].get('puaf_epistemic_in_eval', False)

        # ===== 跨模态融合模块 =====
        self.cmf4_only = (self.fusion_type == 'cmf4_only')
        self.fusion_modules = nn.ModuleList()
        if self.cmf4_only:
            fusion_base = None  # no fusion module; cur_ctx is used directly
        elif self.use_cmf4:
            fusion_base = 'acmf'
        else:
            fusion_base = self.fusion_type
        for i in range(self.num_scales):
            if fusion_base is None:
                pass  # cmf4_only: no fusion module needed
            elif fusion_base == 'acmf':
                self.fusion_modules.append(
                    ACMF_PointAdapter(
                        dim=self.hiden_size,
                        mc_samples=self.puaf_mc_samples,
                        dropout_p=self.puaf_dropout_p,
                        lambda_e=self.puaf_lambda_e,
                        enable_epistemic=self.puaf_enable_epistemic,
                        epistemic_in_eval=self.puaf_epistemic_in_eval,
                    )
                )
            elif fusion_base == 'concat':
                self.fusion_modules.append(
                    ConcatFusion(dim=self.hiden_size)
                )
            elif fusion_base == 'add':
                self.fusion_modules.append(
                    AddFusion(dim=self.hiden_size)
                )
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        if self.use_cmf4:
            cmf_bits = config['model_params'].get('cmf_bits', 10)
            cmf_d_state = config['model_params'].get('cmf_d_state', 16)
            cmf_d_conv = config['model_params'].get('cmf_d_conv', 4)
            cmf_expand = config['model_params'].get('cmf_expand', 2)
            cmf_geo_scale_mode = config['model_params'].get('cmf_geo_scale_mode', 'residual')
            # ---- Ablation switches (defaults = full model) ----
            cmf_cross_mode = config['model_params'].get('cmf_cross_mode', 'cross_parameter')
            cmf_alpha_mode = config['model_params'].get('cmf_alpha_mode', 'gr_mlp')
            self.cmf_use_ctx = config['model_params'].get('cmf_use_ctx', True)
            cmf_use_hilbert = config['model_params'].get('cmf_use_hilbert', True)
            self.cmf_blocks = nn.ModuleList([
                CrossMambaInteraction(
                    dim=self.hiden_size,
                    d_state=cmf_d_state,
                    d_conv=cmf_d_conv,
                    expand=cmf_expand,
                    hilbert_bits=cmf_bits,
                    geo_scale_mode=cmf_geo_scale_mode,
                    cmf_cross_mode=cmf_cross_mode,
                    cmf_alpha_mode=cmf_alpha_mode,
                    cmf_use_ctx=self.cmf_use_ctx,
                    cmf_use_hilbert=cmf_use_hilbert,
                ) for _ in range(self.num_scales)
            ])
            self.ctx_inject = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hiden_size, self.hiden_size),
                    nn.LayerNorm(self.hiden_size),
                    nn.ReLU(True),
                ) for _ in range(self.num_scales)
            ])

            print(
                f"CMF4 enabled | uncertainty_gated={self.cmf_use_uncertainty_gated} | "
                f"geo_conditioned={self.cmf_use_geo_conditioned} | geo_mode={self.cmf_geo_mode} | "
                f"cross_mode={cmf_cross_mode} | alpha_mode={cmf_alpha_mode} | "
                f"use_ctx={self.cmf_use_ctx} | use_hilbert={cmf_use_hilbert}"
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights,
                                           ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max() + 1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    @staticmethod
    def _normalize_geo_value(x):
        x = x.float()
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + 1e-6)

    def _build_geo_cond(self, data_dict, coords, coords_batch):
        z = coords[:, 2:3].float()
        return self._normalize_geo_value(z).clamp(0.0, 1.0)

    def fusion_to_single_KD(self, data_dict, idx, prev_ctx=None):
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']
        last_scale = self.scale_list[idx - 1] if idx > 0 else 1
        img_feat = data_dict['img_scale{}'.format(self.scale_list[idx])]
        pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        coors_inv = data_dict['scale_{}'.format(last_scale)]['coors_inv']

        # 将图像特征投影到统一维度
        img_feat = self.img_proj[idx](img_feat)

        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

        # correspondence
        pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
        pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx)
        pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)

        full_coors = data_dict['layer_{}'.format(idx)]['full_coors']
        full_coors = self.p2img_mapping(full_coors[coors_inv], point2img_index, batch_idx)
        coords = full_coors[:, 1:4]
        coords_batch = full_coors[:, 0].long()

        puaf_module = self.fusion_modules[idx].ugaf
        u_img_a = puaf_module.uncertainty_img(img_feat)
        u_pts_a = puaf_module.uncertainty_pts(pts_feat)
        if puaf_module.enable_epistemic and self.training:
            u_img_e = puaf_module._mc_epistemic(puaf_module.uncertainty_img, img_feat)
            u_pts_e = puaf_module._mc_epistemic(puaf_module.uncertainty_pts, pts_feat)
            u_img = u_img_a + puaf_module.lambda_e * u_img_e
            u_pts = u_pts_a + puaf_module.lambda_e * u_pts_e
        else:
            u_img = u_img_a
            u_pts = u_pts_a
        reliability = torch.exp(-torch.min(u_img, u_pts)).detach()

        geo_cond = self._build_geo_cond(data_dict, coords, coords_batch)
        if geo_cond.dim() == 1:
            geo_cond = geo_cond.unsqueeze(-1)
        if geo_cond.shape[-1] != 1:
            geo_cond = geo_cond[:, :1]

        ctx_in = self.ctx_inject[idx](prev_ctx) if prev_ctx is not None else None
        img_feat, pts_feat, cur_ctx = self.cmf_blocks[idx](
            img_feat,
            pts_feat,
            coords,
            batch_idx=coords_batch,
            reliability=reliability,
            geo_cond=geo_cond,
            ctx=ctx_in,
        )

        fuse_feat = self.fusion_modules[idx](img_feat, pts_feat)

        # fusion prediction
        fuse_pred = self.multihead_fuse_classifier[idx](fuse_feat)

        # Segmentation Loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
        seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
        loss = seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

        # 原始KL散度蒸馏
        xm_loss = F.kl_div(
            F.log_softmax(pts_pred, dim=1),
            F.softmax(fuse_pred.detach(), dim=1),
        )
        loss += xm_loss * self.lambda_xm / self.num_scales

        return loss, fuse_feat, cur_ctx

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []
        prev_ctx = None

        for idx in range(self.num_scales):
            singlescale_loss, fuse_feat, cur_ctx = self.fusion_to_single_KD(
                data_dict, idx, prev_ctx
            )
            img_seg_feat.append(fuse_feat)
            loss += singlescale_loss
            prev_ctx = cur_ctx

        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
        data_dict['loss'] += loss

        return data_dict


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.lambda_off = config.train_params.get('lambda_off', 1.0)
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)
        self.cmf_geo_mode = 'coord'

        self.model_3d = SPVCNN(config)
        g3d_channels = config.model_params.get('g3d_channels', 7)
        gata_out = config.model_params.get('gata_out_channels', 32)
        self.gata = GATAModule(
            g3d_channels=g3d_channels,
            out_channels=gata_out,
        )
        if config.model_params.backbone_2d == 'FastSCNN':
            self.model_2d = FastSCNN_4(
                num_classes=config.model_params.num_classes,
                in_channels=gata_out
            )
        else:
            self.model_2d = ResNetFCN(
                backbone=config.model_params.backbone_2d,
                pretrained=config.model_params.pretrained2d,
                config=config,
                in_channels=gata_out
            )
        self.fusion = xModalKD(config)
        instrument_ugaf_modules(self)

    def forward(self, data_dict, return_uncer=False):
        profile_enabled = bool(data_dict.get('_debug_profile', False))
        if profile_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter() if profile_enabled else None

        # 3D network
        data_dict = self.model_3d(data_dict)
        if profile_enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        # training with 2D network (also run during val when PUAF vis is requested)
        _run_2d = self.training or data_dict.get('_run_puaf_vis', False)
        if _run_2d:
            # make sure loss exists for later accumulation (xModalKD expects it)
            data_dict['loss'] = data_dict.get('loss', 0)
            img = data_dict['img']

            # GATA 几何-纹理显式交互（新接口: rgb + g3d 分离输入）
            rgb = img[:, :3]
            if 'g3d' not in data_dict:
                raise RuntimeError("GATA requires data_dict['g3d'].")
            g3d = data_dict['g3d']
            teacher_offsets = data_dict.get('teacher_offsets', None)
            gata_result = self.gata(rgb, g3d, teacher_offsets=teacher_offsets)
            if isinstance(gata_result, dict):
                img = gata_result['out']
                if 'l_off' in gata_result:
                    data_dict['loss'] = data_dict['loss'] + gata_result['l_off'] * self.lambda_off
            else:
                img = gata_result

            outputs = self.model_2d(img)
            if profile_enabled:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
            img_indices = data_dict['img_indices']

            # 兼容多尺度特征，假设 FastSCNN 返回 (scale2, scale4, scale8, scale16)
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 4:
                # 对每个尺度的特征提取点对应的特征
                scale_features = [outputs[0], outputs[1], outputs[2], outputs[3]]
                scale_names = ['img_scale2', 'img_scale4', 'img_scale8', 'img_scale16']

                for scale_feat, scale_name in zip(scale_features, scale_names):
                    # 提取点特征: (B, C, H, W) -> (N, C)
                    point_features = []
                    for i in range(scale_feat.shape[0]):
                        # [C, H, W] -> [H, W, C]
                        feat = scale_feat[i].permute(1, 2, 0)
                        idx = img_indices[i]  # [N_i, 2]
                        # 按索引采样
                        sampled = feat[idx[:, 0], idx[:, 1]]  # [N_i, C]
                        point_features.append(sampled)
                    data_dict[scale_name] = torch.cat(point_features, 0)
            else:
                # 只返回主输出时，全部写入 img_scale16
                output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                point_features = []
                for i in range(output.shape[0]):
                    feat = output[i].permute(1, 2, 0)
                    idx = img_indices[i]
                    sampled = feat[idx[:, 0], idx[:, 1]]
                    point_features.append(sampled)
                data_dict['img_scale16'] = torch.cat(point_features, 0)

            data_dict = self.fusion(data_dict)
            if profile_enabled:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                data_dict['_profile'] = {
                    't_3d': float(t1 - t0),
                    't_2d': float(t2 - t1),
                    't_fusion': float(t3 - t2),
                }
        elif profile_enabled:
            t_end = time.perf_counter()
            data_dict['_profile'] = {
                't_3d': float(t_end - t0),
                't_2d': 0.0,
                't_fusion': 0.0,
            }

        return data_dict
