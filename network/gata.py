from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class GATAModule(nn.Module):
    """Geo-Aware Texture Adaptation (GATA) Module — G3D geometric stream.

    The geometric stream now takes a **pre-computed dense geometry map** ``g3d``
    (e.g. rasterised principal-direction / curvature / normal fields from the
    point cloud) instead of the former Z-map + Sobel gradient approximation.

    The texture stream still uses ``DeformConv2d`` whose offsets are predicted
    from the geometric features.

    Optional **offset distillation** during training: when ``teacher_offsets``
    is provided the module returns an extra L1 loss ``l_off``.
    """

    def __init__(
        self,
        g3d_channels: int,
        out_channels: int = 32,
        geo_hidden: int = 16,
        geo_out: int = 32,
        mode: str = 'full',
    ) -> None:
        """
        Args:
            mode: 融合模式
                'full'         — 完整 GATA（G3D 预测偏移，支持 offset distillation）
                'standard_dcn' — 消融：偏移由 RGB 自身预测，不依赖 G3D 几何特征
        """
        super().__init__()
        self.g3d_channels = g3d_channels
        self.out_channels = out_channels
        self.geo_out = geo_out
        self.mode = mode

        # --- 1. Geometric stream (G3D) ---
        self.geo_encoder = nn.Sequential(
            nn.Conv2d(g3d_channels, geo_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(geo_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(geo_hidden, geo_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(geo_out),
            nn.ReLU(inplace=True),
        )

        # --- 2. Offset generator ---
        # 3×3 kernel → 9 sampling points → 18 offsets (dy, dx per point)
        # 'full': offsets ← geo_feat  |  'standard_dcn': offsets ← RGB feature
        self.offset_conv = nn.Conv2d(geo_out, 18, kernel_size=3, padding=1)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        if mode == 'standard_dcn':
            # 消融：用一个轻量 RGB 编码器替代几何特征预测偏移
            self.rgb_pre = nn.Sequential(
                nn.Conv2d(3, geo_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(geo_out),
                nn.ReLU(inplace=True),
            )
            self.offset_conv_rgb = nn.Conv2d(geo_out, 18, kernel_size=3, padding=1)
            nn.init.constant_(self.offset_conv_rgb.weight, 0)
            nn.init.constant_(self.offset_conv_rgb.bias, 0)

        # --- 3. Texture stream (DeformConv2d on RGB) ---
        self.dcn = DeformConv2d(3, out_channels, kernel_size=3, padding=1)

        # --- 4. Fusion layer ---
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + geo_out, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        rgb: torch.Tensor,
        g3d: torch.Tensor,
        teacher_offsets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            rgb:             (B, 3, H, W)  — RGB image.
            g3d:             (B, dg, H, W) — dense geometry map (G3D).
            teacher_offsets: (B, 18, H, W) — optional teacher offsets for
                             distillation.  Ignored at inference / when *None*.

        Returns:
            * If ``teacher_offsets is None``: ``out``  (B, out_channels, H, W)
            * Otherwise: ``{'out': out, 'l_off': l_off, 'offsets': offsets}``
        """
        # --- sanity checks ---
        assert rgb.shape[1] == 3, (
            f"rgb must have 3 channels, got {rgb.shape[1]}"
        )
        assert g3d.shape[2:] == rgb.shape[2:], (
            f"Spatial dims mismatch: g3d {g3d.shape[2:]} vs rgb {rgb.shape[2:]}"
        )

        # A. Geometric feature extraction from G3D
        geo_feat = self.geo_encoder(g3d.contiguous())

        # B. Predict deformable offsets
        #    'full'         → offsets from geometric features (G3D)
        #    'standard_dcn' → offsets from RGB features (ablation: no geo guidance)
        if self.mode == 'standard_dcn':
            rgb_pre_feat = self.rgb_pre(rgb.contiguous())
            offsets = self.offset_conv_rgb(rgb_pre_feat)
        else:
            offsets = self.offset_conv(geo_feat)
        assert offsets.shape[1] == 18, (
            f"offsets must have 18 channels, got {offsets.shape[1]}"
        )

        # C. Texture-adaptive feature extraction via deformable conv
        rgb_feat = self.dcn(rgb.contiguous(), offsets)

        # D. Fuse geometric + texture features
        out = self.fusion(torch.cat([rgb_feat, geo_feat], dim=1))

        # E. Optional offset distillation loss
        # 仅在 'full' 模式下应用（standard_dcn 偏移来自 RGB，与 3D teacher 对齐无意义）
        if teacher_offsets is not None and self.mode != 'standard_dcn':
            assert teacher_offsets.shape == offsets.shape, (
                f"teacher_offsets shape must match offsets: "
                f"teacher_offsets={tuple(teacher_offsets.shape)} vs offsets={tuple(offsets.shape)}"
            )
            teacher_offsets = teacher_offsets.detach()
            l_off = F.l1_loss(offsets, teacher_offsets)
            return {"out": out, "l_off": l_off, "offsets": offsets}

        return out
