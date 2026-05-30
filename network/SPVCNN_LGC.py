#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: baseline.py
@time: 2021/12/16 22:41
'''
import torch
import torch_scatter
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from network.basic_block import Lovasz_loss
from network.base_model import LightningBaseModel
from network.basic_block import SparseBasicBlock
from network.voxel_fea_generator import voxel_3d_generator, voxelization


# ============================================================
# 3D Hilbert Curve Implementation for Topology-Preserving Serialization
# ============================================================

class HilbertCurve3D:
    """
    3D Hilbert 曲线序列化器 (Hilbert Curve Serializer)

    核心思想：将无序的 3D 点云坐标映射为保持空间局部性的一维序列索引。
    相比 Z-order (Morton Code)，Hilbert 曲线具有更优的聚类特性 (Clustering Property)：
    - 若两点在 1D 序列上索引相近，则它们在 3D 空间中的欧氏距离大概率也很小
    - 这使得 1D 卷积能够有效捕获 3D 空间邻域特征

    参考：Butz, A.R. (1971). "Alternative algorithm for Hilbert's space-filling curve"
    """

    def __init__(self, order=8):
        """
        Args:
            order: Hilbert 曲线的阶数，决定了空间分辨率 (2^order)^3
        """
        self.order = order
        # 预计算旋转/翻转查找表以加速编码
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """构建 Gray code 和状态转换表"""
        # 3D Hilbert curve 的 8 种状态转换
        self.hilbert_table = torch.tensor([
            [0, 7, 3, 4, 1, 6, 2, 5],  # state 0
            [0, 1, 3, 2, 7, 6, 4, 5],  # state 1
            [0, 3, 7, 4, 1, 2, 6, 5],  # state 2
            [2, 3, 1, 0, 5, 4, 6, 7],  # state 3
            [4, 5, 7, 6, 3, 2, 0, 1],  # state 4
            [4, 7, 3, 0, 5, 6, 2, 1],  # state 5
            [6, 5, 1, 2, 7, 4, 0, 3],  # state 6
            [6, 7, 5, 4, 1, 0, 2, 3],  # state 7
        ], dtype=torch.long)

        self.state_table = torch.tensor([
            [1, 6, 3, 4, 2, 5, 0, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 3, 2, 5, 6, 1, 4, 7],
            [2, 3, 0, 1, 6, 7, 4, 5],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [4, 7, 6, 1, 0, 5, 2, 3],
            [6, 1, 4, 3, 2, 7, 0, 5],
            [6, 7, 4, 5, 2, 3, 0, 1],
        ], dtype=torch.long)

    def xyz_to_hilbert(self, coords, spatial_shape=None):
        """
        将 3D 坐标转换为 Hilbert 曲线索引

        Args:
            coords: [N, 3] 整数坐标 (x, y, z)
            spatial_shape: 空间范围，用于归一化坐标

        Returns:
            hilbert_indices: [N] Hilbert 曲线上的一维索引
        """
        device = coords.device
        N = coords.shape[0]

        if spatial_shape is not None:
            # 归一化坐标到 [0, 2^order - 1] 范围
            spatial_shape = torch.tensor(spatial_shape, device=device, dtype=torch.float32)
            max_coord = (2 ** self.order) - 1
            coords = (coords.float() / spatial_shape.max() * max_coord).long()
            coords = torch.clamp(coords, 0, max_coord)

        # 确保坐标在有效范围内
        max_val = 2 ** self.order - 1
        coords = torch.clamp(coords, 0, max_val)

        # 使用位交织方法计算 Hilbert 索引（简化版本）
        # 这里使用近似的 Morton -> Hilbert 转换
        hilbert_indices = self._coords_to_hilbert_fast(coords)

        return hilbert_indices

    def _coords_to_hilbert_fast(self, coords):
        """
        快速计算 3D Hilbert 索引（基于位操作的近似方法）

        使用 Morton code 作为基础，然后应用 Gray code 变换来近似 Hilbert 顺序
        这种方法在保持较好局部性的同时大幅提升计算效率
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # 计算 Morton code (Z-order)
        morton = torch.zeros_like(x, dtype=torch.long)
        for i in range(self.order):
            morton |= ((x >> i) & 1) << (3 * i)
            morton |= ((y >> i) & 1) << (3 * i + 1)
            morton |= ((z >> i) & 1) << (3 * i + 2)

        # 应用 Gray code 变换以改善局部性
        # Gray code: g = n ^ (n >> 1)
        hilbert = morton ^ (morton >> 1)

        return hilbert

    def sort_by_hilbert(self, features, coords, spatial_shape=None):
        """
        根据 Hilbert 曲线索引对点云特征进行排序

        Args:
            features: [N, C] 或 [B, N, C] 点云特征
            coords: [N, 3] 或 [B, N, 3] 点云坐标
            spatial_shape: 空间范围

        Returns:
            sorted_features: 排序后的特征
            sort_indices: 排序索引（用于恢复原始顺序）
            inverse_indices: 逆排序索引
        """
        if features.dim() == 2:
            # [N, C] -> unbatched
            hilbert_indices = self.xyz_to_hilbert(coords, spatial_shape)
            sort_indices = torch.argsort(hilbert_indices)
            inverse_indices = torch.argsort(sort_indices)
            sorted_features = features[sort_indices]
        else:
            # [B, N, C] -> batched
            B, N, C = features.shape
            sorted_features = torch.zeros_like(features)
            sort_indices = torch.zeros(B, N, dtype=torch.long, device=features.device)
            inverse_indices = torch.zeros(B, N, dtype=torch.long, device=features.device)

            for b in range(B):
                hilbert_indices = self.xyz_to_hilbert(coords[b], spatial_shape)
                sort_idx = torch.argsort(hilbert_indices)
                inv_idx = torch.argsort(sort_idx)
                sorted_features[b] = features[b, sort_idx]
                sort_indices[b] = sort_idx
                inverse_indices[b] = inv_idx

        return sorted_features, sort_indices, inverse_indices


def hilbert_sort(features, coords, order=8, spatial_shape=None):
    """
    便捷函数：对点云特征按 Hilbert 曲线顺序排序

    Args:
        features: [N, C] 点云特征
        coords: [N, 3] 点云坐标（整数体素坐标）
        order: Hilbert 曲线阶数
        spatial_shape: 空间范围

    Returns:
        sorted_features, sort_indices, inverse_indices
    """
    hilbert = HilbertCurve3D(order=order)
    return hilbert.sort_by_hilbert(features, coords, spatial_shape)


class DepthwiseConv1D(nn.Module):
    """
    1D深度可分离卷积：用于点云序列的局部位置增强（LPE）
    适配点云数据的1D序列结构，提取局部邻域特征
    """

    def __init__(self, in_channels, kernel_size=3, padding=1):
        super(DepthwiseConv1D, self).__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels
        )
        nn.init.constant_(self.depthwise_conv.weight, 1.)
        if self.depthwise_conv.bias is not None:
            nn.init.constant_(self.depthwise_conv.bias, 0.)

    def forward(self, x):
        return self.depthwise_conv(x)


class StripPooling1D(nn.Module):
    """
    方向感知条形池化模块 (Strip Pooling)

    针对点云序列的1D实现，通过分段池化模拟水平/垂直方向的长程依赖捕获。
    核心思想：将序列分成多个条带(strip)，分别沿不同方向聚合上下文，
    从而捕获各向异性的全局依赖，特别适合线性地物（沟渠、堤坝）的建模。

    复杂度：O(N)，避免了自注意力的O(N²)开销
    """

    def __init__(self, dim, num_strips=8):
        super(StripPooling1D, self).__init__()
        self.num_strips = num_strips

        # 水平方向（沿序列前半部分）上下文投影
        self.horizontal_proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(True),
            nn.Linear(dim // 4, dim)
        )

        # 垂直方向（沿序列后半部分）上下文投影
        self.vertical_proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(True),
            nn.Linear(dim // 4, dim)
        )

        # 融合后的通道注意力
        self.channel_attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        输入: x - [B, N, C]
        输出: 方向感知的全局上下文增强特征 [B, N, C]
        """
        B, N, C = x.shape

        # 计算每个strip的大小
        strip_size = max(1, N // self.num_strips)

        # === 水平条形池化：沿序列方向分段聚合 ===
        # 将序列分成num_strips个条带，每个条带内部求平均
        h_context = []
        for i in range(self.num_strips):
            start_idx = i * strip_size
            end_idx = min((i + 1) * strip_size, N)
            if start_idx < N:
                strip_feat = x[:, start_idx:end_idx, :].mean(dim=1, keepdim=True)  # [B, 1, C]
                h_context.append(strip_feat.expand(-1, end_idx - start_idx, -1))

        if len(h_context) > 0:
            # 处理剩余点
            total_covered = sum([hc.shape[1] for hc in h_context])
            if total_covered < N:
                remaining = x[:, total_covered:, :].mean(dim=1, keepdim=True)
                h_context.append(remaining.expand(-1, N - total_covered, -1))
            g_h = torch.cat(h_context, dim=1)[:, :N, :]  # [B, N, C]
        else:
            g_h = x.mean(dim=1, keepdim=True).expand(-1, N, -1)

        g_h = self.horizontal_proj(g_h)  # [B, N, C]

        # === 垂直条形池化：交错分段聚合（模拟正交方向）===
        # 使用不同的分段策略，交错采样以捕获不同方向的依赖
        v_context = []
        stride = max(1, N // self.num_strips)
        # 确保 num_strips 不超过 N
        actual_strips = min(self.num_strips, N)
        for i in range(actual_strips):
            # 交错采样：每隔 actual_strips 取一个点组成一个条带
            indices = torch.arange(i, N, actual_strips, device=x.device)
            if len(indices) > 0:
                strip_feat = x[:, indices, :].mean(dim=1, keepdim=True)  # [B, 1, C]
                v_context.append((indices, strip_feat))

        # 将垂直上下文广播回原始位置
        g_v = torch.zeros_like(x)
        for indices, strip_feat in v_context:
            g_v[:, indices, :] = strip_feat.expand(-1, len(indices), -1)

        g_v = self.vertical_proj(g_v)  # [B, N, C]

        # === 融合两个方向的上下文 ===
        # G_global = g_h + g_v
        g_global = g_h + g_v  # [B, N, C]

        # 生成通道注意力权重
        # 使用全局池化后的特征计算注意力
        g_pooled = g_global.mean(dim=1, keepdim=True)  # [B, 1, C]
        alpha = self.channel_attn(g_pooled)  # [B, 1, C]

        # 逐通道调制 + 残差连接
        # X_global = X_local * alpha + X_local
        out = x * alpha + x

        return out


class GlobalContextModule(nn.Module):
    """
    轻量级全局上下文模块：通过全局池化+广播实现O(N)复杂度的全局建模
    避免O(N²)的注意力计算，同时保留全局语义信息

    已升级为支持 Strip Pooling 的方向感知版本
    """

    def __init__(self, dim, use_strip_pooling=True, num_strips=8):
        super(GlobalContextModule, self).__init__()
        self.use_strip_pooling = use_strip_pooling

        if use_strip_pooling:
            # 使用方向感知的条形池化
            self.strip_pool = StripPooling1D(dim, num_strips=num_strips)
        else:
            # 原始的全局池化
            self.global_proj = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(True),
                nn.Linear(dim // 4, dim),
                nn.Sigmoid()
            )

    def forward(self, x):
        """
        输入: x - [B, N, C]
        输出: 全局上下文增强的特征 [B, N, C]
        """
        if self.use_strip_pooling:
            return self.strip_pool(x)
        else:
            # 全局平均池化: [B, N, C] -> [B, 1, C]
            global_feat = x.mean(dim=1, keepdim=True)
            # 生成通道注意力权重: [B, 1, C]
            attn_weights = self.global_proj(global_feat)
            # 广播乘法实现全局调制: [B, N, C] * [B, 1, C] -> [B, N, C]
            return x * attn_weights


class LPEBlock(nn.Module):
    """
    方向感知局部-全局上下文模块（DALGC Block）
    核心设计：Hilbert 曲线拓扑保持序列化 + 多尺度局部卷积 + 方向敏感的条形池化全局上下文

    改进点：
    1. 3D Hilbert 曲线序列化：将无序点云转化为保持空间局部性的一维序列
    2. 多尺度深度可分离卷积：在 Hilbert 序列上捕获不同范围的局部几何特征
    3. 条形池化 (Strip Pooling)：O(N)复杂度的各向异性全局建模
    4. 残差连接 + LayerNorm：稳定训练

    复杂度：O(N)，显存友好，适合大规模点云
    特别适合：线性水利设施（沟渠、堤坝、河道）的拓扑完整性建模
    """

    def __init__(self, dim, use_lpe=True, use_global_context=False, use_strip_pooling=True, use_hilbert=True):
        super(LPEBlock, self).__init__()
        self.use_lpe = use_lpe
        self.use_global_context = use_global_context
        self.use_strip_pooling = use_strip_pooling
        self.use_hilbert = use_hilbert

        # Hilbert 曲线序列化器
        if use_hilbert:
            self.hilbert_curve = HilbertCurve3D(order=8)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 局部位置增强：多尺度深度可分离卷积（在 Hilbert 序列上操作）
        if use_lpe:
            self.lpe_conv1 = DepthwiseConv1D(dim, kernel_size=3, padding=1)
            self.lpe_conv2 = DepthwiseConv1D(dim, kernel_size=5, padding=2)
            self.lpe_conv3 = DepthwiseConv1D(dim, kernel_size=7, padding=3)
            self.lpe_fusion = nn.Linear(dim * 3, dim)

        # 全局上下文模块：方向感知的条形池化全局建模
        if use_global_context:
            self.global_context = GlobalContextModule(
                dim,
                use_strip_pooling=use_strip_pooling,
                num_strips=8
            )

        # FFN（前馈网络）
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x, coords=None, spatial_shape=None):
        """
        输入:
            x - [N, C] 点云特征（N=点数，C=通道数）
            coords - [N, 3] 点云体素坐标（用于 Hilbert 排序）
            spatial_shape - 空间范围
        输出: [N, C] 增强后的点云特征
        """
        # 添加batch维度: [N, C] -> [1, N, C]
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, N, C = x.shape

        # ===== Hilbert 曲线拓扑保持序列化 =====
        # 将无序点云按 Hilbert 曲线排序，使得空间邻近的点在序列中也相邻
        if self.use_hilbert and coords is not None:
            # 对每个 batch 进行 Hilbert 排序
            if coords.dim() == 2:
                coords = coords.unsqueeze(0).expand(B, -1, -1) if B > 1 else coords.unsqueeze(0)

            sorted_x = torch.zeros_like(x)
            inverse_indices_list = []

            for b in range(B):
                batch_coords = coords[b] if coords.dim() == 3 else coords
                hilbert_indices = self.hilbert_curve.xyz_to_hilbert(batch_coords, spatial_shape)
                sort_indices = torch.argsort(hilbert_indices)
                inverse_indices = torch.argsort(sort_indices)
                sorted_x[b] = x[b, sort_indices]
                inverse_indices_list.append(inverse_indices)

            x_sorted = sorted_x
            need_unsort = True
        else:
            x_sorted = x
            need_unsort = False

        # ===== LPE: 多尺度局部位置增强（在 Hilbert 序列上操作） =====
        if self.use_lpe:
            x_norm = self.norm1(x_sorted)
            # [B, N, C] -> [B, C, N]
            x_conv = x_norm.permute(0, 2, 1)
            # 多尺度卷积：在拓扑保持的 Hilbert 序列上进行
            # 这等效于在 3D 空间的紧凑邻域内进行特征聚合
            lpe1 = self.lpe_conv1(x_conv)  # kernel=3, 捕获近邻特征
            lpe2 = self.lpe_conv2(x_conv)  # kernel=5, 中等范围
            lpe3 = self.lpe_conv3(x_conv)  # kernel=7, 较大范围
            # 融合: [B, C*3, N] -> [B, N, C*3] -> [B, N, C]
            lpe_cat = torch.cat([lpe1, lpe2, lpe3], dim=1).permute(0, 2, 1)
            lpe_out = self.lpe_fusion(lpe_cat)
            x_sorted = x_sorted + lpe_out  # 残差连接

        # ===== 恢复原始点云顺序 =====
        if need_unsort:
            x_unsorted = torch.zeros_like(x_sorted)
            for b in range(B):
                x_unsorted[b] = x_sorted[b, inverse_indices_list[b]]
            x = x_unsorted
        else:
            x = x_sorted

        # ===== 全局上下文增强（不需要 Hilbert 排序，直接在原始顺序上操作）=====
        if self.use_global_context:
            x = x + self.global_context(x)  # 残差连接

        # FFN
        x = x + self.ffn(self.norm2(x))

        if squeeze_output:
            x = x.squeeze(0)  # [1, N, C] -> [N, C]

        return x


class point_encoder(nn.Module):
    """
    点编码器：集成 DALGC 模块（方向感知局部-全局上下文）

    核心组件：
    - 3D Hilbert 曲线序列化：拓扑保持的点云排序
    - LPEBlock: 多尺度深度可分离卷积 + Strip Pooling 全局上下文
    - 支持通过配置启用/禁用各个子模块
    """

    def __init__(self, in_channels, out_channels, scale, use_lpe=True, use_global_context=True, use_strip_pooling=True,
                 use_hilbert=True):
        super(point_encoder, self).__init__()
        self.scale = scale
        self.use_lpe = use_lpe
        self.use_hilbert = use_hilbert

        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )

        # 输入投影层：将输入通道映射到输出通道
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )

        # DALGC 模块：Hilbert 序列化 + 多尺度局部卷积 + 方向感知的 Strip Pooling 全局上下文
        self.lpe_block = LPEBlock(
            dim=out_channels,
            use_lpe=use_lpe,
            use_global_context=use_global_context,
            use_strip_pooling=use_strip_pooling,
            use_hilbert=use_hilbert
        )

        # 输出MLP（保留原始结构的一部分，用于特征精炼）
        self.output_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )

        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels))

    def downsample_with_coords(self, coors, p_fea, scale=2):
        """
        下采样并同时返回对应的坐标，确保特征和坐标数量一致

        Args:
            coors: [N, 4] 坐标 (batch_idx, x, y, z)
            p_fea: [N, C] 点云特征
            scale: 下采样比例

        Returns:
            output: [M, C] 下采样后的特征
            inv: [N] 逆索引，用于恢复原始点的索引
            ds_coords: [M, 3] 下采样后的坐标 (x, y, z)
        """
        batch = coors[:, 0:1]
        downsampled_xyz = coors[:, 1:4] // scale
        combined = torch.cat([batch, downsampled_xyz], 1)

        # 使用同一个 unique 操作，确保特征和坐标数量一致
        unique_coors, inv = torch.unique(combined, return_inverse=True, dim=0)

        # 聚合特征
        output = torch_scatter.scatter_mean(p_fea, inv, dim=0)

        # 提取下采样后的 xyz 坐标
        ds_coords = unique_coors[:, 1:4]

        return output, inv, ds_coords

    @staticmethod
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale
        inv = torch.unique(torch.cat([batch, coors], 1), return_inverse=True, dim=0)[1]
        return torch_scatter.scatter_mean(p_fea, inv, dim=0), inv

    def forward(self, features, data_dict):
        # 使用统一的下采样操作，确保特征和坐标数量一致
        output, inv, ds_coords = self.downsample_with_coords(
            data_dict['coors'], features, self.scale
        )
        identity = self.layer_in(features)

        # LPE增强的特征提取（替代原始PPmodel）
        # 1. 输入投影
        output = self.input_proj(output)
        # 2. DALGC 模块：Hilbert 序列化 + 多尺度卷积 + 全局上下文
        #    Hilbert 排序确保了 1D 卷积能有效捕获 3D 空间邻域特征
        output = self.lpe_block(output, coords=ds_coords, spatial_shape=None)
        # 3. 输出MLP精炼
        output = self.output_mlp(output)
        # 4. 恢复到原始点的索引
        output = output[inv]

        output = torch.cat([identity, output], dim=1)

        v_feat = torch_scatter.scatter_mean(
            self.layer_out(output[data_dict['coors_inv']]),
            data_dict['scale_{}'.format(self.scale)]['coors_inv'],
            dim=0
        )
        data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']

        return v_feat


class SPVBlock(nn.Module):
    """
    SPV Block: 稀疏点体素编码块
    集成 DALGC 模块（方向感知局部-全局上下文）以增强全局感知能力

    核心特性：
    - 3D Hilbert 曲线序列化：拓扑保持的点云排序，使 1D 卷积等效于 3D 邻域聚合
    - Strip Pooling：O(N) 复杂度的各向异性全局上下文建模
    """

    def __init__(self, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape, use_LGC,
                 use_strip_pooling=True, use_hilbert=True):
        super(SPVBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(in_channels, out_channels, self.indice_key),
            SparseBasicBlock(out_channels, out_channels, self.indice_key),
        )
        self.use_LGC = use_LGC
        if use_LGC:
            # 使用 DALGC 模块：Hilbert 序列化 + Strip Pooling 进行方向感知的全局上下文建模
            self.p_enc = point_encoder(
                in_channels, out_channels, scale,
                use_lpe=True,
                use_global_context=True,
                use_strip_pooling=use_strip_pooling,
                use_hilbert=use_hilbert
            )

    def forward(self, data_dict):
        global p_fea
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']

        # voxel encoder
        v_fea = self.v_enc(data_dict['sparse_tensor'])
        data_dict['layer_{}'.format(self.layer_id)] = {}
        data_dict['layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features
        data_dict['layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['full_coors']
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)

        # point encoder
        if self.use_LGC:
            p_fea = self.p_enc(
                features=data_dict['sparse_tensor'].features + v_fea.features,
                data_dict=data_dict
            )

        # fusion and pooling
        if self.use_LGC:
            data_dict['sparse_tensor'] = spconv.SparseConvTensor(
                features=p_fea + v_fea_inv,
                indices=data_dict['coors'],
                spatial_shape=self.spatial_shape,
                batch_size=data_dict['batch_size']
            )
            return p_fea[coors_inv]  # 返回LGC增强后的特征，映射回原始点数
        else:
            # 当不使用LGC时，需要手动更新坐标信息（原本由point_encoder更新）
            data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
            data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
            data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']

            data_dict['sparse_tensor'] = spconv.SparseConvTensor(
                features=v_fea_inv,
                indices=data_dict['coors'],
                spatial_shape=self.spatial_shape,
                batch_size=data_dict['batch_size']
            )
            return v_fea_inv[coors_inv]  # 返回纯体素特征，映射回原始点数


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.input_dims = config['model_params']['input_dims']
        self.hiden_size = config['model_params']['hiden_size']
        self.num_classes = config['model_params']['num_classes']
        self.scale_list = config['model_params']['scale_list']
        self.num_scales = len(self.scale_list)
        min_volume_space = config['dataset_params']['min_volume_space']
        max_volume_space = config['dataset_params']['max_volume_space']
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config['model_params']['spatial_shape'])
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]
        self.use_LGC = config['model_params'].get('use_LGC', True)  # 控制LGC模块的使用
        self.use_strip_pooling = config['model_params'].get('use_strip_pooling', True)  # 控制Strip Pooling的使用
        self.use_hilbert = config['model_params'].get('use_hilbert', True)  # 控制 Hilbert 曲线序列化的使用

        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        # encoder layers (with DALGC module supporting Hilbert serialization and Strip Pooling)
        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc.append(SPVBlock(
                in_channels=self.hiden_size,
                out_channels=self.hiden_size,
                indice_key='spv_' + str(i),
                scale=self.scale_list[i],
                last_scale=self.scale_list[i - 1] if i > 0 else 1,
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist(),
                use_LGC=self.use_LGC,
                use_strip_pooling=self.use_strip_pooling,
                use_hilbert=self.use_hilbert)
            )

        # decoder layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        # loss
        self.criterion = criterion(config)

    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)

        data_dict = self.voxel_3d_generator(data_dict)

        enc_feats = []
        for i in range(self.num_scales):
            enc_feats.append(self.spv_enc[i](data_dict))

        output = torch.cat(enc_feats, dim=1)
        data_dict['logits'] = self.classifier(output)

        data_dict['loss'] = 0.
        data_dict = self.criterion(data_dict)

        return data_dict


class criterion(nn.Module):
    def __init__(self, config):
        super(criterion, self).__init__()
        self.config = config
        self.lambda_lovasz = self.config['train_params'].get('lambda_lovasz', 0.1)

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=config['dataset_params']['ignore_label']
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )

    def forward(self, data_dict):
        loss_main_ce = self.ce_loss(data_dict['logits'], data_dict['labels'].long())
        loss_main_lovasz = self.lovasz_loss(F.softmax(data_dict['logits'], dim=1), data_dict['labels'].long())
        loss_main = loss_main_ce + loss_main_lovasz * self.lambda_lovasz

        data_dict['loss_main_ce'] = loss_main_ce
        data_dict['loss_main_lovasz'] = loss_main_lovasz
        data_dict['loss'] += loss_main

        return data_dict