import torch.nn as nn
import torch.nn.functional as F
import torch


class PointwiseUncertaintyEstimator(nn.Module):
    """
    Args:
        in_channels: 输入通道数
        reduction: 通道压缩比例（默认4）
        dropout_p: MC Dropout 概率（用于 epistemic 不确定性估计，默认0.2）
    """

    def __init__(self, in_channels, reduction=4, dropout_p=0.2):
        super(PointwiseUncertaintyEstimator, self).__init__()
        hidden_dim = max(in_channels // reduction, 8)
        self.dropout_p = dropout_p
        # 使用 Shared MLP 对每个点独立建模
        self.uncertainty_branch = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # 输出标量不确定性
        )

    def forward(self, x):
        """
        Args:
            x: (N, C) 点云特征
        Returns:
            uncertainty: (N, 1) 每个点的对数方差（不确定性）
        """
        # 逐点预测不确定性 log(σ²)
        uncertainty = self.uncertainty_branch(x)  # (N, 1)
        return uncertainty

    def stochastic_forward(self, x):
        h = self.uncertainty_branch[0](x)  # Linear -> (N, hidden_dim)
        h = self.uncertainty_branch[1](h)  # ReLU
        h = F.dropout(h, p=self.dropout_p, training=True)  # 始终生效的 MC Dropout
        return h  # (N, hidden_dim)


class PointUGAF(nn.Module):
    def __init__(self, c1, c2, mc_samples=4, dropout_p=0.2, lambda_e=1.0,
                 eps=1e-6, enable_epistemic=True, epistemic_in_eval=False):
        super(PointUGAF, self).__init__()

        # Epistemic (MC Dropout) hyper-parameters
        self.mc_samples = mc_samples
        self.lambda_e = lambda_e
        self.eps = eps
        self.enable_epistemic = enable_epistemic
        self.epistemic_in_eval = epistemic_in_eval

        # A. 像素级不确定性估计分支（支持 MC Dropout）
        self.uncertainty_img = PointwiseUncertaintyEstimator(c1, dropout_p=dropout_p)
        self.uncertainty_pts = PointwiseUncertaintyEstimator(c2, dropout_p=dropout_p)

        # B. 点级语义细化
        self.conv = nn.Sequential(
            nn.Linear(c2, c2),
            nn.LayerNorm(c2),
            nn.ReLU(inplace=True)
        )
        # 空间重要性掩码生成
        self.point_attn = nn.Sequential(
            nn.Linear(c2, c2 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // 4, 1),
            nn.Sigmoid()
        )

    def _mc_epistemic(self, estimator, x):
        if self.mc_samples <= 1:
            return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        # 收集 T 次隐藏特征向量 z_m^{(i,t)}
        samples = torch.stack(
            [estimator.stochastic_forward(x) for _ in range(self.mc_samples)],
            dim=0,
        )  # (T, N, hidden_dim)
        # 逐通道方差: v_{m,e}^{(i)} ∈ R^{hidden_dim}
        var_z = samples.var(dim=0, unbiased=False)  # (N, hidden_dim)
        # 通道均值 → 点级标量: u_{m,e}^{(i)}
        u_e = torch.log(var_z.mean(dim=-1, keepdim=True) + self.eps)  # (N, 1)
        return u_e

    def forward(self, x1, x2, return_uncertainty=False):
        """
        Args:
            x1: (N, c1) 图像特征（投影后）
            x2: (N, c2) 点云特征
            return_uncertainty: 是否额外返回不确定性字典（调试/可视化用）
        Returns:
            X_out: (N, c2) 融合后的特征
            （可选）uncertainty_dict: dict  包含各分量不确定性
        """
        # ========== A. 偶然 (Aleatoric) 不确定性 ==========
        u_img_a = self.uncertainty_img(x1)  # (N, 1)
        u_pts_a = self.uncertainty_pts(x2)  # (N, 1)

        # ========== A+. 认知 (Epistemic) 不确定性 — MC Dropout ==========
        if self.enable_epistemic and (self.training or self.epistemic_in_eval):
            u_img_e = self._mc_epistemic(self.uncertainty_img, x1)  # (N, 1)
            u_pts_e = self._mc_epistemic(self.uncertainty_pts, x2)  # (N, 1)
            u_img = u_img_a + self.lambda_e * u_img_e
            u_pts = u_pts_a + self.lambda_e * u_pts_e
        else:
            u_img_e = u_pts_e = None
            u_img = u_img_a
            u_pts = u_pts_a

        # 基于空间方差反比的动态加权（Softmax over -u）
        # M_I = exp(-u_I) / (exp(-u_I) + exp(-u_P))
        neg_u = torch.cat([-u_img, -u_pts], dim=1)  # (N, 2)
        confidence_mask = F.softmax(neg_u, dim=1)  # (N, 2)

        M_img = confidence_mask[:, 0:1]  # (N, 1) 图像置信度掩码
        M_pts = confidence_mask[:, 1:2]  # (N, 1) 点云置信度掩码

        # 空间加权融合
        # X_fuse = M_I ⊙ X_I + M_P ⊙ X_P
        X_fuse = M_img * x1 + M_pts * x2  # (N, c2)

        # ========== B. 点级语义细化 ==========
        # 非线性特征变换
        X_fuse_conv = self.conv(X_fuse)

        # 空间重要性掩码
        point_weight = self.point_attn(X_fuse_conv)  # (N, 1)

        # 最终输出
        X_out = X_fuse_conv * point_weight

        if return_uncertainty:
            unc_dict = {
                'u_img_aleatoric': u_img_a,
                'u_pts_aleatoric': u_pts_a,
                'u_img_epistemic': u_img_e,
                'u_pts_epistemic': u_pts_e,
                'u_img': u_img,
                'u_pts': u_pts,
                'M_img': M_img,
                'M_pts': M_pts,
            }
            return X_out, unc_dict
        return X_out


class UGAF_PointAdapter(nn.Module):

    def __init__(self, dim, mc_samples=4, dropout_p=0.2, lambda_e=1.0,
                 eps=1e-6, enable_epistemic=True, epistemic_in_eval=False):
        super(UGAF_PointAdapter, self).__init__()
        self.dim = dim

        # 特征适配（当输入维度不一致时使用）
        self.adapt = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )

        # UGAF路径：不确定性引导自适应融合（支持 aleatoric + epistemic）
        self.ugaf = PointUGAF(
            dim, dim,
            mc_samples=mc_samples,
            dropout_p=dropout_p,
            lambda_e=lambda_e,
            eps=eps,
            enable_epistemic=enable_epistemic,
            epistemic_in_eval=epistemic_in_eval,
        )

    def forward(self, img_feat, pts_feat):
        """
        Args:
            img_feat: (N, C) 2D图像特征（投影到点后）
            pts_feat: (N, C) 3D点云特征
        Returns:
            fused_feat: (N, C) 融合后的特征
        """
        # 特征适配
        x1 = self.adapt(img_feat)  # 2D特征
        x2 = pts_feat  # 3D特征

        # UGAF 不确定性引导融合
        fused = self.ugaf(x1, x2)

        return fused


# ============ 保留原有接口的兼容性别名 ============
class PointSimpleAttention(nn.Module):
    """
    [已废弃] 原全局通道注意力模块，保留用于向后兼容
    建议使用 PointwiseUncertaintyEstimator 替代
    """

    def __init__(self, in_channels, reduction=16):
        super(PointSimpleAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, max(in_channels // reduction, 8))
        self.fc2 = nn.Linear(max(in_channels // reduction, 8), in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gap = x.mean(dim=0)
        attention = F.relu(self.fc1(gap))
        attention = self.sigmoid(self.fc2(attention))
        return attention.unsqueeze(0)


# 兼容性别名
PointACMF = PointUGAF
ACMF_PointAdapter = UGAF_PointAdapter