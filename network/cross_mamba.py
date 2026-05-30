import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn


try:
    from mamba_ssm import Mamba  # type: ignore
    _MAMBA_AVAILABLE = True
except Exception:
    Mamba = None
    _MAMBA_AVAILABLE = False


class _FallbackSSM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.proj(out)


def _hilbert_index_3d(coords: torch.Tensor, bits: int) -> torch.Tensor:
    x = coords.clone().long()
    n = 3
    m = 1 << (bits - 1)
    q = m
    while q > 1:
        p = q - 1
        for i in range(n):
            mask = (x[:, i] & q) != 0
            x0 = x[:, 0]
            x0 = torch.where(mask, x0 ^ p, x0)
            t = (x[:, 0] ^ x[:, i]) & p
            x0 = torch.where(~mask, x0 ^ t, x0)
            xi = x[:, i]
            xi = torch.where(~mask, xi ^ t, xi)
            x[:, 0] = x0
            x[:, i] = xi
        q >>= 1

    for i in range(1, n):
        x[:, i] ^= x[:, i - 1]

    t = torch.zeros_like(x[:, 0])
    q = m
    while q > 1:
        mask = (x[:, n - 1] & q) != 0
        t = torch.where(mask, t ^ (q - 1), t)
        q >>= 1

    for i in range(n):
        x[:, i] ^= t

    h = torch.zeros_like(x[:, 0])
    for i in range(bits - 1, -1, -1):
        for j in range(n):
            h = (h << 1) | ((x[:, j] >> i) & 1)
    return h


def _normalize_coords(coords: torch.Tensor, bits: int) -> torch.Tensor:
    coords = coords.float()
    min_c = coords.min(dim=0).values
    max_c = coords.max(dim=0).values
    span = (max_c - min_c).clamp(min=1.0)
    scaled = (coords - min_c) / span
    scaled = scaled * (2**bits - 1)
    return scaled.round().clamp(0, 2**bits - 1).long()


def _hilbert_sort_indices(
    coords: torch.Tensor,
    batch_idx: Optional[torch.Tensor],
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = coords.device
    if batch_idx is None:
        batch_idx = torch.zeros(coords.shape[0], device=device, dtype=torch.long)

    unique_batches = torch.unique(batch_idx, sorted=True)
    sort_chunks = []
    for b in unique_batches:
        mask = batch_idx == b
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        coords_b = _normalize_coords(coords[idx], bits)
        keys = _hilbert_index_3d(coords_b, bits)
        order = torch.argsort(keys)
        sort_chunks.append(idx[order])

    if len(sort_chunks) == 0:
        sort_idx = torch.arange(coords.shape[0], device=device)
    else:
        sort_idx = torch.cat(sort_chunks, dim=0)

    inv_idx = torch.empty_like(sort_idx)
    inv_idx[sort_idx] = torch.arange(sort_idx.numel(), device=device)
    return sort_idx, inv_idx


class _CrossSequenceBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if _MAMBA_AVAILABLE:
            self.seq = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            print("mamba_ssm found")
        else:
            warnings.warn(
                "mamba_ssm not found, using bidirectional GRU fallback for Cross-Mamba.",
                RuntimeWarning,
            )
            self.seq = _FallbackSSM(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class CrossMambaInteraction(nn.Module):
    """Cross-Mamba Interaction with ablation switches.

    Ablation dimensions (all have defaults that reproduce the original full behaviour):

    * ``cmf_cross_mode``  – ``"cross_parameter"`` (default, FiLM) or ``"feature_concat"``
    * ``cmf_alpha_mode``  – ``"none"`` | ``"r"`` | ``"g"`` | ``"gr_mlp"`` (default)
    * ``cmf_use_ctx``     – ``True`` (default) / ``False``
    * ``cmf_use_hilbert`` – ``True`` (default) / ``False``
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        hilbert_bits: int = 10,
        geo_scale_mode: str = "residual",
        # ---- ablation switches ----
        cmf_cross_mode: str = "cross_parameter",
        cmf_alpha_mode: str = "gr_mlp",
        cmf_use_ctx: bool = True,
        cmf_use_hilbert: bool = True,
    ):
        super().__init__()
        self.hilbert_bits = hilbert_bits
        if geo_scale_mode not in ("residual", "gate"):
            raise ValueError(f"Unsupported geo_scale_mode: {geo_scale_mode}")
        if cmf_cross_mode not in ("cross_parameter", "feature_concat"):
            raise ValueError(f"Unsupported cmf_cross_mode: {cmf_cross_mode}")
        if cmf_alpha_mode not in ("none", "r", "g", "gr_mlp"):
            raise ValueError(f"Unsupported cmf_alpha_mode: {cmf_alpha_mode}")

        self.geo_scale_mode = geo_scale_mode
        self.cmf_cross_mode = cmf_cross_mode
        self.cmf_alpha_mode = cmf_alpha_mode
        self.cmf_use_ctx = cmf_use_ctx
        self.cmf_use_hilbert = cmf_use_hilbert

        self.img_norm = nn.LayerNorm(dim)
        self.pts_norm = nn.LayerNorm(dim)

        # ---------- alpha gating ----------
        if cmf_alpha_mode == "gr_mlp":
            alpha_hidden = max(dim // 4, 8)
            self.alpha_mlp = nn.Sequential(
                nn.Linear(2, alpha_hidden),
                nn.GELU(),
                nn.Linear(alpha_hidden, 1),
            )
        elif cmf_alpha_mode == "g":
            # Learnable scalar transform on geometric condition
            self.alpha_g_linear = nn.Linear(1, 1)

        # ---------- cross-modality interaction ----------
        if cmf_cross_mode == "cross_parameter":
            cond_hidden = max(dim // 4, 8)
            self.img_cond = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, cond_hidden),
                nn.GELU(),
                nn.Linear(cond_hidden, dim * 2),
            )
            self.pts_cond = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, cond_hidden),
                nn.GELU(),
                nn.Linear(cond_hidden, dim * 2),
            )
        elif cmf_cross_mode == "feature_concat":
            self.img_cross_proj = nn.Linear(dim, dim)
            self.pts_cross_proj = nn.Linear(dim, dim)

        # Interaction gates (used in both cross modes)
        self.img_gate = nn.Linear(dim, dim)
        self.pts_gate = nn.Linear(dim, dim)

        self.img_seq = _CrossSequenceBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.pts_seq = _CrossSequenceBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.img_out_norm = nn.LayerNorm(dim)
        self.pts_out_norm = nn.LayerNorm(dim)
        self.ctx_norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_reliability(
        self, reliability: Optional[torch.Tensor], sort_idx: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return per-modality reliability (N,1) in sorted order, or *None*."""
        if reliability is None:
            return None, None
        reliability = reliability.clamp(0.0, 1.0)
        if reliability.shape[-1] == 2:
            r_img = reliability[:, 0:1]
            r_pts = reliability[:, 1:2]
        elif reliability.shape[-1] == 1:
            r_img = reliability
            r_pts = reliability
        else:
            raise ValueError("reliability must be shape (N,2) or (N,1)")
        return r_img[sort_idx], r_pts[sort_idx]

    def _compute_alpha(
        self,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
        r_img_sorted: Optional[torch.Tensor],
        r_pts_sorted: Optional[torch.Tensor],
        geo_sorted: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute (alpha_img, alpha_pts) each of shape (N,1) or *None*."""

        if self.cmf_alpha_mode == "none":
            return None, None

        if self.cmf_alpha_mode == "r":
            # alpha = reliability of the *other* modality
            return r_pts_sorted, r_img_sorted

        if self.cmf_alpha_mode == "g":
            if geo_sorted is not None:
                alpha = torch.sigmoid(self.alpha_g_linear(geo_sorted))
            else:
                # Missing geo → safe default = 1 (no attenuation)
                alpha = torch.ones((N, 1), device=device, dtype=dtype)
            return alpha, alpha

        # ---- "gr_mlp" (default / full model) ----
        if geo_sorted is None and r_img_sorted is None and r_pts_sorted is None:
            return None, None

        g_val = geo_sorted if geo_sorted is not None else torch.zeros(
            (N, 1), device=device, dtype=dtype,
        )
        r_for_img = r_pts_sorted if r_pts_sorted is not None else torch.ones(
            (N, 1), device=device, dtype=dtype,
        )
        r_for_pts = r_img_sorted if r_img_sorted is not None else torch.ones(
            (N, 1), device=device, dtype=dtype,
        )
        alpha_img = torch.sigmoid(self.alpha_mlp(torch.cat([g_val, r_for_img], dim=-1)))
        alpha_pts = torch.sigmoid(self.alpha_mlp(torch.cat([g_val, r_for_pts], dim=-1)))
        return alpha_img, alpha_pts

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        img_feat: torch.Tensor,
        pts_feat: torch.Tensor,
        coords: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
        geo_cond: Optional[torch.Tensor] = None,
        ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_idx is None:
            batch_idx = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)

        # ---------- Hilbert serialisation ----------
        if self.cmf_use_hilbert:
            sort_idx, inv_idx = _hilbert_sort_indices(coords, batch_idx, self.hilbert_bits)
        else:
            # IMPORTANT: in no-Hilbert ablation we still group by batch id,
            # otherwise non-contiguous batch ordering can create many tiny
            # sequence segments and drastically slow down per-segment SSM calls.
            # This keeps "no spatial Hilbert ordering" while preserving runtime.
            sort_idx = torch.argsort(batch_idx)
            inv_idx = torch.empty_like(sort_idx)
            inv_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)

        img_sorted = img_feat[sort_idx]
        pts_sorted = pts_feat[sort_idx]

        # ---------- Progressive context injection ----------
        _ctx_sorted: Optional[torch.Tensor] = None
        if self.cmf_use_ctx and ctx is not None:
            _ctx_sorted = ctx[sort_idx]
            img_sorted = img_sorted + _ctx_sorted
            pts_sorted = pts_sorted + _ctx_sorted

        # ---------- Reliability & geo ----------
        r_img_sorted, r_pts_sorted = self._parse_reliability(reliability, sort_idx)

        geo_sorted: Optional[torch.Tensor] = None
        if geo_cond is not None:
            geo_sorted = geo_cond.clamp(0.0, 1.0)[sort_idx]

        # ---------- Alpha gating ----------
        alpha_img, alpha_pts = self._compute_alpha(
            img_sorted.shape[0], img_sorted.device, img_sorted.dtype,
            r_img_sorted, r_pts_sorted, geo_sorted,
        )

        # ---------- Cross modulation ----------
        if self.cmf_cross_mode == "cross_parameter":
            img_gamma, img_beta = self.img_cond(pts_sorted).chunk(2, dim=-1)
            pts_gamma, pts_beta = self.pts_cond(img_sorted).chunk(2, dim=-1)

            if alpha_img is not None:
                img_gamma = img_gamma * alpha_img
                img_beta = img_beta * alpha_img
            if alpha_pts is not None:
                pts_gamma = pts_gamma * alpha_pts
                pts_beta = pts_beta * alpha_pts

            # Direct reliability scaling (only in gr_mlp mode to maintain
            # current full-model behaviour)
            if self.cmf_alpha_mode == "gr_mlp":
                if r_pts_sorted is not None:
                    img_gamma = img_gamma * r_pts_sorted
                    img_beta = img_beta * r_pts_sorted
                if r_img_sorted is not None:
                    pts_gamma = pts_gamma * r_img_sorted
                    pts_beta = pts_beta * r_img_sorted

            img_in = self.img_norm(img_sorted)
            img_in = img_in * (1.0 + torch.tanh(img_gamma)) + img_beta
            pts_in = self.pts_norm(pts_sorted)
            pts_in = pts_in * (1.0 + torch.tanh(pts_gamma)) + pts_beta
        else:
            # "feature_concat": feature-level interaction (no FiLM)
            img_in = self.img_norm(img_sorted) + self.img_cross_proj(pts_sorted)
            pts_in = self.pts_norm(pts_sorted) + self.pts_cross_proj(img_sorted)

        # ---------- SSM per batch segment ----------
        img_out_sorted = torch.zeros_like(img_in)
        pts_out_sorted = torch.zeros_like(pts_in)

        batch_sorted = batch_idx[sort_idx]
        if batch_sorted.numel() > 0:
            boundaries = torch.nonzero(batch_sorted[1:] != batch_sorted[:-1], as_tuple=False)
            boundaries = boundaries.squeeze(-1) + 1
            segment_starts = [0] + boundaries.tolist() + [batch_sorted.numel()]
            for s, e in zip(segment_starts[:-1], segment_starts[1:]):
                img_seq = img_in[s:e].unsqueeze(0)
                pts_seq = pts_in[s:e].unsqueeze(0)
                img_out_sorted[s:e] = self.img_seq(img_seq).squeeze(0)
                pts_out_sorted[s:e] = self.pts_seq(pts_seq).squeeze(0)

        img_out = img_out_sorted
        pts_out = pts_out_sorted

        # ---------- Interaction gate ----------
        img_gate = torch.tanh(self.img_gate(pts_sorted))
        pts_gate = torch.tanh(self.pts_gate(img_sorted))

        if alpha_img is not None:
            img_gate = img_gate * alpha_img
        if alpha_pts is not None:
            pts_gate = pts_gate * alpha_pts

        # Direct reliability scaling on gates (only in gr_mlp mode)
        if self.cmf_alpha_mode == "gr_mlp":
            if r_pts_sorted is not None:
                img_gate = img_gate * r_pts_sorted
            if r_img_sorted is not None:
                pts_gate = pts_gate * r_img_sorted

        if geo_sorted is not None:
            geo_scale = 0.5 + geo_sorted
            if self.geo_scale_mode == "gate":
                img_gate = img_gate * geo_scale
                pts_gate = pts_gate * geo_scale

        img_out = img_out * (1.0 + img_gate)
        pts_out = pts_out * (1.0 + pts_gate)

        if geo_sorted is not None and self.geo_scale_mode == "residual":
            geo_scale = 0.5 + geo_sorted
            img_out = img_out * geo_scale
            pts_out = pts_out * geo_scale

        img_out = img_out[inv_idx]
        pts_out = pts_out[inv_idx]

        # ---------- Residual + norm ----------
        # Include ctx in residual path to match pre-refactor behaviour where
        # ctx was added to img_feat/pts_feat *before* entering this module.
        if self.cmf_use_ctx and ctx is not None:
            img_res = img_feat + ctx
            pts_res = pts_feat + ctx
        else:
            img_res = img_feat
            pts_res = pts_feat

        img_feat_new = self.img_out_norm(img_res + img_out)
        pts_feat_new = self.pts_out_norm(pts_res + pts_out)

        fused_ctx = self.ctx_norm((img_feat_new + pts_feat_new) * 0.5)
        return img_feat_new, pts_feat_new, fused_ctx
