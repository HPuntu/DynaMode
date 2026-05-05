'''Slow-mode residual head + composite model.

This module implements v9 of the roadmap: a parallel, independent low-k
predictor that runs alongside the v8 SpectralConvDiT trunk and adds an
additive correction to the trunk's low-k output.

Design goals
------------
The :class:`SlowBranch` itself is intentionally agnostic about the trunk it
is paired with. It takes raw per-residue and scalar conditioning signals
(native coordinates, temperature, protein size, window position, sequence,
DSSP, NMA RMSF prior) and emits a dense prediction of the low-k spectral
coefficients directly. It contains no spectral-trunk internals, no
assumptions about the mid/high-k pathway, and no shared parameters with
any specific trunk. This keeps it reusable:

- v9 pairs it with ``SpectralConvDiT`` as an additive residual on low-k.
- v10 pairs the same branch with a block-diagonal-mixer trunk (same
  semantics, different mixer inside the trunk).
- v11 will promote :class:`SlowBranch` to a standalone model inside a
  fully-parallel dual-branch architecture.

The final output projection is zero-initialised so the module emits
identical outputs at step 0 to a trunk without it — safe warm-start from
any v8 checkpoint.

Classes
-------
:class:`SlowBranch`
    Standalone low-k predictor. Outputs ``(B, L, K_slow, C_in)`` tensor
    in raw coefficient space.

:class:`SpectralConvSlowBranch`
    Composite model wrapping a :class:`SpectralConvDiT` trunk and a
    :class:`SlowBranch` head. Its ``forward`` signature matches the trunk
    so downstream pipeline code (:mod:`src.models.model_wrapper`,
    :mod:`src.train`, :mod:`src.models.diffusion`) can consume it
    unchanged.
'''

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.models.spectral_conv import SpectralConvDiT
from src.models.modules import (
    RotaryEmbedding,
    SmoothScalarEmbedding,
    SpectralBlock,
    SwiGLU,
    WindowContextEmbedding,
)


class SlowBranch(nn.Module):
    '''Residue-level transformer predicting the low-k spectral coefficients.

    The module is conditioned on global protein-level signals (pooled
    sequence / DSSP / RMSF / structure + scalars) and receives per-residue
    local features (native coords, residue type, DSSP one-hot, RMSF prior)
    as token inputs. It runs a small transformer stack and projects the
    tokens to ``K_slow * in_channels`` low-k coefficients per residue.

    The final projection is zero-initialised — the module's forward output
    is therefore the zero tensor at initialisation, which makes it safe to
    add to another model's predictions without perturbing warm-start
    behaviour.

    Args:
        in_channels: Coordinate channels predicted per slow mode.
        cond_channels: Native-coord channels supplied as per-residue input.
        K_slow: Number of low-k modes predicted by this branch.
        d_model: Hidden width of the internal transformer.
        depth: Number of transformer blocks.
        num_heads: Attention heads (must divide ``d_model``).
        mlp_ratio: SwiGLU expansion ratio in each block.
        use_seq_conditioning: If True, consume ``res_type`` as a per-residue
            embedding input and pool it for global conditioning.
        use_ss_conditioning: Same as above for ``dssp``.
        seq_embed_dim: Residue-type embedding width.
        ss_embed_dim: DSSP embedding width.
        num_res_types: Vocabulary size for residue-type embeddings.
        num_dssp_states: Number of DSSP secondary-structure states.
        use_rmsf_prior: If True, append per-residue RMSF prior as a scalar
            input feature and pool for global conditioning.

    Shape contract:
        Inputs:
            native_coords  (B, L, cond_channels)
            temp_norm      (B,)            — normalised to [0, 1] via
                                             ``(T_K - 250) / 200`` upstream;
                                             this module de-normalises back
                                             to Kelvin for the context MLP.
            size_scalar    (B,)            — number of valid residues.
            win_pos        (B,)            — window start fraction in [0, 1].
            res_type       (B, L) or None
            dssp           (B, L) or None
            rmsf_prior     (B, L) or None
            mask           (B, L) or None
        Output:
            delta_slow     (B, L, K_slow, in_channels) — zero at init.
    '''

    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 3,
        K_slow: int = 16,
        d_model: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        use_seq_conditioning: bool = False,
        use_ss_conditioning: bool = False,
        seq_embed_dim: int = 16,
        ss_embed_dim: int = 8,
        num_res_types: int = 21,
        num_dssp_states: int = 8,
        use_rmsf_prior: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f'd_model={d_model} must be divisible by num_heads={num_heads}'
            )

        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.K_slow = int(K_slow)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.use_seq_conditioning = bool(use_seq_conditioning)
        self.use_ss_conditioning = bool(use_ss_conditioning)
        self.use_rmsf_prior = bool(use_rmsf_prior)
        self.seq_embed_dim = int(seq_embed_dim) if self.use_seq_conditioning else 0
        self.ss_embed_dim = int(ss_embed_dim) if self.use_ss_conditioning else 0
        self.num_res_types = int(num_res_types)
        self.num_dssp_states = int(num_dssp_states)

        # Per-residue local feature width: native coords + optional embeddings +
        # optional RMSF scalar.
        local_feat_dim = (
            self.cond_channels
            + self.seq_embed_dim
            + self.ss_embed_dim
            + (1 if self.use_rmsf_prior else 0)
        )
        self.local_proj = nn.Linear(local_feat_dim, self.d_model)

        # Optional per-residue embeddings.
        if self.use_seq_conditioning:
            self.residue_embed = nn.Embedding(self.num_res_types, self.seq_embed_dim)
            nn.init.normal_(self.residue_embed.weight, std=0.02)
        if self.use_ss_conditioning:
            self.ss_local_proj = nn.Linear(self.num_dssp_states, self.ss_embed_dim)

        # Global (per-protein) conditioning stack projected into ``d_model``
        # to drive the AdaLN-Zero modulation inside each SpectralBlock.
        self.temp_embedder = SmoothScalarEmbedding(self.d_model)
        self.win_ctx_mlp = WindowContextEmbedding(self.d_model)
        if self.use_seq_conditioning:
            self.seq_global_proj = nn.Linear(self.seq_embed_dim, self.d_model)
        if self.use_ss_conditioning:
            self.ss_global_proj = nn.Linear(self.ss_embed_dim, self.d_model)
        if self.use_rmsf_prior:
            self.rmsf_global_proj = nn.Linear(1, self.d_model)

        # Transformer stack over residues.
        self.rope = RotaryEmbedding(self.d_model // self.num_heads)
        self.blocks = nn.ModuleList([
            SpectralBlock(
                hidden_size=self.d_model,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                dropout=attn_dropout,
                use_cross_attn=False,
                use_freq_coords=False,
            )
            for _ in range(self.depth)
        ])
        self.final_norm = nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1e-6)

        # Zero-init final projection so the module's output is 0 at init.
        self.out_proj = nn.Linear(self.d_model, self.K_slow * self.in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        '''Mean over residues, ignoring padded positions.

        Args:
            x: ``(B, L, D)``.
            mask: ``(B, L)`` or None.

        Returns:
            ``(B, D)`` per-protein mean.
        '''
        if mask is None:
            return x.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (x * m).sum(dim=1) / denom

    def _build_cond(
        self,
        temp_norm: torch.Tensor,
        size_scalar: torch.Tensor,
        win_pos: torch.Tensor,
        seq_pooled: torch.Tensor | None,
        ss_pooled: torch.Tensor | None,
        rmsf_pooled: torch.Tensor | None,
    ) -> torch.Tensor:
        '''Assemble global conditioning vector ``(B, d_model)``.'''
        # De-normalise temperature to Kelvin for the context MLP.
        temp_k = temp_norm.float() * 200.0 + 250.0
        c = self.temp_embedder(temp_norm) + self.win_ctx_mlp(win_pos, temp_k, size_scalar)
        if seq_pooled is not None:
            c = c + self.seq_global_proj(seq_pooled)
        if ss_pooled is not None:
            c = c + self.ss_global_proj(ss_pooled)
        if rmsf_pooled is not None:
            c = c + self.rmsf_global_proj(rmsf_pooled.unsqueeze(-1))
        return c

    def forward(
        self,
        native_coords: torch.Tensor,
        temp_norm: torch.Tensor,
        size_scalar: torch.Tensor,
        win_pos: torch.Tensor,
        res_type: torch.Tensor | None = None,
        dssp: torch.Tensor | None = None,
        rmsf_prior: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        '''Predict low-k coefficient correction.

        Args:
            native_coords: ``(B, L, cond_channels)`` reference structure.
            temp_norm: ``(B,)`` temperature, normalised to ``[0, 1]`` by
                ``(T_K - 250) / 200`` upstream.
            size_scalar: ``(B,)`` number of valid residues per sample.
            win_pos: ``(B,)`` window start fraction.
            res_type: ``(B, L)`` integer labels or ``(B, L, num_res_types)``
                one-hot / soft. Optional.
            dssp: ``(B, L)`` integer labels or ``(B, L, num_dssp_states)``
                one-hot. Optional.
            rmsf_prior: ``(B, L)`` per-residue unitless NMA RMSF. Optional.
            mask: ``(B, L)`` validity mask. Optional.

        Returns:
            ``(B, L, K_slow, in_channels)`` low-k coefficient prediction.
        '''
        B, L = native_coords.shape[:2]
        device = native_coords.device
        dtype = native_coords.dtype

        # Per-residue local feature assembly.
        feats = [native_coords]
        if self.use_seq_conditioning:
            if res_type is None:
                seq_local = torch.zeros(B, L, self.seq_embed_dim, device=device, dtype=dtype)
            else:
                if res_type.ndim == 3:
                    res_idx = res_type.argmax(dim=-1)
                else:
                    res_idx = res_type
                res_idx = res_idx.long().clamp(0, self.num_res_types - 1)
                seq_local = self.residue_embed(res_idx)
            feats.append(seq_local)
        else:
            seq_local = None

        if self.use_ss_conditioning:
            if dssp is None:
                ss_base = torch.zeros(B, L, self.num_dssp_states, device=device, dtype=dtype)
                ss_base[..., -1] = 1.0
            else:
                if dssp.ndim == 2:
                    ss_base = F.one_hot(
                        dssp.long().clamp(0, self.num_dssp_states - 1),
                        num_classes=self.num_dssp_states,
                    ).to(dtype)
                else:
                    ss_base = dssp.to(dtype)
            ss_local = self.ss_local_proj(ss_base)
            feats.append(ss_local)
        else:
            ss_local = None

        if self.use_rmsf_prior:
            if rmsf_prior is None:
                rmsf_local = torch.zeros(B, L, 1, device=device, dtype=dtype)
            else:
                rmsf_local = rmsf_prior.to(dtype).unsqueeze(-1)
            feats.append(rmsf_local)
        else:
            rmsf_local = None

        local = torch.cat(feats, dim=-1)
        if mask is not None:
            local = local * mask.unsqueeze(-1).to(local.dtype)
        tokens = self.local_proj(local)  # (B, L, d_model)

        # Global conditioning.
        seq_pooled = self._masked_mean(seq_local, mask) if seq_local is not None else None
        ss_pooled = self._masked_mean(ss_local, mask) if ss_local is not None else None
        rmsf_pooled = (
            self._masked_mean(rmsf_local, mask).squeeze(-1)
            if rmsf_local is not None
            else None
        )
        c = self._build_cond(temp_norm, size_scalar, win_pos,
                             seq_pooled, ss_pooled, rmsf_pooled)

        # Transformer stack with residue self-attention.
        rope_freqs = self.rope(tokens)
        for block in self.blocks:
            tokens = block(tokens, c, rope_freqs, context=None, mask=mask)

        tokens = self.final_norm(tokens)
        out = self.out_proj(tokens)  # (B, L, K_slow * in_channels)
        out = out.view(B, L, self.K_slow, self.in_channels)
        if mask is not None:
            out = out * mask.unsqueeze(-1).unsqueeze(-1).to(out.dtype)
        return out


class SpectralConvSlowBranch(nn.Module):
    '''Composite model: SpectralConvDiT trunk + additive SlowBranch head.

    The trunk and the slow branch run in parallel. The branch's output is
    added to the trunk's predicted low-k coefficients at the end of the
    forward pass. Because the slow branch is zero-initialised, step-0
    behaviour is identical to a bare SpectralConvDiT — a v8 checkpoint can
    be loaded with ``strict=False`` and training continues without a
    discontinuity.

    The branch is responsible for the low-k regime by design; the trunk's
    :attr:`SpectralConvDiT.use_low_k_correction_head` feature is therefore
    forced off to avoid dueling additive corrections.
    '''

    # Pipeline-facing class attribute mirrored from the trunk.
    is_time_domain: bool = False

    def __init__(
        self,
        # Trunk parameters (mirrors SpectralConvDiT.__init__).
        top_k_freqs: int = 256,
        in_channels: int = 3,
        cond_channels: int = 3,
        freq_hidden_size: int = 8,
        depth: int = 12,
        num_heads: int = 4,
        spectral_modes: int = 64,
        attn_dropout: float = 0.0,
        freq_scale: torch.Tensor | None = None,
        conditioned_freq_scale: dict | None = None,
        cfg_dropout: bool = False,
        prediction_target: str = 'v',
        is_dct: bool = True,
        use_hilbert: bool = False,
        use_hilbert_dct: bool = False,
        hilbert_mode: str = 'every_block',
        use_rmsf_prior_gain: bool = False,
        use_low_k_correction_head: bool = False,  # forced False below
        low_k_correction_modes: int | str = 1,
        use_seq_conditioning: bool = False,
        seq_embed_dim: int = 16,
        use_ss_conditioning: bool = False,
        ss_embed_dim: int = 8,
        cond_dim: int = 512,
        # Slow-branch parameters.
        K_slow: int = 16,
        slow_d_model: int = 128,
        slow_depth: int = 3,
        slow_num_heads: int = 4,
        slow_mlp_ratio: float = 4.0,
        slow_attn_dropout: float = 0.0,
        slow_use_rmsf_prior: bool = False,
    ) -> None:
        super().__init__()
        if use_low_k_correction_head:
            # Silently force off rather than raising — lets the same config
            # file drive both v8 and v9 training.
            use_low_k_correction_head = False

        self.trunk = SpectralConvDiT(
            top_k_freqs=top_k_freqs,
            in_channels=in_channels,
            cond_channels=cond_channels,
            freq_hidden_size=freq_hidden_size,
            depth=depth,
            num_heads=num_heads,
            spectral_modes=spectral_modes,
            attn_dropout=attn_dropout,
            freq_scale=freq_scale,
            conditioned_freq_scale=conditioned_freq_scale,
            cfg_dropout=cfg_dropout,
            prediction_target=prediction_target,
            is_dct=is_dct,
            use_hilbert=use_hilbert,
            use_hilbert_dct=use_hilbert_dct,
            hilbert_mode=hilbert_mode,
            use_rmsf_prior_gain=use_rmsf_prior_gain,
            use_low_k_correction_head=use_low_k_correction_head,
            low_k_correction_modes=low_k_correction_modes,
            use_seq_conditioning=use_seq_conditioning,
            seq_embed_dim=seq_embed_dim,
            use_ss_conditioning=use_ss_conditioning,
            ss_embed_dim=ss_embed_dim,
            cond_dim=cond_dim,
        )

        self.K_slow = int(K_slow)
        self.slow_branch = SlowBranch(
            in_channels=in_channels,
            cond_channels=cond_channels,
            K_slow=self.K_slow,
            d_model=slow_d_model,
            depth=slow_depth,
            num_heads=slow_num_heads,
            mlp_ratio=slow_mlp_ratio,
            attn_dropout=slow_attn_dropout,
            use_seq_conditioning=use_seq_conditioning,
            use_ss_conditioning=use_ss_conditioning,
            seq_embed_dim=seq_embed_dim,
            ss_embed_dim=ss_embed_dim,
            use_rmsf_prior=slow_use_rmsf_prior,
        )

    # ------------------------------------------------------------------
    # Pipeline-facing attribute delegation.
    # ------------------------------------------------------------------
    @property
    def prediction_target(self) -> str:
        return self.trunk.prediction_target

    @property
    def is_dct(self) -> bool:
        return self.trunk.is_dct

    @property
    def freq_scale(self):
        return self.trunk.freq_scale

    @property
    def spectral_adapter(self):
        return self.trunk.spectral_adapter

    @property
    def top_k_freqs(self) -> int:
        return self.trunk.top_k_freqs

    @property
    def in_channels(self) -> int:
        return self.trunk.in_channels

    @property
    def use_low_k_correction_head(self) -> bool:
        return self.trunk.use_low_k_correction_head

    @property
    def use_rmsf_prior_gain(self) -> bool:
        return self.trunk.use_rmsf_prior_gain

    @property
    def cfg_dropout(self) -> bool:
        return self.trunk.cfg_dropout

    def apply_rmsf_prior_gain(
        self,
        x: torch.Tensor,
        rmsf_prior: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.trunk.apply_rmsf_prior_gain(x, rmsf_prior, mask=mask)

    # ------------------------------------------------------------------
    # Forward pass.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        temp: torch.Tensor,
        native_coords: torch.Tensor,
        mask: torch.Tensor | None = None,
        win_pos: torch.Tensor | None = None,
        cond_drop_mask: torch.Tensor | None = None,
        native_angles: torch.Tensor | None = None,
        res_type: torch.Tensor | None = None,
        dssp: torch.Tensor | None = None,
        rmsf_prior: torch.Tensor | None = None,
        freq_scale_override: torch.Tensor | None = None,
        scale_cond: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        trunk_out = self.trunk(
            x, t, temp, native_coords,
            mask=mask, win_pos=win_pos, cond_drop_mask=cond_drop_mask,
            native_angles=native_angles, res_type=res_type, dssp=dssp,
            rmsf_prior=rmsf_prior, freq_scale_override=freq_scale_override,
            scale_cond=scale_cond, return_aux=return_aux,
        )

        if isinstance(trunk_out, dict):
            pred = trunk_out['pred']
        else:
            pred = trunk_out

        B, L, D = pred.shape
        C_in = self.trunk.in_channels
        K = self.trunk.top_k_freqs
        # Sanity: trunk output last dim = K * C_in.
        if D != K * C_in:
            raise RuntimeError(
                f'Trunk output last dim {D} != top_k_freqs*in_channels = {K*C_in}. '
                f'SpectralConvSlowBranch assumes the trunk flattens modes × channels.'
            )

        # Assemble slow-branch inputs.
        if win_pos is None:
            win_pos_in = torch.zeros(B, device=pred.device, dtype=pred.dtype)
        else:
            win_pos_in = win_pos.to(pred.dtype)
        if mask is not None:
            size_scalar = mask.float().sum(dim=1)
        else:
            size_scalar = torch.full((B,), float(L), device=pred.device, dtype=pred.dtype)

        delta_slow = self.slow_branch(
            native_coords=native_coords.to(pred.dtype),
            temp_norm=temp.to(pred.dtype),
            size_scalar=size_scalar.to(pred.dtype),
            win_pos=win_pos_in,
            res_type=res_type,
            dssp=dssp,
            rmsf_prior=rmsf_prior,
            mask=mask,
        )  # (B, L, K_slow, C_in)

        # Add to trunk's low-k modes. Trunk output is in raw coefficient
        # space (post-denormalisation), so the slow branch contributes
        # directly at the same scale.
        pred_kc = pred.view(B, L, K, C_in)
        # Avoid in-place on a view that may be a leaf tensor in autograd.
        delta_full = torch.zeros_like(pred_kc)
        delta_full[:, :, :self.K_slow, :] = delta_slow
        pred_kc = pred_kc + delta_full

        if mask is not None:
            pred_kc = pred_kc * mask.unsqueeze(-1).unsqueeze(-1).to(pred_kc.dtype)

        pred_out = pred_kc.view(B, L, D)

        if isinstance(trunk_out, dict):
            trunk_out['pred'] = pred_out
            trunk_out['delta_slow'] = delta_slow
            return trunk_out
        return pred_out
