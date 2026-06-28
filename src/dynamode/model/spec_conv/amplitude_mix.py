from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamode.model.spec_conv.block_mix import SpectralConvBlockMix
from dynamode.model.spec_conv.blocks import AuxSpectralTransformerBlock
from dynamode.model.modules import RotaryEmbedding, SmoothScalarEmbedding, WindowContextEmbedding


class LowKAmplitudeHead(nn.Module):
    """Predict multiplicative amplitude gains for a narrow low-k set.

    The head consumes the trunk's current low-k prediction for a small
    context window together with residue-level conditioning, then outputs
    log-gains whose zero initialisation preserves the trunk prediction at
    step 0. The caller applies those gains to the amplitudes of the
    selected target modes while preserving the trunk's predicted vector
    direction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 3,
        context_modes: int = 4,
        target_modes: int = 1,
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
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )

        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.context_modes = int(context_modes)
        self.target_modes = int(target_modes)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.use_seq_conditioning = bool(use_seq_conditioning)
        self.use_ss_conditioning = bool(use_ss_conditioning)
        self.use_rmsf_prior = bool(use_rmsf_prior)
        self.seq_embed_dim = int(seq_embed_dim) if self.use_seq_conditioning else 0
        self.ss_embed_dim = int(ss_embed_dim) if self.use_ss_conditioning else 0
        self.num_res_types = int(num_res_types)
        self.num_dssp_states = int(num_dssp_states)

        local_feat_dim = (
            self.context_modes * self.in_channels
            + self.cond_channels
            + self.seq_embed_dim
            + self.ss_embed_dim
            + (1 if self.use_rmsf_prior else 0)
        )
        self.local_proj = nn.Linear(local_feat_dim, self.d_model)

        if self.use_seq_conditioning:
            self.residue_embed = nn.Embedding(self.num_res_types, self.seq_embed_dim)
            nn.init.normal_(self.residue_embed.weight, std=0.02)
            self.seq_global_proj = nn.Linear(self.seq_embed_dim, self.d_model)
        if self.use_ss_conditioning:
            self.ss_local_proj = nn.Linear(self.num_dssp_states, self.ss_embed_dim)
            self.ss_global_proj = nn.Linear(self.ss_embed_dim, self.d_model)
        if self.use_rmsf_prior:
            self.rmsf_global_proj = nn.Linear(1, self.d_model)

        self.temp_embedder = SmoothScalarEmbedding(self.d_model)
        self.win_ctx_mlp = WindowContextEmbedding(self.d_model)
        self.rope = RotaryEmbedding(self.d_model // self.num_heads)
        self.blocks = nn.ModuleList([
            AuxSpectralTransformerBlock(
                hidden_size=self.d_model,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                dropout=attn_dropout,
                use_cross_attn=False,
                use_freq_coords=False,
            )
            for _ in range(int(depth))
        ])
        self.final_norm = nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(self.d_model, self.target_modes)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        m = mask.to(x.dtype).unsqueeze(-1)
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
        low_k_context: torch.Tensor,
        native_coords: torch.Tensor,
        temp_norm: torch.Tensor,
        size_scalar: torch.Tensor,
        win_pos: torch.Tensor,
        res_type: torch.Tensor | None = None,
        dssp: torch.Tensor | None = None,
        rmsf_prior: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = low_k_context.shape
        device = low_k_context.device
        dtype = low_k_context.dtype

        feats = [low_k_context, native_coords.to(dtype) / 10.0]

        seq_local = None
        if self.use_seq_conditioning:
            if res_type is None:
                seq_local = torch.zeros(B, L, self.seq_embed_dim, device=device, dtype=dtype)
            else:
                idx = res_type.argmax(dim=-1) if res_type.ndim == 3 else res_type
                idx = idx.long().clamp(0, self.num_res_types - 1)
                seq_local = self.residue_embed(idx).to(dtype)
            feats.append(seq_local)

        ss_local = None
        if self.use_ss_conditioning:
            if dssp is None:
                base = torch.zeros(B, L, self.num_dssp_states, device=device, dtype=dtype)
                base[..., -1] = 1.0
            else:
                if dssp.ndim == 2:
                    base = F.one_hot(
                        dssp.long().clamp(0, self.num_dssp_states - 1),
                        num_classes=self.num_dssp_states,
                    ).to(dtype)
                else:
                    base = dssp.to(dtype)
            ss_local = self.ss_local_proj(base)
            feats.append(ss_local)

        rmsf_local = None
        if self.use_rmsf_prior:
            rmsf_local = (
                torch.zeros(B, L, 1, device=device, dtype=dtype)
                if rmsf_prior is None
                else rmsf_prior.to(dtype).unsqueeze(-1)
            )
            feats.append(rmsf_local)

        local = torch.cat(feats, dim=-1)
        if mask is not None:
            local = local * mask.unsqueeze(-1).to(local.dtype)
        tokens = self.local_proj(local)

        seq_pooled = self._masked_mean(seq_local, mask) if seq_local is not None else None
        ss_pooled = self._masked_mean(ss_local, mask) if ss_local is not None else None
        rmsf_pooled = (
            self._masked_mean(rmsf_local, mask).squeeze(-1)
            if rmsf_local is not None else None
        )
        c = self._build_cond(
            temp_norm=temp_norm.to(dtype),
            size_scalar=size_scalar.to(dtype),
            win_pos=win_pos.to(dtype),
            seq_pooled=seq_pooled,
            ss_pooled=ss_pooled,
            rmsf_pooled=rmsf_pooled,
        )

        rope_freqs = self.rope(tokens)
        for block in self.blocks:
            tokens = block(tokens, c, rope_freqs, context=None, mask=mask)

        tokens = self.final_norm(tokens)
        log_gain = self.out_proj(tokens)
        if mask is not None:
            log_gain = log_gain * mask.unsqueeze(-1).to(log_gain.dtype)
        return log_gain


class SpectralConvBlockMixAmplitude(nn.Module):
    """SpecConv trunk plus a narrow low-k amplitude calibration head."""

    is_time_domain: bool = False

    def __init__(
        self,
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
        prediction_target: str = "v",
        is_dct: bool = True,
        use_hilbert: bool = False,
        use_hilbert_dct: bool = False,
        hilbert_mode: str = "every_block",
        use_rmsf_prior_gain: bool = False,
        use_low_k_correction_head: bool = False,
        low_k_correction_modes: int | str = 1,
        use_seq_conditioning: bool = False,
        seq_embed_dim: int = 16,
        use_ss_conditioning: bool = False,
        ss_embed_dim: int = 8,
        cond_dim: int = 512,
        prediction_distribution: str = "deterministic",
        distribution_logvar_min: float = -6.0,
        distribution_logvar_max: float = 2.0,
        band_edges: tuple[int, ...] | str | None = None,
        warm_start_from_dense: bool = True,
        amp_head_context_modes: int = 4,
        amp_head_target_modes: int = 1,
        amp_head_d_model: int = 128,
        amp_head_depth: int = 3,
        amp_head_num_heads: int = 4,
        amp_head_mlp_ratio: float = 4.0,
        amp_head_attn_dropout: float = 0.0,
        amp_head_use_rmsf_prior: bool = False,
        use_shake: bool = False,
        shake_n_iter: int = 20,
        shake_target: float = 3.8,
    ) -> None:
        super().__init__()
        self.use_shake = bool(use_shake)
        self.shake_n_iter = int(shake_n_iter)
        self.shake_target = float(shake_target)
        self._prediction_distribution = str(prediction_distribution)
        self.distribution_logvar_min = float(distribution_logvar_min)
        self.distribution_logvar_max = float(distribution_logvar_max)
        self.target_modes = int(amp_head_target_modes)
        self.context_modes = max(int(amp_head_context_modes), self.target_modes)
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)

        self.trunk = SpectralConvBlockMix(
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
            band_edges=band_edges,
            warm_start_from_dense=warm_start_from_dense,
        )
        self.amp_head = LowKAmplitudeHead(
            in_channels=in_channels,
            cond_channels=cond_channels,
            context_modes=self.context_modes,
            target_modes=self.target_modes,
            d_model=amp_head_d_model,
            depth=amp_head_depth,
            num_heads=amp_head_num_heads,
            mlp_ratio=amp_head_mlp_ratio,
            attn_dropout=amp_head_attn_dropout,
            use_seq_conditioning=use_seq_conditioning,
            use_ss_conditioning=use_ss_conditioning,
            seq_embed_dim=seq_embed_dim,
            ss_embed_dim=ss_embed_dim,
            use_rmsf_prior=amp_head_use_rmsf_prior,
        )

    @property
    def prediction_target(self) -> str:
        return self.trunk.prediction_target

    @property
    def prediction_distribution(self) -> str:
        return self._prediction_distribution

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
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        trunk_out = self.trunk(
            x=x,
            t=t,
            temp=temp,
            native_coords=native_coords,
            mask=mask,
            win_pos=win_pos,
            cond_drop_mask=cond_drop_mask,
            native_angles=native_angles,
            res_type=res_type,
            dssp=dssp,
            rmsf_prior=rmsf_prior,
            freq_scale_override=freq_scale_override,
            scale_cond=scale_cond,
            return_aux=return_aux,
        )
        pred = trunk_out["pred"] if isinstance(trunk_out, dict) else trunk_out

        B, L, D = pred.shape
        K = self.top_k_freqs
        C = self.in_channels
        pred_kc = pred.view(B, L, K, C)

        if win_pos is None:
            win_pos_in = torch.zeros(B, device=pred.device, dtype=pred.dtype)
        else:
            win_pos_in = win_pos.to(pred.dtype)
        if mask is not None:
            size_scalar = mask.float().sum(dim=1)
        else:
            size_scalar = torch.full((B,), float(L), device=pred.device, dtype=pred.dtype)

        low_k_context = pred_kc[:, :, :self.context_modes, :].reshape(B, L, self.context_modes * C)
        # The low-k amplitude head is small but its attention backward can be
        # numerically fragile in bf16. Keep this calibration path in fp32 while
        # allowing gradients to flow back into the trunk prediction.
        autocast_device = pred.device.type if pred.device.type in {"cuda", "cpu"} else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            log_gain = self.amp_head(
                low_k_context=low_k_context.float(),
                native_coords=native_coords.float(),
                temp_norm=temp.float(),
                size_scalar=size_scalar.float(),
                win_pos=win_pos_in.float(),
                res_type=res_type,
                dssp=dssp,
                rmsf_prior=rmsf_prior.float() if rmsf_prior is not None else None,
                mask=mask,
            )
        # Bound multiplicative calibration. The head is zero-initialised, but
        # early training can otherwise produce very large log-gains and
        # overflow the exponential in bf16/DDP runs.
        log_gain = torch.clamp(log_gain, min=-4.0, max=4.0)
        gain = torch.exp(log_gain).to(pred.dtype).unsqueeze(-1)

        target_vecs = pred_kc[:, :, :self.target_modes, :]
        # Direction-preserving amplitude scaling simplifies exactly to
        # multiplying the vector by a scalar gain. Avoid explicitly dividing by
        # the current amplitude: near-zero low-k vectors make that quotient's
        # gradient singular even though the forward expression cancels.
        calibrated = target_vecs * gain
        pred_kc = pred_kc.clone()
        pred_kc[:, :, :self.target_modes, :] = calibrated

        if mask is not None:
            pred_kc = pred_kc * mask.unsqueeze(-1).unsqueeze(-1).to(pred_kc.dtype)
        pred_out = pred_kc.view(B, L, D)

        if return_aux:
            out = dict(trunk_out) if isinstance(trunk_out, dict) else {"pred": pred}
            out["pred"] = pred_out
            out["amp_log_gain"] = log_gain
            out["amp_gain"] = gain.squeeze(-1)
            out["amp_modes"] = self.target_modes
            return out
        return pred_out
