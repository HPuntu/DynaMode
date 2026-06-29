'''
Spectral convolution block-diagonal mixing trunk plus a narrow low-k
amplitude calibration head. The dense nn.Linear(K, K) freq_mixer inside
every :class:`SpectralConvBlock` is replaced by a block-diagonal operator
whose blocks align with physically motivated frequency bands.

Default band_edges = (0, 1, 9, 33, 129, 256) for K = 256 keeps the DC mode
separate from the lowest non-zero DCT modes:

- [0, 1): DC / mean native-relative displacement.
- [1, 9): lowest trajectory-shape modes, RMSF-like flexibility signal.
- [9, 33): secondary-structure fluctuations.
- [33, 129): backbone vibrations.
- [129, 256): thermal jitter.
'''


from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamode.model.frequency_bands import parse_band_edges
from dynamode.model.modules import (
    RotaryEmbedding,
    SmoothScalarEmbedding,
    WindowContextEmbedding,
)
from dynamode.model.spec_conv.blocks import AuxSpectralTransformerBlock
from dynamode.model.spec_conv.spectral_conv import SpectralConvBlock, SpectralConvDiT



def default_block_mix_band_edges(top_k_freqs: int) -> tuple[int, ...]:
    '''
    Default DC-separated spectral block-mix bands.

    For the canonical K = 256 case this returns
    (0, 1, 9, 33, 129, 256), i.e. DC | 1-8 | 9-32 |
    33-128 | 129+.  For smaller K values the same absolute cutpoints
    are clipped and empty bands are removed.
    '''
    K = int(top_k_freqs)
    if K <= 0:
        raise ValueError(f'top_k_freqs must be positive, got {top_k_freqs}')
    candidates = (0, 1, 9, 33, 129, K)
    edges: list[int] = []
    for edge in candidates:
        edge = min(max(int(edge), 0), K)
        if not edges or edge > edges[-1]:
            edges.append(edge)
    if edges[-1] != K:
        edges.append(K)
    return _validate_band_edges(edges, K)


def resolve_block_mix_band_edges(
    band_edges: Iterable[int] | str | None,
    top_k_freqs: int,
) -> tuple[int, ...]:
    '''
    Resolve block-mix bands from explicit edges or a named scheme.

    None and "block_mix" use the current DC-separated default. Explicit
    edge tuples/strings are accepted for experiments with different DCT
    truncation, but legacy named partitions are intentionally not part of
    the public model surface.
    '''
    if band_edges is None:
        return default_block_mix_band_edges(top_k_freqs)
    return parse_band_edges(band_edges, top_k_freqs, default_scheme='block_mix')


def _validate_band_edges(band_edges: Iterable[int], n_freqs: int) -> tuple[int, ...]:
    '''
    Sanity-check and normalise a band-edge specification.

    Raises ValueError if the edges do not start at 0, end at
    n_freqs, or fail to be strictly increasing.
    '''
    if isinstance(band_edges, str):
        return parse_band_edges(band_edges, n_freqs, default_scheme='block_mix')
    edges = tuple(int(e) for e in band_edges)
    if len(edges) < 2:
        raise ValueError(f'band_edges must have at least two entries, got {edges!r}')
    if edges[0] != 0:
        raise ValueError(f'band_edges must start at 0, got {edges!r}')
    if edges[-1] != int(n_freqs):
        raise ValueError(
            f'band_edges last entry must equal n_freqs={n_freqs}, got {edges!r}'
        )
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            raise ValueError(f'band_edges must be strictly increasing, got {edges!r}')
    return edges


class BlockDiagonalMixer(nn.Module):
    '''
    Block-diagonal linear mixer operating on the last axis.

    For an input of shape (..., K) and band edges [b_0 = 0, b_1,
    ..., b_G = K], the module applies an independent dense
    nn.Linear(b_{i+1} - b_i, b_{i+1} - b_i) to each slice
    x[..., b_i : b_{i+1}] and concatenates the outputs.

    Relative to nn.Linear(K, K) this is a structural constraint:
    outputs in band i depend only on inputs in band i. The
    number of parameters is Σ_i (b_{i+1} - b_i)^2 versus K^2.

    band_edges = Increasing sequence (0, b_1, ..., K) defining
                 the block boundaries.
    bias = Whether each block's linear has a bias term.

    The implementation is fully-vectorised over the batch dimensions and
    adds no significant latency over the dense mixer for typical K.
    '''

    def __init__(self, band_edges: Iterable[int], bias: bool = True) -> None:
        super().__init__()
        edges = tuple(int(e) for e in band_edges)
        if len(edges) < 2 or edges[0] != 0 or any(
            b <= a for a, b in zip(edges[:-1], edges[1:])
        ):
            raise ValueError(
                f'band_edges must be strictly increasing from 0, got {edges!r}'
            )
        self.band_edges = edges
        self.n_freqs = int(edges[-1])
        self.bias = bool(bias)
        self.bands = nn.ModuleList([
            nn.Linear(b1 - b0, b1 - b0, bias=self.bias)
            for b0, b1 in zip(edges[:-1], edges[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply each band's linear to the corresponding slice of x.
        '''
        if x.shape[-1] != self.n_freqs:
            raise RuntimeError(
                f'BlockDiagonalMixer expected last dim {self.n_freqs}, '
                f'got {x.shape[-1]}'
            )
        slices = []
        for band, b0, b1 in zip(
            self.bands, self.band_edges[:-1], self.band_edges[1:]
        ):
            slices.append(band(x[..., b0:b1]))
        return torch.cat(slices, dim=-1)


def _install_block_diagonal_mixers(
    trunk: SpectralConvDiT,
    band_edges: Iterable[int],
) -> SpectralConvDiT:
    '''
    Replace each block's freq_mixer with a :class:`BlockDiagonalMixer`.

    trunk = A constructed :class:`SpectralConvDiT`.
    band_edges = The block boundaries.
    '''
    edges = _validate_band_edges(band_edges, trunk.top_k_freqs)
    for block in trunk.blocks:
        if not isinstance(block, SpectralConvBlock):
            continue
        dense = block.freq_mixer
        if isinstance(dense, BlockDiagonalMixer):
            continue
        if not isinstance(dense, nn.Linear):
            raise TypeError(
                f'Expected SpectralConvBlock.freq_mixer to be nn.Linear, '
                f'got {type(dense).__name__}'
            )
        new_mixer = BlockDiagonalMixer(edges, bias=dense.bias is not None)
        # Keep on the same device / dtype as the rest of the block.
        ref_param = next(block.parameters())
        new_mixer = new_mixer.to(device=ref_param.device, dtype=ref_param.dtype)
        block.freq_mixer = new_mixer
    return trunk


class SpectralConvBlockMix(nn.Module):
    '''
    SpectralConvDiT trunk with block-diagonal mixers.

    Architecturally identical to :class:`SpectralConvDiT` except that
    every block's freq_mixer is a :class:`BlockDiagonalMixer`. The
    band structure is specified by band_edges and is a strict
    constraint on cross-mode mixing inside the trunk.

    The composite's forward signature matches the trunk's exactly, so
    the unified pipeline (batch adapter, training loop, inference) sees
    no difference.

    band_edges = Block boundaries. Must start at 0 and end at
                 top_k_freqs. Defaults to
                 the DC-separated DC | 1-8 | 9-32 | 33-128 | 129+
                 partition when left at None.
    '''

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
        prediction_target: str = 'v',
        is_dct: bool = True,
        use_hilbert: bool = False,
        use_hilbert_dct: bool = False,
        hilbert_mode: str = 'every_block',
        use_rmsf_prior_gain: bool = False,
        use_low_k_correction_head: bool = False,
        low_k_correction_modes: int | str = 1,
        use_seq_conditioning: bool = False,
        seq_embed_dim: int = 16,
        use_ss_conditioning: bool = False,
        ss_embed_dim: int = 8,
        cond_dim: int = 512,
        band_edges: Iterable[int] | str | None = None,
    ) -> None:
        super().__init__()
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

        self.band_edges = resolve_block_mix_band_edges(band_edges, top_k_freqs)

        _install_block_diagonal_mixers(self.trunk, self.band_edges)

    @staticmethod
    def _default_band_edges(top_k_freqs: int) -> tuple[int, ...]:
        '''Default DC-separated band structure for arbitrary top_k_freqs.'''
        return default_block_mix_band_edges(top_k_freqs)

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
    def forward(self, *args, **kwargs):
        '''Pass-through to the trunk (mixer swap is transparent).'''
        return self.trunk(*args, **kwargs)


class LowKAmplitudeHead(nn.Module):
    '''
    Predict multiplicative amplitude gains for a narrow low-k set.

    The head consumes the trunk's current low-k prediction for a small
    context window together with residue-level conditioning, then outputs
    log-gains whose zero initialisation preserves the trunk prediction at
    step 0. The caller applies those gains to the amplitudes of the
    selected target modes while preserving the trunk's predicted vector
    direction.
    '''

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
        c = self.temp_embedder(temp_norm) + self.win_ctx_mlp(
            win_pos, temp_k, size_scalar
        )
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
                seq_local = torch.zeros(
                    B, L, self.seq_embed_dim, device=device, dtype=dtype
                )
            else:
                idx = res_type.argmax(dim=-1) if res_type.ndim == 3 else res_type
                idx = idx.long().clamp(0, self.num_res_types - 1)
                seq_local = self.residue_embed(idx).to(dtype)
            feats.append(seq_local)

        ss_local = None
        if self.use_ss_conditioning:
            if dssp is None:
                base = torch.zeros(
                    B, L, self.num_dssp_states, device=device, dtype=dtype
                )
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

        seq_pooled = (
            self._masked_mean(seq_local, mask) if seq_local is not None else None
        )
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
    '''SpecConv block-diagonal trunk plus low-k amplitude calibration head.'''

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
        band_edges: tuple[int, ...] | str | None = None,
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
            size_scalar = torch.full(
                (B,), float(L), device=pred.device, dtype=pred.dtype
            )

        low_k_context = pred_kc[:, :, :self.context_modes, :].reshape(
            B, L, self.context_modes * C
        )
        # The low-k amplitude head is small but its attention backward can be
        # numerically fragile in bf16. Keep this calibration path in fp32 while
        # allowing gradients to flow back into the trunk prediction.
        autocast_device = (
            pred.device.type if pred.device.type in {"cuda", "cpu"} else "cpu"
        )
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


__all__ = [
    "BlockDiagonalMixer",
    "LowKAmplitudeHead",
    "SpectralConvBlockMix",
    "SpectralConvBlockMixAmplitude",
    "default_block_mix_band_edges",
    "resolve_block_mix_band_edges",
]
