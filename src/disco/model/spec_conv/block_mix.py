'''Block-diagonal cross-frequency mixer + composite model.

This module implements v10 of the roadmap: the dense ``nn.Linear(K, K)``
``freq_mixer`` inside every :class:`SpectralConvBlock` is replaced by a
block-diagonal operator whose blocks align with physically-motivated
frequency bands. This constrains cross-mode mixing to within each band
and prevents spurious energy leakage between slow / secondary / backbone
/ thermal regimes through the trunk.

The module is deliberately modular:

- :class:`BlockDiagonalMixer` is a drop-in replacement for
  ``nn.Linear(K, K)`` and can be swapped into any architecture that uses
  a per-mode linear cross-mixer.
- :class:`SpectralConvBlockMix` is the composite model for v10: it
  instantiates a :class:`SpectralConvDiT` trunk and replaces each
  block's ``freq_mixer`` with a :class:`BlockDiagonalMixer`.
- :func:`convert_dense_mixer_checkpoint` is an offline helper to
  translate a v8 (dense-mixer) checkpoint into a v10 (block-diagonal)
  state-dict by copying within-band sub-matrices from the dense weights.

The same :class:`BlockDiagonalMixer` is reused by v11 inside the fast
branch's trunk, so no code duplication is required when we promote the
architecture further.

Band-edge design
----------------
Default ``band_edges = (0, 1, 9, 33, 129, 256)`` for ``K = 256`` keeps
the DC mode separate from the lowest non-zero DCT modes:

- ``[0, 1)``: DC / mean native-relative displacement.
- ``[1, 9)``: lowest trajectory-shape modes, RMSF-like flexibility signal.
- ``[9, 33)``: secondary-structure fluctuations.
- ``[33, 129)``: backbone vibrations.
- ``[129, 256)``: thermal jitter.

Pass a different tuple if the training window or DCT truncation differs;
the last edge must equal ``top_k_freqs``.
'''

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from src.models.mask import parse_band_edges
from src.models.models.spectral_conv import SpectralConvBlock, SpectralConvDiT


def default_block_mix_band_edges(top_k_freqs: int) -> tuple[int, ...]:
    '''Default DC-separated spectral block-mix bands.

    For the canonical ``K = 256`` case this returns
    ``(0, 1, 9, 33, 129, 256)``, i.e. ``DC | 1-8 | 9-32 |
    33-128 | 129+``.  For smaller ``K`` values the same absolute cutpoints
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


def legacy_block_mix_band_edges(top_k_freqs: int) -> tuple[int, ...]:
    '''Legacy spectral block-mix bands used before DC was split out.

    For the canonical ``K = 256`` case this returns ``(0, 8, 32, 128,
    256)``, i.e. ``0-7 | 8-31 | 32-127 | 128+``. Use this when loading
    older block-mix/v12 checkpoints trained before the DC-separated
    ``DC | 1-8 | 9-32 | 33-128 | 129+`` default.
    '''
    K = int(top_k_freqs)
    if K <= 0:
        raise ValueError(f'top_k_freqs must be positive, got {top_k_freqs}')
    candidates = (0, 8, 32, 128, K)
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
    '''Resolve block-mix bands from explicit edges or a named scheme.

    ``None`` and ``"block_mix"`` use the current DC-separated default.
    ``"legacy"`` / ``"block_mix_legacy"`` restore the older 4-band
    partition expected by pre-DC-split checkpoints.
    '''
    if band_edges is None:
        return default_block_mix_band_edges(top_k_freqs)
    return parse_band_edges(band_edges, top_k_freqs, default_scheme='block_mix')


def _validate_band_edges(band_edges: Iterable[int], n_freqs: int) -> tuple[int, ...]:
    '''Sanity-check and normalise a band-edge specification.

    Raises ``ValueError`` if the edges do not start at 0, end at
    ``n_freqs``, or fail to be strictly increasing.
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
    '''Block-diagonal linear mixer operating on the last axis.

    For an input of shape ``(..., K)`` and band edges ``[b_0 = 0, b_1,
    ..., b_G = K]``, the module applies an independent dense
    ``nn.Linear(b_{i+1} - b_i, b_{i+1} - b_i)`` to each slice
    ``x[..., b_i : b_{i+1}]`` and concatenates the outputs.

    Relative to ``nn.Linear(K, K)`` this is a structural constraint:
    outputs in band ``i`` depend only on inputs in band ``i``. The
    number of parameters is ``Σ_i (b_{i+1} - b_i)^2`` versus ``K^2``.

    Args:
        band_edges: Increasing sequence ``(0, b_1, ..., K)`` defining
            the block boundaries.
        bias: Whether each block's linear has a bias term.

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
        '''Apply each band's linear to the corresponding slice of ``x``.

        Args:
            x: Input tensor of shape ``(..., n_freqs)``.

        Returns:
            Same-shape tensor.
        '''
        if x.shape[-1] != self.n_freqs:
            raise RuntimeError(
                f'BlockDiagonalMixer expected last dim {self.n_freqs}, got {x.shape[-1]}'
            )
        slices = []
        for band, b0, b1 in zip(
            self.bands, self.band_edges[:-1], self.band_edges[1:]
        ):
            slices.append(band(x[..., b0:b1]))
        return torch.cat(slices, dim=-1)

    @classmethod
    def from_dense(
        cls,
        dense: nn.Linear,
        band_edges: Iterable[int],
    ) -> 'BlockDiagonalMixer':
        '''Warm-start construction from a dense ``nn.Linear(K, K)``.

        Copies the within-band sub-matrices of ``dense.weight`` into each
        of the new block linears. Off-band weights are discarded — this
        is the intentional constraint of the v10 architecture.

        Args:
            dense: A trained or freshly-initialised dense mixer.
            band_edges: Same sequence passed to the constructor.

        Returns:
            A new :class:`BlockDiagonalMixer` whose within-band weights
            match ``dense`` exactly.
        '''
        K = dense.in_features
        if dense.out_features != K:
            raise ValueError(
                f'dense must be square, got in={K}, out={dense.out_features}'
            )
        edges = _validate_band_edges(band_edges, K)
        has_bias = dense.bias is not None
        mixer = cls(edges, bias=has_bias)
        with torch.no_grad():
            W = dense.weight.detach()  # (K_out, K_in)
            b = dense.bias.detach() if has_bias else None
            for i, (b0, b1) in enumerate(zip(edges[:-1], edges[1:])):
                mixer.bands[i].weight.data.copy_(W[b0:b1, b0:b1])
                if b is not None:
                    mixer.bands[i].bias.data.copy_(b[b0:b1])
        return mixer


def replace_freq_mixers(
    trunk: SpectralConvDiT,
    band_edges: Iterable[int],
    warm_start_from_dense: bool = True,
) -> SpectralConvDiT:
    '''Replace each block's ``freq_mixer`` with a :class:`BlockDiagonalMixer`.

    When ``warm_start_from_dense`` is True (default), the existing dense
    mixer's within-band weights are copied into the new module. This is
    the correct behaviour when the trunk has already loaded v8
    checkpoint weights.

    When False, a fresh random :class:`BlockDiagonalMixer` is installed.
    This is appropriate when training from scratch.

    Args:
        trunk: A constructed :class:`SpectralConvDiT`.
        band_edges: The block boundaries.
        warm_start_from_dense: Copy within-band sub-matrices from the
            existing dense mixer.

    Returns:
        The same ``trunk`` object, modified in place.
    '''
    edges = _validate_band_edges(band_edges, trunk.top_k_freqs)
    for block in trunk.blocks:
        if not isinstance(block, SpectralConvBlock):
            continue
        dense = block.freq_mixer
        if not isinstance(dense, nn.Linear):
            # Already swapped — skip.
            continue
        if warm_start_from_dense:
            new_mixer = BlockDiagonalMixer.from_dense(dense, edges)
        else:
            new_mixer = BlockDiagonalMixer(
                edges, bias=dense.bias is not None
            )
        # Keep on the same device / dtype as the rest of the block.
        ref_param = next(block.parameters())
        new_mixer = new_mixer.to(device=ref_param.device, dtype=ref_param.dtype)
        block.freq_mixer = new_mixer
    return trunk


class SpectralConvBlockMix(nn.Module):
    '''Composite model: SpectralConvDiT trunk with block-diagonal mixers.

    Architecturally identical to :class:`SpectralConvDiT` except that
    every block's ``freq_mixer`` is a :class:`BlockDiagonalMixer`. The
    band structure is specified by ``band_edges`` and is a strict
    constraint on cross-mode mixing inside the trunk.

    The composite's forward signature matches the trunk's exactly, so
    the unified pipeline (batch adapter, training loop, inference) sees
    no difference.

    Args:
        band_edges: Block boundaries. Must start at 0 and end at
            ``top_k_freqs``. Defaults to
            the DC-separated ``DC | 1-8 | 9-32 | 33-128 | 129+``
            partition when left at ``None``.
        warm_start_from_dense: If True (default), the freshly-built
            dense mixers are immediately copied into the new
            :class:`BlockDiagonalMixer` instances inside ``__init__``.
            For training from scratch this has no meaningful effect
            (both start random), but it guarantees that loading a
            v8 / v9 checkpoint via the :func:`convert_dense_mixer_checkpoint`
            helper will warm-start cleanly.

    All other kwargs mirror :class:`SpectralConvDiT.__init__`.
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
        warm_start_from_dense: bool = True,
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

        replace_freq_mixers(
            self.trunk,
            self.band_edges,
            warm_start_from_dense=warm_start_from_dense,
        )

    @staticmethod
    def _default_band_edges(top_k_freqs: int) -> tuple[int, ...]:
        '''Default DC-separated band structure for arbitrary ``top_k_freqs``.'''
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


# ----------------------------------------------------------------------
# Offline checkpoint converter.
# ----------------------------------------------------------------------
def convert_dense_mixer_checkpoint(
    src_state: dict[str, torch.Tensor],
    band_edges: Iterable[int] | str,
    top_k_freqs: int,
    depth: int,
) -> dict[str, torch.Tensor]:
    '''Translate a dense-mixer state-dict into a block-diagonal layout.

    The v8 trunk stores each dense mixer as
    ``blocks.{i}.freq_mixer.{weight,bias}``. After the trunk is wrapped in
    :class:`SpectralConvBlockMix`, the same slot becomes
    ``trunk.blocks.{i}.freq_mixer.bands.{g}.{weight,bias}`` where ``g``
    indexes the band.

    This helper reads the dense keys out of ``src_state``, copies the
    within-band sub-matrices into the new keys, and returns a new state
    dict suitable for ``SpectralConvBlockMix.load_state_dict(..., strict=False)``.

    All non-mixer keys are re-emitted under a ``trunk.`` prefix so they
    match the wrapped module's parameter names. Keys that already start
    with ``trunk.`` are passed through unchanged.

    Args:
        src_state: The state-dict loaded from a v8 / v9 checkpoint.
        band_edges: Band boundaries to apply.
        top_k_freqs: Used to validate ``band_edges``.
        depth: Number of transformer blocks in the trunk.

    Returns:
        A new state-dict with dense mixer keys replaced by block-diagonal
        ones, under the ``trunk.`` prefix expected by
        :class:`SpectralConvBlockMix`.
    '''
    edges = resolve_block_mix_band_edges(band_edges, top_k_freqs)
    widths = [b1 - b0 for b0, b1 in zip(edges[:-1], edges[1:])]
    starts = [b0 for b0 in edges[:-1]]

    out: dict[str, torch.Tensor] = {}
    dense_keys_handled: set[str] = set()

    for block_idx in range(depth):
        w_key_src = f'blocks.{block_idx}.freq_mixer.weight'
        b_key_src = f'blocks.{block_idx}.freq_mixer.bias'
        # Allow either the bare or the already-wrapped form as input.
        alt_w = f'trunk.blocks.{block_idx}.freq_mixer.weight'
        alt_b = f'trunk.blocks.{block_idx}.freq_mixer.bias'
        w = src_state.get(w_key_src, src_state.get(alt_w))
        b = src_state.get(b_key_src, src_state.get(alt_b))
        if w is None:
            # Block has no dense mixer in the source — likely already
            # converted or missing entirely. Skip.
            continue
        dense_keys_handled.update({w_key_src, b_key_src, alt_w, alt_b})

        for g, (start, width) in enumerate(zip(starts, widths)):
            new_w_key = f'trunk.blocks.{block_idx}.freq_mixer.bands.{g}.weight'
            new_b_key = f'trunk.blocks.{block_idx}.freq_mixer.bands.{g}.bias'
            out[new_w_key] = w[start : start + width, start : start + width].clone()
            if b is not None:
                out[new_b_key] = b[start : start + width].clone()

    # Copy everything else through, prepending ``trunk.`` where absent.
    for k, v in src_state.items():
        if k in dense_keys_handled:
            continue
        out_key = k if k.startswith('trunk.') else f'trunk.{k}'
        out[out_key] = v

    return out
