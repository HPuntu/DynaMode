from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional, Literal
import torch
import torch.nn as nn

from src.models.models.transformer import SpectralDiT
from src.models.models.spectral_conv import SpectralConvDiT
from src.models.models.slow_branch import SpectralConvSlowBranch
from src.models.models.block_mix import (
    SpectralConvBlockMix,
    convert_dense_mixer_checkpoint,
    resolve_block_mix_band_edges,
)
from src.models.models.hierarchical import (
    HierarchicalSpectralDiffusion,
    convert_dense_mixer_checkpoint as convert_hierarchical_dense_checkpoint,
)
from src.models.models.cascade import CascadeSpectralDiffusion
from src.models.models.dual_branch import DualBranchSpectralDiffusion
from src.models.models.fno import FNO
from src.models.models.fno_manifold import FNOManifold
from src.models.models.hno import HNO
from src.models.models.v13 import FNO2, FNO2Bishop
from src.models.models.v14 import V14A, V14B, V14C, V14D
from src.models.models.v15 import V15
from src.models.models.v16 import V16
from src.models.models.v12 import (
    SpectralConvBlockMixAmplitude,
    SpectralConvBlockMixAmplitudeEGNN,
    SpectralConvBlockMixAmplitudeRefined,
    SpectralConvBlockMixSlowHybrid,
    SpectralConvBlockMixSlowHybridEGNN,
)
from src.models.models.v17 import (
    SpectralConvBlockMixAmplitudeBondGraphRefined,
    SpectralConvBlockMixAmplitudeSpectralGraphRefined,
)
from src.models.adapters import (
    make_spectral_batch_adapter,
    make_fno_batch_adapter,
    make_manifold_batch_adapter,
    identity_output,
)


# Configs
@dataclass(frozen=True)
class BaseDiffusionConfig:
    in_channels: int = 3
    cond_channels: int = 3
    depth: int = 12
    num_heads: int = 4
    prediction_target: Literal["v", "x_0", "noise"] = "v"

@dataclass(frozen=True)
class SpectralDiTConfig(BaseDiffusionConfig):
    top_k_freqs: int = 256
    freq_hidden_size: int = 8
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    freq_scale: Optional[torch.Tensor] = None
    conditioned_freq_scale: Optional[dict[str, Any]] = None
    cfg_dropout: bool = True
    is_dct: bool = True
    use_seq_conditioning: bool = False
    seq_embed_dim: int = 16
    use_ss_conditioning: bool = False
    ss_embed_dim: int = 8
    use_low_k_correction_head: bool = False
    low_k_correction_modes: int | str = 1
    cond_dim: int = 512

@dataclass(frozen=True)
class SpectralConvDiTConfig(BaseDiffusionConfig):
    top_k_freqs: int = 256
    freq_hidden_size: int = 8
    spectral_modes: int = 64
    attn_dropout: float = 0.0
    freq_scale: Optional[torch.Tensor] = None
    conditioned_freq_scale: Optional[dict[str, Any]] = None
    cfg_dropout: bool = True
    is_dct: bool = True
    use_hilbert: bool = False
    use_hilbert_dct: bool = False
    hilbert_mode: str = "every_block"
    use_rmsf_prior_gain: bool = False
    use_low_k_correction_head: bool = False
    low_k_correction_modes: int | str = 1
    use_seq_conditioning: bool = False
    seq_embed_dim: int = 16
    use_ss_conditioning: bool = False
    ss_embed_dim: int = 8
    cond_dim: int = 512


@dataclass(frozen=True)
class SpectralConvSlowBranchConfig(SpectralConvDiTConfig):
    # Slow-branch hyperparameters (v9).
    K_slow: int = 16
    slow_d_model: int = 128
    slow_depth: int = 3
    slow_num_heads: int = 4
    slow_mlp_ratio: float = 4.0
    slow_attn_dropout: float = 0.0
    slow_use_rmsf_prior: bool = False


@dataclass(frozen=True)
class SpectralConvBlockMixConfig(SpectralConvDiTConfig):
    # Block-diagonal mixer hyperparameters (v10).
    band_edges: Optional[tuple[int, ...] | str] = None  # defaults resolved at build time
    warm_start_from_dense: bool = True


@dataclass(frozen=True)
class HierarchicalSpectralConfig(SpectralConvDiTConfig):
    # Single-trunk hierarchical frequency sampler. ``frequency_ordering`` controls
    # group communication inside the spectral-conv mixer; optional input masking
    # is off by default until the training objective masks losses consistently.
    band_edges: Optional[tuple[int, ...] | str] = None
    frequency_ordering: str = "causal"
    warm_start_from_dense: bool = True
    mask_training_prob: float = 0.0
    mask_training_ordering: Optional[str] = None
    mask_training_mode: str = "next"
    use_target_group_conditioning: bool = True


@dataclass(frozen=True)
class CascadeSpectralConfig(SpectralConvDiTConfig):
    # Architectural cascade: DC transformer -> low-k spectral transformer -> spectral-conv bands.
    prediction_target: Literal["v", "x_0", "noise"] = "x_0"
    window_size: int = 256
    cascade_band_edges: Optional[tuple[int, ...] | str] = None
    cascade_context_mode: str = "idct_summary"
    cascade_detach_context: bool = True
    cascade_dc_depth: Optional[int] = None
    cascade_low_depth: Optional[int] = None
    cascade_high_depth: Optional[int] = None


@dataclass(frozen=True)
class SpectralConvBlockMixAmplitudeConfig(SpectralConvBlockMixConfig):
    amp_head_context_modes: int = 4
    amp_head_target_modes: int = 1
    amp_head_d_model: int = 128
    amp_head_depth: int = 3
    amp_head_num_heads: int = 4
    amp_head_mlp_ratio: float = 4.0
    amp_head_attn_dropout: float = 0.0
    amp_head_use_rmsf_prior: bool = False
    # Optional differentiable SHAKE applied to reconstructed CA coords.
    use_shake: bool = False
    shake_n_iter: int = 20
    shake_target: float = 3.8


@dataclass(frozen=True)
class SpectralConvBlockMixAmplitudeRefinedConfig(SpectralConvBlockMixAmplitudeConfig):
    # v12c: v12a + light CA-coord refiner + SHAKE (on by default).
    refiner_hidden: int = 32
    refiner_depth: int = 2
    refiner_kernel_size: int = 5
    refiner_max_delta: float | None = 0.5
    use_shake: bool = True


@dataclass(frozen=True)
class SpectralConvBlockMixAmplitudeSpectralGraphRefinedConfig(
    SpectralConvBlockMixAmplitudeRefinedConfig
):
    # v17a: v12c + native-graph correction of absolute low-mode spectra.
    use_spectral_graph_refiner: bool = True
    spectral_graph_refiner_modes: int = 17
    spectral_graph_refiner_hidden: int = 128
    spectral_graph_refiner_depth: int = 3
    spectral_graph_refiner_msg_hidden: int | None = None
    spectral_graph_refiner_sequence_window: int = 2
    spectral_graph_refiner_knn: int = 16
    spectral_graph_refiner_use_sequence_edges: bool = True
    spectral_graph_refiner_use_native_knn: bool = True
    spectral_graph_refiner_max_delta: float | None = 0.25


@dataclass(frozen=True)
class SpectralConvBlockMixAmplitudeBondGraphRefinedConfig(
    SpectralConvBlockMixAmplitudeRefinedConfig
):
    # v17b: v12c + native-graph correction of adjacent bond spectra.
    use_bond_spectral_graph_refiner: bool = True
    bond_spectral_graph_refiner_modes: int = 17
    bond_spectral_graph_refiner_hidden: int = 128
    bond_spectral_graph_refiner_depth: int = 3
    bond_spectral_graph_refiner_msg_hidden: int | None = None
    bond_spectral_graph_refiner_sequence_window: int = 2
    bond_spectral_graph_refiner_knn: int = 16
    bond_spectral_graph_refiner_use_sequence_edges: bool = True
    bond_spectral_graph_refiner_use_native_knn: bool = True
    bond_spectral_graph_refiner_max_delta: float | None = 0.25
    bond_spectral_graph_refiner_blend: float = 1.0


@dataclass(frozen=True)
class SpectralConvBlockMixSlowHybridConfig(SpectralConvBlockMixConfig):
    K_slow: int = 4
    slow_mode_start: int = 0
    slow_d_model: int = 128
    slow_depth: int = 3
    slow_num_heads: int = 4
    slow_mlp_ratio: float = 4.0
    slow_attn_dropout: float = 0.0
    slow_use_rmsf_prior: bool = False


@dataclass(frozen=True)
class SpectralConvBlockMixAmplitudeEGNNConfig(SpectralConvBlockMixAmplitudeConfig):
    # v12a_egnn: v12a + per-frame SE(3)-equivariant CA refiner (+ SHAKE).
    egnn_h_dim: int = 32
    egnn_hidden: int = 64
    egnn_depth: int = 3
    egnn_seq_window: int = 12
    egnn_max_len: int = 1024
    egnn_t_chunk: int = 64
    use_shake: bool = True


@dataclass(frozen=True)
class SpectralConvBlockMixSlowHybridEGNNConfig(SpectralConvBlockMixSlowHybridConfig):
    # v12b_egnn: v12b + per-frame SE(3)-equivariant CA refiner (+ SHAKE).
    egnn_h_dim: int = 32
    egnn_hidden: int = 64
    egnn_depth: int = 3
    egnn_seq_window: int = 12
    egnn_max_len: int = 1024
    egnn_t_chunk: int = 64
    use_shake: bool = True
    shake_n_iter: int = 20
    shake_target: float = 3.8


@dataclass(frozen=True)
class DualBranchConfig:
    # v11 dual-branch spectral diffusion: slow + fast branches connected by
    # one-way cross-attention. Distinct hyperparameter prefixes for each
    # branch; shared conditioning toggles are top-level.
    top_k_freqs: int = 256
    K_slow: int = 16
    in_channels: int = 3
    cond_channels: int = 3
    prediction_target: Literal["v", "x_0", "noise"] = "v"
    is_dct: bool = True
    freq_scale: Optional[torch.Tensor] = None
    conditioned_freq_scale: Optional[dict[str, Any]] = None
    cfg_dropout: bool = True
    # Slow branch.
    slow_d_model: int = 192
    slow_depth: int = 6
    slow_num_heads: int = 4
    slow_mlp_ratio: float = 4.0
    slow_attn_dropout: float = 0.0
    slow_use_rmsf_prior: bool = False
    slow_predicts_amplitude_only: bool = False
    # Fast branch.
    fast_freq_hidden_size: int = 8
    fast_depth: int = 12
    fast_num_heads: int = 4
    fast_spectral_modes: int = 64
    fast_attn_dropout: float = 0.0
    fast_band_edges: Optional[tuple[int, ...]] = None
    fast_cond_dim: int = 512
    fast_use_hilbert: bool = False
    fast_use_hilbert_dct: bool = False
    fast_hilbert_mode: str = "every_block"
    fast_use_rmsf_prior_gain: bool = False
    # Shared conditioning.
    use_seq_conditioning: bool = False
    seq_embed_dim: int = 16
    use_ss_conditioning: bool = False
    ss_embed_dim: int = 8


@dataclass(frozen=True)
class FNOConfig(BaseDiffusionConfig):
    window_size: int = 256
    hidden_per_time: int = 8
    spectral_modes: int = 64
    coord_scale: float = 5.0
    use_dropout: bool = True
    cond_dim: int = 512


@dataclass(frozen=True)
class FNOManifoldConfig(FNOConfig):
    in_channels: int = FNOManifold.latent_dim
    cond_channels: int = 3
    coord_scale: float = 1.0
    length_eps: float = 0.35
    anchor_scale: float = 10.0
    tangent_scale: float = 0.25
    length_logit_scale: float = 1.0
    use_exp_map: bool = True


@dataclass(frozen=True)
class HNOConfig(BaseDiffusionConfig):
    window_size: int = 256
    hidden_per_time: int = 8
    spectral_modes: int = 64
    coord_scale: float = 5.0
    use_dropout: bool = True
    cond_dim: int = 512


@dataclass(frozen=True)
class FNO2Config(BaseDiffusionConfig):
    window_size: int = 256
    hidden_per_time: int = 16
    spectral_modes: int = 32
    coord_scale: float = 5.0
    use_dropout: bool = True
    cond_dim: int = 512
    use_seq_conditioning: bool = True
    seq_embed_dim: int = 16
    use_ss_conditioning: bool = True
    ss_embed_dim: int = 8
    temporal_ablation_mode: Literal["normal", "off", "freq_noise"] = "normal"
    block_ablation_mode: Literal["normal", "no_mlp", "temporal_only"] = "normal"
    temporal_gate_init: float = 0.0


@dataclass(frozen=True)
class FNO2BishopConfig(FNO2Config):
    pass


@dataclass(frozen=True)
class V15Config(FNO2Config):
    spectral_modes: int = 4


@dataclass(frozen=True)
class V16Config(FNO2Config):
    spectral_modes: int = 4


@dataclass(frozen=True)
class V14Config(BaseDiffusionConfig):
    window_size: int = 256
    hidden_size: int = 192
    spectral_modes: int = 32
    coord_scale: float = 5.0
    use_dropout: bool = True
    cond_dim: int = 512
    use_seq_conditioning: bool = True
    seq_embed_dim: int = 16
    use_ss_conditioning: bool = True
    ss_embed_dim: int = 8
    jepa_latent_dim: int = 128


@dataclass(frozen=True)
class V14AConfig(V14Config):
    pass


@dataclass(frozen=True)
class V14BConfig(V14Config):
    pass


@dataclass(frozen=True)
class V14CConfig(V14Config):
    pass


@dataclass(frozen=True)
class V14DConfig(V14Config):
    pass


# Registry spec
@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_type: type
    build_fn: Callable[[Any], nn.Module]
    input_adapter_factory: Callable[[Any], Callable[[dict[str, Any]], dict[str, Any]]]
    output_adapter: Callable[[Any], Any]
    # Optional: translate a loaded state-dict before ``load_state_dict``.
    # Used by composite models to accept legacy (spec_conv / v8) checkpoints.
    state_dict_translator: Optional[
        Callable[[dict[str, torch.Tensor], Any], dict[str, torch.Tensor]]
    ] = None


def identity_input(batch: dict[str, Any]) -> dict[str, Any]:
    return batch


def identity_output(output: Any) -> Any:
    return output


# ----------------------------------------------------------------------
# Composite-model helpers.
# ----------------------------------------------------------------------
def _is_spec_conv_legacy_state(state: dict[str, torch.Tensor]) -> bool:
    '''True if ``state`` looks like a bare SpectralConvDiT checkpoint.

    Composite models (SpectralConvSlowBranch, SpectralConvBlockMix) wrap
    the trunk under ``self.trunk``, so their native keys start with
    ``trunk.``. Legacy v8 checkpoints have keys like ``blocks.0.*`` and
    ``final_proj.*`` without any ``trunk.`` prefix.
    '''
    sample = next(iter(state), None)
    if sample is None:
        return False
    return not any(k.startswith("trunk.") for k in state.keys())


def _prepend_trunk(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    '''Prepend ``trunk.`` to every key that doesn't already have it.'''
    return {
        (k if k.startswith("trunk.") else f"trunk.{k}"): v
        for k, v in state.items()
    }


def _slow_branch_translator(
    state: dict[str, torch.Tensor], cfg: Any
) -> dict[str, torch.Tensor]:
    '''Translate a v8 (SpectralConvDiT) state-dict for v9 composites.

    Prepends ``trunk.`` to every key. The ``slow_branch.*`` parameters are
    not present in v8 checkpoints — ``strict=False`` loading leaves them
    at their zero-init state, which is the intended warm-start.
    '''
    if not _is_spec_conv_legacy_state(state):
        return state
    return _prepend_trunk(state)


def _block_mix_translator(
    state: dict[str, torch.Tensor], cfg: Any
) -> dict[str, torch.Tensor]:
    '''Translate a v8 state-dict into v10 layout (block-diagonal mixers).

    Walks each block's dense ``freq_mixer`` parameters and rewrites them as
    ``freq_mixer.bands.{g}.*`` entries carrying the within-band
    sub-matrices. All other keys are passed through (with ``trunk.``
    prepended if needed).
    '''
    if not _is_spec_conv_legacy_state(state):
        return state
    # Determine band_edges and depth from the config.
    top_k_freqs = int(getattr(cfg, "top_k_freqs", 256))
    depth = int(getattr(cfg, "depth", 12))
    band_edges = resolve_block_mix_band_edges(
        getattr(cfg, "band_edges", None),
        top_k_freqs,
    )
    return convert_dense_mixer_checkpoint(
        state,
        band_edges=band_edges,
        top_k_freqs=top_k_freqs,
        depth=depth,
    )


def _hierarchical_translator(
    state: dict[str, torch.Tensor], cfg: Any
) -> dict[str, torch.Tensor]:
    '''Translate a v8 state-dict into the ordered group-mixer hierarchy.'''
    if not _is_spec_conv_legacy_state(state):
        return state
    top_k_freqs = int(getattr(cfg, "top_k_freqs", 256))
    depth = int(getattr(cfg, "depth", 12))
    band_edges = getattr(cfg, "band_edges", None)
    ordering = getattr(cfg, "frequency_ordering", "causal")
    return convert_hierarchical_dense_checkpoint(
        state,
        band_edges=band_edges,
        top_k_freqs=top_k_freqs,
        depth=depth,
        ordering=ordering,
    )


MODEL_REGISTRY = {
    "spectral_dit": ModelSpec(
        name="spectral_dit",
        config_type=SpectralDiTConfig,
        build_fn=lambda cfg: SpectralDiT(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "spectral_dit_low_k": ModelSpec(
        name="spectral_dit_low_k",
        config_type=SpectralDiTConfig,
        build_fn=lambda cfg: SpectralDiT(
            **{**asdict(cfg), "use_low_k_correction_head": True}
        ),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "spectral_conv_dit": ModelSpec(
        name="spectral_conv_dit",
        config_type=SpectralConvDiTConfig,
        build_fn=lambda cfg: SpectralConvDiT(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    # v9: SpectralConvDiT trunk + additive SlowBranch head.
    "spectral_conv_slow_branch": ModelSpec(
        name="spectral_conv_slow_branch",
        config_type=SpectralConvSlowBranchConfig,
        build_fn=lambda cfg: SpectralConvSlowBranch(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
        state_dict_translator=_slow_branch_translator,
    ),
    # v10: SpectralConvDiT trunk with block-diagonal freq mixers.
    "spectral_conv_block_mix": ModelSpec(
        name="spectral_conv_block_mix",
        config_type=SpectralConvBlockMixConfig,
        build_fn=lambda cfg: SpectralConvBlockMix(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
        state_dict_translator=_block_mix_translator,
    ),
    "hierarchical": ModelSpec(
        name="hierarchical",
        config_type=HierarchicalSpectralConfig,
        build_fn=lambda cfg: HierarchicalSpectralDiffusion(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
            include_frequency_masks=True,
        ),
        output_adapter=identity_output,
        state_dict_translator=_hierarchical_translator,
    ),
    "cascade": ModelSpec(
        name="cascade",
        config_type=CascadeSpectralConfig,
        build_fn=lambda cfg: CascadeSpectralDiffusion(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "spectral_conv_block_mix_amplitude": ModelSpec(
        name="spectral_conv_block_mix_amplitude",
        config_type=SpectralConvBlockMixAmplitudeConfig,
        build_fn=lambda cfg: SpectralConvBlockMixAmplitude(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "v12c": ModelSpec(
        name="v12c",
        config_type=SpectralConvBlockMixAmplitudeRefinedConfig,
        build_fn=lambda cfg: SpectralConvBlockMixAmplitudeRefined(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "v17a": ModelSpec(
        name="v17a",
        config_type=SpectralConvBlockMixAmplitudeSpectralGraphRefinedConfig,
        build_fn=lambda cfg: SpectralConvBlockMixAmplitudeSpectralGraphRefined(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "v17b": ModelSpec(
        name="v17b",
        config_type=SpectralConvBlockMixAmplitudeBondGraphRefinedConfig,
        build_fn=lambda cfg: SpectralConvBlockMixAmplitudeBondGraphRefined(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    # v12d: v12a or v12b + per-frame SE(3)-equivariant EGNN CA refiner.
    "v12a_egnn": ModelSpec(
        name="v12a_egnn",
        config_type=SpectralConvBlockMixAmplitudeEGNNConfig,
        build_fn=lambda cfg: SpectralConvBlockMixAmplitudeEGNN(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "v12b_egnn": ModelSpec(
        name="v12b_egnn",
        config_type=SpectralConvBlockMixSlowHybridEGNNConfig,
        build_fn=lambda cfg: SpectralConvBlockMixSlowHybridEGNN(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "spectral_conv_block_mix_slow_hybrid": ModelSpec(
        name="spectral_conv_block_mix_slow_hybrid",
        config_type=SpectralConvBlockMixSlowHybridConfig,
        build_fn=lambda cfg: SpectralConvBlockMixSlowHybrid(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    # v11: fully-parallel dual-branch spectral diffusion.
    "dual_branch": ModelSpec(
        name="dual_branch",
        config_type=DualBranchConfig,
        build_fn=lambda cfg: DualBranchSpectralDiffusion(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_spectral_batch_adapter(
            top_k_freqs=cfg.top_k_freqs,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
            is_dct=cfg.is_dct,
            conditioned_freq_scale=cfg.conditioned_freq_scale,
        ),
        output_adapter=identity_output,
    ),
    "fno": ModelSpec(
        name="fno",
        config_type=FNOConfig,
        build_fn=lambda cfg: FNO(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "fno_manifold": ModelSpec(
        name="fno_manifold",
        config_type=FNOManifoldConfig,
        build_fn=lambda cfg: FNOManifold(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_manifold_batch_adapter(
            window_size=cfg.window_size,
            latent_dim=FNOManifold.latent_dim,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "hno": ModelSpec(
        name="hno",
        config_type=HNOConfig,
        build_fn=lambda cfg: HNO(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "fno2": ModelSpec(
        name="fno2",
        config_type=FNO2Config,
        build_fn=lambda cfg: FNO2(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "fno2_bishop": ModelSpec(
        name="fno2_bishop",
        config_type=FNO2BishopConfig,
        build_fn=lambda cfg: FNO2Bishop(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "v15": ModelSpec(
        name="v15",
        config_type=V15Config,
        build_fn=lambda cfg: V15(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "v16": ModelSpec(
        name="v16",
        config_type=V16Config,
        build_fn=lambda cfg: V16(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_manifold_batch_adapter(
            window_size=cfg.window_size,
            latent_dim=V16.latent_dim,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "v14a": ModelSpec(
        name="v14a",
        config_type=V14AConfig,
        build_fn=lambda cfg: V14A(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "v14b": ModelSpec(
        name="v14b",
        config_type=V14BConfig,
        build_fn=lambda cfg: V14B(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "v14c": ModelSpec(
        name="v14c",
        config_type=V14CConfig,
        build_fn=lambda cfg: V14C(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
    "v14d": ModelSpec(
        name="v14d",
        config_type=V14DConfig,
        build_fn=lambda cfg: V14D(**asdict(cfg)),
        input_adapter_factory=lambda cfg: make_fno_batch_adapter(
            window_size=cfg.window_size,
            in_channels=cfg.in_channels,
            cond_channels=cfg.cond_channels,
        ),
        output_adapter=identity_output,
    ),
}
