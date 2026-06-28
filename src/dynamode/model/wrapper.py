'''
Provides a single point of call API for model loading. Uses registry configs and registers
to direct to respective models and default configs.

Supplies model registry for the supported DynaMode architectures.

- spectral_dit_low_k: the transformer baseline with the additive low-k
  correction head enabled by config.
- spectral_conv_block_mix_amplitude: the block-mix SpecConv trunk with
  differentiable SHAKE and a low-k amplitude calibration head.
'''


from __future__ import annotations
from dataclasses import asdict, dataclass
import inspect
from typing import Any, Callable, Literal
import torch
import torch.nn as nn

from dynamode.model.adapters import make_spectral_batch_adapter
from dynamode.model.spec_conv.amplitude_mix import SpectralConvBlockMixAmplitude
from dynamode.model.transformer.transformer import SpectralDiT



@dataclass(frozen=True)
class BaseDiffusionConfig:
    in_channels: int = 3
    cond_channels: int = 3
    depth: int = 12
    num_heads: int = 4
    prediction_target: Literal["v", "x_0", "noise"] = "x_0"


@dataclass(frozen=True)
class SpectralDiTConfig(BaseDiffusionConfig):
    top_k_freqs: int = 256
    freq_hidden_size: int = 8
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    freq_scale: torch.Tensor | None = None
    conditioned_freq_scale: dict[str, Any] | None = None
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
class SpectralConvBlockMixAmplitudeConfig(BaseDiffusionConfig):
    top_k_freqs: int = 256
    freq_hidden_size: int = 8
    spectral_modes: int = 64
    attn_dropout: float = 0.0
    freq_scale: torch.Tensor | None = None
    conditioned_freq_scale: dict[str, Any] | None = None
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
    band_edges: tuple[int, ...] | str | None = None
    warm_start_from_dense: bool = True
    amp_head_context_modes: int = 4
    amp_head_target_modes: int = 1
    amp_head_d_model: int = 128
    amp_head_depth: int = 3
    amp_head_num_heads: int = 4
    amp_head_mlp_ratio: float = 4.0
    amp_head_attn_dropout: float = 0.0
    amp_head_use_rmsf_prior: bool = False
    use_shake: bool = False
    shake_n_iter: int = 20
    shake_target: float = 3.8
    prediction_distribution: str = "deterministic"
    distribution_logvar_min: float = -6.0
    distribution_logvar_max: float = 2.0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_type: type
    build_fn: Callable[[Any], nn.Module]
    input_adapter_factory: Callable[[Any], Callable[[dict[str, Any]], dict[str, Any]]]
    output_adapter: Callable[[Any], Any]
    state_dict_translator: Callable[
        [dict[str, torch.Tensor], Any], dict[str, torch.Tensor]
    ] | None = None


def identity_output(output: Any) -> Any:
    return output


def _constructor_kwargs(model_cls: type[nn.Module], cfg: Any) -> dict[str, Any]:
    params = inspect.signature(model_cls.__init__).parameters
    accepted = {
        name for name, param in params.items()
        if name != "self"
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {
        key: value for key, value in asdict(cfg).items()
        if key in accepted
    }


def _spectral_adapter_factory(cfg: BaseDiffusionConfig):
    return make_spectral_batch_adapter(
        top_k_freqs=cfg.top_k_freqs,
        in_channels=cfg.in_channels,
        cond_channels=cfg.cond_channels,
        is_dct=cfg.is_dct,
        conditioned_freq_scale=cfg.conditioned_freq_scale,
    )


MODEL_REGISTRY = {
    "spectral_dit_low_k": ModelSpec(
        name="spectral_dit_low_k",
        config_type=SpectralDiTConfig,
        build_fn=lambda cfg: SpectralDiT(
            **{
                **_constructor_kwargs(SpectralDiT, cfg),
                "use_low_k_correction_head": True,
            }
        ),
        input_adapter_factory=_spectral_adapter_factory,
        output_adapter=identity_output,
    ),
    "spectral_conv_block_mix_amplitude": ModelSpec(
        name="spectral_conv_block_mix_amplitude",
        config_type=SpectralConvBlockMixAmplitudeConfig,
        build_fn=lambda cfg: SpectralConvBlockMixAmplitude(
            **_constructor_kwargs(SpectralConvBlockMixAmplitude, cfg)
        ),
        input_adapter_factory=_spectral_adapter_factory,
        output_adapter=identity_output,
    ),
}


# Final Wrapper interface point

SUPPORTED_MODEL_TYPES = tuple(MODEL_REGISTRY)

class UnifiedWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        config: BaseDiffusionConfig,
        checkpoint_path: str | None = None,
        strict_load: bool = True,
    ):
        super().__init__()

        registry = MODEL_REGISTRY
        if model_name not in registry:
            raise ValueError(f"Unknown model_name={model_name!r}")

        self.spec = registry[model_name]
        self.input_adapter = self.spec.input_adapter_factory(config)

        if not isinstance(config, self.spec.config_type):
            raise TypeError(
                f"{model_name} expects config type {self.spec.config_type.__name__}, "
                f"got {type(config).__name__}"
            )

        self.config = config
        self.model = self.spec.build_fn(config)
        self._forward_signature = inspect.signature(self.model.forward)
        self._accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self._forward_signature.parameters.values()
        )
        self._accepted_forward_kwargs = {
            name
            for name, p in self._forward_signature.parameters.items()
            if name != "self"
            and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            translator = getattr(self.spec, "state_dict_translator", None)
            if translator is not None:
                state = translator(state, config)
            self.model.load_state_dict(state, strict=strict_load)

    @property
    def prediction_target(self) -> str:
        return getattr(self.config, "prediction_target", "noise")

    @property
    def is_dct(self) -> bool:
        return getattr(self.model, "is_dct", True)

    @property
    def is_time_domain(self) -> bool:
        return getattr(self.model, "is_time_domain", False)

    @property
    def is_manifold_domain(self) -> bool:
        return getattr(self.model, "is_manifold_domain", False)

    @property
    def freq_scale(self):
        return getattr(self.model, "freq_scale", None)

    def _adapt_input(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.input_adapter(batch)

    def _adapt_output(self, output: Any) -> Any:
        return self.spec.output_adapter(output)

    def forward(self, batch: dict[str, Any]) -> Any:
        kwargs = self._adapt_input(batch)
        if not self._accepts_var_kwargs:
            kwargs = {
                k: v for k, v in kwargs.items()
                if k in self._accepted_forward_kwargs
            }
        out = self.model(**kwargs)
        return self._adapt_output(out)
    
