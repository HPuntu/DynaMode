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
from dynamode.model.spec_conv.block_mix import SpectralConvBlockMixAmplitude
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


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def make_model_config(
    model_type: str,
    *,
    in_channels: int,
    cond_channels: int,
    top_k_freqs: int,
    freq_scale: torch.Tensor | None = None,
    conditioned_freq_scale: dict[str, Any] | None = None,
    is_dct: bool = True,
    **kwargs: Any,
) -> BaseDiffusionConfig:
    '''Create the public model config dataclass for a resolved runtime stack.'''
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unknown model_type={model_type!r}. Supported public models: {supported}")

    depth = int(_config_value(kwargs, "num_layers", _config_value(kwargs, "depth", 12)))
    num_heads = int(_config_value(kwargs, "num_heads", 8))
    prediction_target = str(_config_value(kwargs, "prediction_target", "x_0"))
    cfg_dropout = bool(_config_value(kwargs, "conditioning_dropout", _config_value(kwargs, "cfg_dropout", False)))

    if model_type == "spectral_dit_low_k":
        return SpectralDiTConfig(
            in_channels=int(in_channels),
            cond_channels=int(cond_channels),
            depth=depth,
            num_heads=num_heads,
            top_k_freqs=int(top_k_freqs),
            freq_hidden_size=int(_config_value(kwargs, "freq_hidden_size", 12)),
            mlp_ratio=float(_config_value(kwargs, "mlp_ratio", 4.0)),
            attn_dropout=float(_config_value(kwargs, "attn_dropout", 0.0)),
            freq_scale=freq_scale,
            conditioned_freq_scale=conditioned_freq_scale,
            cfg_dropout=cfg_dropout,
            prediction_target=prediction_target,
            is_dct=bool(is_dct),
            use_seq_conditioning=bool(_config_value(kwargs, "use_seq_conditioning", False)),
            seq_embed_dim=int(_config_value(kwargs, "seq_embed_dim", 16)),
            use_ss_conditioning=bool(_config_value(kwargs, "use_ss_conditioning", False)),
            ss_embed_dim=int(_config_value(kwargs, "ss_embed_dim", 8)),
            use_low_k_correction_head=True,
            low_k_correction_modes=_config_value(kwargs, "low_k_correction_modes", 1),
            cond_dim=int(_config_value(kwargs, "cond_dim", 512)),
        )

    spectral_modes = _config_value(kwargs, "spectral_modes", top_k_freqs)
    band_edges = kwargs.get("band_edges", kwargs.get("fast_band_edges", None))
    return SpectralConvBlockMixAmplitudeConfig(
        in_channels=int(in_channels),
        cond_channels=int(cond_channels),
        depth=depth,
        num_heads=num_heads,
        top_k_freqs=int(top_k_freqs),
        freq_hidden_size=int(_config_value(kwargs, "freq_hidden_size", 12)),
        spectral_modes=int(top_k_freqs if spectral_modes is None else spectral_modes),
        attn_dropout=float(_config_value(kwargs, "attn_dropout", 0.0)),
        freq_scale=freq_scale,
        conditioned_freq_scale=conditioned_freq_scale,
        cfg_dropout=cfg_dropout,
        prediction_target=prediction_target,
        is_dct=bool(is_dct),
        use_hilbert=bool(_config_value(kwargs, "use_hilbert_spatial", _config_value(kwargs, "use_hilbert", False))),
        use_hilbert_dct=bool(_config_value(kwargs, "use_hilbert_spatial_dct", _config_value(kwargs, "use_hilbert_dct", False))),
        hilbert_mode=str(_config_value(kwargs, "hilbert_mode", "every_block")),
        use_rmsf_prior_gain=bool(_config_value(kwargs, "use_rmsf_prior_gain", False)),
        use_low_k_correction_head=bool(_config_value(kwargs, "use_low_k_correction_head", False)),
        low_k_correction_modes=_config_value(kwargs, "low_k_correction_modes", 1),
        use_seq_conditioning=bool(_config_value(kwargs, "use_seq_conditioning", False)),
        seq_embed_dim=int(_config_value(kwargs, "seq_embed_dim", 16)),
        use_ss_conditioning=bool(_config_value(kwargs, "use_ss_conditioning", False)),
        ss_embed_dim=int(_config_value(kwargs, "ss_embed_dim", 8)),
        cond_dim=int(_config_value(kwargs, "cond_dim", 512)),
        band_edges=band_edges,
        amp_head_context_modes=int(_config_value(kwargs, "amp_head_context_modes", 4)),
        amp_head_target_modes=int(_config_value(kwargs, "amp_head_target_modes", 1)),
        amp_head_d_model=int(_config_value(kwargs, "amp_head_d_model", 128)),
        amp_head_depth=int(_config_value(kwargs, "amp_head_depth", 3)),
        amp_head_num_heads=int(_config_value(kwargs, "amp_head_num_heads", 4)),
        amp_head_mlp_ratio=float(_config_value(kwargs, "amp_head_mlp_ratio", 4.0)),
        amp_head_attn_dropout=float(_config_value(kwargs, "amp_head_attn_dropout", 0.0)),
        amp_head_use_rmsf_prior=bool(_config_value(kwargs, "amp_head_use_rmsf_prior", False)),
        use_shake=bool(_config_value(kwargs, "use_shake", False)),
        shake_n_iter=int(_config_value(kwargs, "shake_n_iter", 20)),
        shake_target=float(_config_value(kwargs, "shake_target", 3.8)),
    )


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


class ClassifierFreeGuidanceWrapper(nn.Module):
    '''Callable wrapper that applies classifier-free guidance at sampling time.

    Diffusion samplers call this object with tensor arguments rather than the
    batch dictionary accepted by :class:`UnifiedWrapper`. The wrapper converts
    those arguments back into the public batch contract, optionally evaluates
    both conditional and unconditional passes, and returns the guided model
    prediction. It deliberately knows nothing about diffusion schedules or
    coordinate decoding.
    '''

    def __init__(self, base_model: UnifiedWrapper, guidance_scale: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.guidance_scale = float(guidance_scale)
        self.prediction_target = getattr(base_model, "prediction_target", "noise")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        norm_temps: torch.Tensor,
        native_coords: torch.Tensor,
        native_angles: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.guidance_scale <= 1.0:
            return self.base_model({
                "x": x_t,
                "t": t,
                "temp": norm_temps,
                "native_coords": native_coords,
                "native_angles": native_angles,
                "mask": mask,
                "cond_drop_mask": None,
                **kwargs,
            })

        batch_size = x_t.shape[0]
        device = x_t.device
        cond_drop_mask = torch.cat([
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.ones(batch_size, dtype=torch.bool, device=device),
        ])
        duplicated_kwargs = {
            key: (
                torch.cat([value, value], dim=0)
                if torch.is_tensor(value)
                else value
            )
            for key, value in kwargs.items()
        }

        out = self.base_model({
            "x": torch.cat([x_t, x_t], dim=0),
            "t": torch.cat([t, t], dim=0),
            "temp": torch.cat([norm_temps, norm_temps], dim=0),
            "native_coords": torch.cat([native_coords, native_coords], dim=0),
            "native_angles": (
                torch.cat([native_angles, native_angles], dim=0)
                if native_angles is not None
                else None
            ),
            "mask": torch.cat([mask, mask], dim=0) if mask is not None else None,
            "cond_drop_mask": cond_drop_mask,
            **duplicated_kwargs,
        })
        cond, uncond = torch.chunk(out, 2, dim=0)
        return uncond + self.guidance_scale * (cond - uncond)


CFGModelWrapper = ClassifierFreeGuidanceWrapper
