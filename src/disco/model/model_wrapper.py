from __future__ import annotations

from typing import Any
import inspect
import torch
import torch.nn as nn

import src.models.registry as model_registry
from src.models.registry import BaseDiffusionConfig


class UnifiedWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        config: BaseDiffusionConfig,
        checkpoint_path: str | None = None,
        strict_load: bool = True,
    ):
        super().__init__()

        registry = model_registry.MODEL_REGISTRY
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
    
