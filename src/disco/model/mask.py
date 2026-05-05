"""Frequency-band masking and ordered group mixers for spectral models.

The helpers here keep two related ideas separate:

* frequency masks for hierarchical / masked diffusion experiments, where
  selected DCT groups are hidden from the denoiser input;
* ordering-aware frequency mixers, where each output group can receive
  information from itself, earlier groups, or all groups.

All band edges are half-open intervals ``[edge_i, edge_{i+1})`` over DCT mode
indices. Channel-expanded masks assume the flattened spectral layout used by
the training pipeline: ``[..., K * C]`` with channel index varying fastest.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import torch
import torch.nn as nn


ORDERING_ALIASES = {
    "block": "independent",
    "block_diagonal": "independent",
    "diag": "independent",
    "low_to_high": "causal",
    "forward": "causal",
    "reverse": "reverse_causal",
    "high_to_low": "reverse_causal",
    "full": "bidirectional",
    "all": "bidirectional",
    "bidir": "bidirectional",
    "bimodal": "outside_in",
    "two_sided": "outside_in",
}


@dataclass(frozen=True)
class FrequencyMaskSelection:
    """A sampled hierarchy state over frequency groups.

    ``observed_groups`` are the bands left visible in the input. ``target_groups``
    are the bands intended for supervision in a masked objective. The model code
    only applies the observed mask; the training loop can use the target groups
    to weight losses.
    """

    ordering: str
    step_index: int
    observed_groups: tuple[int, ...]
    target_groups: tuple[int, ...]


def canonical_ordering(ordering: str | None) -> str:
    """Normalise frequency ordering names used by configs and notebooks."""
    if ordering is None:
        return "causal"
    key = str(ordering).strip().lower().replace("-", "_")
    key = ORDERING_ALIASES.get(key, key)
    valid = {
        "independent",
        "causal",
        "reverse_causal",
        "bidirectional",
        "outside_in",
    }
    if key not in valid:
        raise ValueError(
            f"Unknown frequency ordering {ordering!r}. "
            f"Expected one of {sorted(valid | set(ORDERING_ALIASES))}."
        )
    return key


def default_band_edges(top_k_freqs: int, scheme: str = "block_mix") -> tuple[int, ...]:
    """Return a sensible default frequency grouping for ``K`` modes.

    ``block_mix`` mirrors the DC-separated spectral-conv block-mix bands.
    ``block_mix_legacy`` restores the pre-DC-split spectral-conv bands used
    by older v10/v12 checkpoints.
    ``low_k`` keeps the older experimental low-mode grouping used by some
    masked-denoising notebook sweeps.
    """
    K = int(top_k_freqs)
    if K <= 0:
        raise ValueError(f"top_k_freqs must be positive, got {top_k_freqs}")

    scheme = str(scheme).strip().lower().replace("-", "_")
    if scheme in {"low_k", "lowk", "dct_low", "dct"}:
        candidates = (0, 1, 5, 17, 65, K)
    elif scheme in {"legacy", "block_mix_legacy", "legacy_block_mix", "spec_conv_legacy"}:
        candidates = (0, 8, 32, 128, K)
    elif scheme in {"block_mix", "physical", "spec_conv"}:
        candidates = (0, 1, 9, 33, 129, K)
    else:
        raise ValueError(
            f"Unknown default band scheme {scheme!r}. Use 'block_mix', "
            "'block_mix_legacy', or 'low_k'."
        )

    clipped = [min(max(int(edge), 0), K) for edge in candidates]
    edges = []
    for edge in clipped:
        if not edges or edge > edges[-1]:
            edges.append(edge)
    if edges[0] != 0:
        edges.insert(0, 0)
    if edges[-1] != K:
        edges.append(K)
    return validate_band_edges(edges, K)


def validate_band_edges(band_edges: Iterable[int], top_k_freqs: int) -> tuple[int, ...]:
    """Validate and normalise a half-open frequency band specification."""
    K = int(top_k_freqs)
    edges = tuple(int(edge) for edge in band_edges)
    if len(edges) < 2:
        raise ValueError(f"band_edges must have at least two entries, got {edges!r}")
    if edges[0] != 0:
        raise ValueError(f"band_edges must start at 0, got {edges!r}")
    if edges[-1] != K:
        raise ValueError(f"band_edges must end at top_k_freqs={K}, got {edges!r}")
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            raise ValueError(f"band_edges must be strictly increasing, got {edges!r}")
    return edges


def _parse_range_token(token: str, top_k_freqs: int) -> tuple[int, int]:
    """Parse one inclusive range token into a half-open interval."""
    raw = token.strip()
    if ":" in raw:
        raw = raw.split(":", 1)[1].strip()
    key = raw.upper()
    K = int(top_k_freqs)

    if key == "DC":
        return 0, 1

    match = re.fullmatch(r"(\d+)\s*(?:\+|\.\.|\-\s*\*)", key)
    if match:
        start = int(match.group(1))
        return start, K

    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", key)
    if match:
        start = int(match.group(1))
        end = int(match.group(2)) + 1
        return start, end

    if re.fullmatch(r"\d+", key):
        value = int(key)
        return value, value + 1

    raise ValueError(
        f"Invalid frequency band token {token!r}. Use forms like "
        "'DC', '1-4', '17+', or explicit edge lists like '0,1,9,33,129,256'."
    )


def parse_band_edges(
    spec: str | Iterable[int] | None,
    top_k_freqs: int,
    default_scheme: str = "block_mix",
) -> tuple[int, ...]:
    """Parse a frequency band spec into validated half-open edges.

    Supported forms:
    * ``None`` -> :func:`default_band_edges`
    * iterable of integer edges, e.g. ``(0, 1, 9, 33, 129, 256)``
    * named default: ``"block_mix"``, ``"block_mix_legacy"``, or ``"low_k"``
    * edge string: ``"0,1,9,33,129,256"``
    * inclusive ranges: ``"DC,1-4,5-16,17+"``
    """
    K = int(top_k_freqs)
    if spec is None:
        return default_band_edges(K, scheme=default_scheme)
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return default_band_edges(K, scheme=default_scheme)
        lowered = text.lower().replace("-", "_")
        if lowered in {
            "block_mix", "physical", "spec_conv",
            "legacy", "block_mix_legacy", "legacy_block_mix", "spec_conv_legacy",
            "low_k", "lowk", "dct_low", "dct",
        }:
            return default_band_edges(K, scheme=lowered)

        tokens = [token.strip() for token in text.split(",") if token.strip()]
        if not tokens:
            return default_band_edges(K, scheme=default_scheme)

        numeric_tokens = []
        for token in tokens:
            cleaned = token.split(":", 1)[-1].strip()
            if not re.fullmatch(r"\d+", cleaned):
                numeric_tokens = []
                break
            numeric_tokens.append(int(cleaned))
        if numeric_tokens and numeric_tokens[0] == 0:
            return validate_band_edges(numeric_tokens, K)

        intervals = [_parse_range_token(token, K) for token in tokens]
        intervals.sort(key=lambda item: item[0])
        if intervals[0][0] != 0:
            raise ValueError(f"Frequency ranges must start at 0 or DC, got {spec!r}")

        edges = [0]
        cursor = 0
        for start, end in intervals:
            if start != cursor:
                raise ValueError(
                    f"Frequency ranges must be contiguous; expected start {cursor}, "
                    f"got {start} in {spec!r}"
                )
            if end <= start:
                raise ValueError(f"Empty frequency range {start}:{end} in {spec!r}")
            edges.append(min(end, K))
            cursor = end
            if cursor >= K:
                break
        if edges[-1] < K:
            edges.append(K)
        return validate_band_edges(edges, K)

    return validate_band_edges(spec, K)


def group_order(num_groups: int, ordering: str | None = "causal") -> tuple[int, ...]:
    """Return the frequency-group generation order."""
    G = int(num_groups)
    if G <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    mode = canonical_ordering(ordering)
    if mode in {"independent", "causal", "bidirectional"}:
        return tuple(range(G))
    if mode == "reverse_causal":
        return tuple(reversed(range(G)))
    if mode == "outside_in":
        order: list[int] = []
        left, right = 0, G - 1
        while left <= right:
            order.append(left)
            if right != left:
                order.append(right)
            left += 1
            right -= 1
        return tuple(order)
    raise AssertionError(f"Unhandled ordering mode {mode!r}")


def group_dependency_mask(
    num_groups: int,
    ordering: str | None = "causal",
    *,
    include_self: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build an output-group by input-group dependency mask.

    ``mask[g_out, g_in]`` is true if output group ``g_out`` may read input group
    ``g_in``. ``causal`` is low-to-high, ``outside_in`` alternates low/high
    groups, and ``bidirectional`` is full all-to-all.
    """
    G = int(num_groups)
    mode = canonical_ordering(ordering)
    if mode == "bidirectional":
        return torch.ones(G, G, dtype=torch.bool, device=device)
    if mode == "independent":
        return torch.eye(G, dtype=torch.bool, device=device)

    order = group_order(G, mode)
    pos = {group: idx for idx, group in enumerate(order)}
    dep = torch.zeros(G, G, dtype=torch.bool, device=device)
    for out_group in range(G):
        out_pos = pos[out_group]
        for in_group in range(G):
            in_pos = pos[in_group]
            dep[out_group, in_group] = in_pos <= out_pos if include_self else in_pos < out_pos
    return dep


def group_ids_for_modes(
    top_k_freqs: int,
    band_edges: Iterable[int],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a length-``K`` tensor assigning each mode to a group index."""
    edges = validate_band_edges(band_edges, top_k_freqs)
    group_ids = torch.empty(int(top_k_freqs), dtype=torch.long, device=device)
    for group_idx, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        group_ids[start:end] = group_idx
    return group_ids


def frequency_channel_mask(
    top_k_freqs: int,
    channels: int,
    band_edges: Iterable[int],
    *,
    keep_groups: Sequence[int] | torch.Tensor | None = None,
    mask_groups: Sequence[int] | torch.Tensor | None = None,
    keep_modes: Sequence[int] | torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a flattened ``(K * C,)`` mask for the spectral-volume layout."""
    K = int(top_k_freqs)
    C = int(channels)
    edges = validate_band_edges(band_edges, K)
    G = len(edges) - 1
    mode_keep = torch.ones(K, dtype=torch.bool, device=device)

    if keep_groups is not None:
        keep = torch.zeros(G, dtype=torch.bool, device=device)
        keep[torch.as_tensor(keep_groups, dtype=torch.long, device=device)] = True
        mode_keep = keep[group_ids_for_modes(K, edges, device=device)]

    if mask_groups is not None:
        drop = torch.zeros(G, dtype=torch.bool, device=device)
        drop[torch.as_tensor(mask_groups, dtype=torch.long, device=device)] = True
        mode_keep = mode_keep & ~drop[group_ids_for_modes(K, edges, device=device)]

    if keep_modes is not None:
        explicit = torch.zeros(K, dtype=torch.bool, device=device)
        explicit[torch.as_tensor(keep_modes, dtype=torch.long, device=device)] = True
        mode_keep = mode_keep & explicit

    mask = mode_keep.repeat_interleave(C)
    if dtype is not None:
        return mask.to(dtype=dtype)
    return mask


def apply_frequency_mask(
    x: torch.Tensor,
    top_k_freqs: int,
    channels: int,
    band_edges: Iterable[int],
    *,
    keep_groups: Sequence[int] | torch.Tensor | None = None,
    mask_groups: Sequence[int] | torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Mask frequency groups in either flattened or ``(..., K, C)`` layout."""
    K = int(top_k_freqs)
    C = int(channels)
    if mask is None:
        flat_mask = frequency_channel_mask(
            K,
            C,
            band_edges,
            keep_groups=keep_groups,
            mask_groups=mask_groups,
            device=x.device,
            dtype=torch.bool,
        )
    else:
        flat_mask = mask.to(device=x.device, dtype=torch.bool)
        if flat_mask.shape[-1] == K:
            flat_mask = flat_mask.repeat_interleave(C, dim=-1)

    if x.shape[-1] == K * C:
        view_mask = flat_mask
        if view_mask.ndim == 1:
            while view_mask.ndim < x.ndim:
                view_mask = view_mask.unsqueeze(0)
        elif view_mask.ndim == x.ndim - 1:
            # Common per-sample case: (B, K*C) -> (B, 1, K*C)
            view_mask = view_mask.unsqueeze(-2)
        elif view_mask.ndim != x.ndim:
            raise ValueError(
                f"Cannot broadcast frequency mask shape {tuple(flat_mask.shape)} "
                f"to x shape {tuple(x.shape)}"
            )
        return torch.where(view_mask, x, x.new_tensor(fill_value))

    if x.ndim >= 2 and x.shape[-2:] == (K, C):
        mode_mask = flat_mask.reshape(K, C).any(dim=-1)
        view_mask = mode_mask
        while view_mask.ndim < x.ndim - 1:
            view_mask = view_mask.unsqueeze(0)
        view_mask = view_mask.unsqueeze(-1)
        return torch.where(view_mask, x, x.new_tensor(fill_value))

    raise ValueError(
        f"apply_frequency_mask expected last dim K*C={K*C} or trailing dims "
        f"(K,C)=({K},{C}), got shape {tuple(x.shape)}"
    )


def sample_frequency_mask_selection(
    num_groups: int,
    ordering: str | None = "causal",
    *,
    mode: str = "next",
    step_index: int | None = None,
    generator: torch.Generator | None = None,
) -> FrequencyMaskSelection:
    """Sample a masked-diffusion hierarchy state.

    ``mode='next'`` targets the next group after the visible prefix.
    ``mode='suffix'`` targets all remaining groups after the visible prefix.
    """
    G = int(num_groups)
    order = group_order(G, ordering)
    if step_index is None:
        step_tensor = torch.randint(0, G, (1,), generator=generator)
        step_index = int(step_tensor.item())
    step_index = max(0, min(int(step_index), G - 1))

    visible = tuple(order[:step_index])
    if mode == "next":
        target = (order[step_index],)
    elif mode == "suffix":
        target = tuple(order[step_index:])
    else:
        raise ValueError(f"Unknown frequency mask selection mode {mode!r}")

    return FrequencyMaskSelection(
        ordering=canonical_ordering(ordering),
        step_index=step_index,
        observed_groups=visible,
        target_groups=target,
    )


class FrequencyGroupedMixer(nn.Module):
    """Linear mixer over frequency groups with configurable dependencies.

    This is a drop-in replacement for the spectral-conv block's
    ``nn.Linear(K, K)`` mixer. Each output band has its own linear projection
    from the allowed input bands. ``ordering='causal'`` gives low-to-high
    hierarchical mixing, ``ordering='outside_in'`` alternates low and high
    anchors, ``ordering='bidirectional'`` recovers full cross-band mixing, and
    ``ordering='independent'`` is block-diagonal.
    """

    def __init__(
        self,
        band_edges: Iterable[int],
        ordering: str | None = "causal",
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        edges = tuple(int(edge) for edge in band_edges)
        self.band_edges = validate_band_edges(edges, edges[-1])
        self.n_freqs = self.band_edges[-1]
        self.ordering = canonical_ordering(ordering)
        self.num_groups = len(self.band_edges) - 1
        dep = group_dependency_mask(self.num_groups, self.ordering)
        self.register_buffer("dependency_mask", dep, persistent=False)

        self.allowed_groups: list[tuple[int, ...]] = []
        self.group_mixers = nn.ModuleList()
        for out_group, (out_start, out_end) in enumerate(zip(self.band_edges[:-1], self.band_edges[1:])):
            allowed = tuple(int(i) for i in torch.nonzero(dep[out_group], as_tuple=False).flatten().tolist())
            in_width = sum(self.band_edges[i + 1] - self.band_edges[i] for i in allowed)
            out_width = out_end - out_start
            self.allowed_groups.append(allowed)
            self.group_mixers.append(nn.Linear(in_width, out_width, bias=bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_freqs:
            raise RuntimeError(
                f"FrequencyGroupedMixer expected last dim {self.n_freqs}, got {x.shape[-1]}"
            )
        parts = [
            x[..., start:end]
            for start, end in zip(self.band_edges[:-1], self.band_edges[1:])
        ]
        outputs = []
        for mixer, allowed in zip(self.group_mixers, self.allowed_groups):
            source = torch.cat([parts[group] for group in allowed], dim=-1)
            outputs.append(mixer(source))
        return torch.cat(outputs, dim=-1)

    @classmethod
    def from_dense(
        cls,
        dense: nn.Linear,
        band_edges: Iterable[int],
        ordering: str | None = "causal",
    ) -> "FrequencyGroupedMixer":
        """Warm-start an ordered group mixer from a dense ``K x K`` mixer."""
        K = int(dense.in_features)
        if dense.out_features != K:
            raise ValueError(
                f"dense mixer must be square, got in={dense.in_features}, out={dense.out_features}"
            )
        mixer = cls(band_edges, ordering=ordering, bias=dense.bias is not None)
        if mixer.n_freqs != K:
            raise ValueError(f"band_edges imply K={mixer.n_freqs}, dense mixer has K={K}")

        with torch.no_grad():
            W = dense.weight.detach()
            b = dense.bias.detach() if dense.bias is not None else None
            for out_group, group_mixer in enumerate(mixer.group_mixers):
                out_start = mixer.band_edges[out_group]
                out_end = mixer.band_edges[out_group + 1]
                col_chunks = []
                for in_group in mixer.allowed_groups[out_group]:
                    in_start = mixer.band_edges[in_group]
                    in_end = mixer.band_edges[in_group + 1]
                    col_chunks.append(W[out_start:out_end, in_start:in_end])
                group_mixer.weight.data.copy_(torch.cat(col_chunks, dim=1))
                if b is not None:
                    group_mixer.bias.data.copy_(b[out_start:out_end])
        return mixer


def replace_freq_mixers_with_grouped(
    trunk: nn.Module,
    band_edges: Iterable[int],
    ordering: str | None = "causal",
    *,
    warm_start_from_dense: bool = True,
) -> nn.Module:
    """Replace ``block.freq_mixer`` modules on a spectral-conv trunk."""
    edges = validate_band_edges(band_edges, int(getattr(trunk, "top_k_freqs")))
    for block in getattr(trunk, "blocks", []):
        dense = getattr(block, "freq_mixer", None)
        if dense is None:
            continue
        if isinstance(dense, FrequencyGroupedMixer):
            continue
        if not isinstance(dense, nn.Linear):
            raise TypeError(
                f"Expected block.freq_mixer to be nn.Linear or FrequencyGroupedMixer, "
                f"got {type(dense).__name__}"
            )
        if warm_start_from_dense:
            new_mixer = FrequencyGroupedMixer.from_dense(dense, edges, ordering=ordering)
        else:
            new_mixer = FrequencyGroupedMixer(edges, ordering=ordering, bias=dense.bias is not None)
        ref_param = next(block.parameters())
        block.freq_mixer = new_mixer.to(device=ref_param.device, dtype=ref_param.dtype)
    return trunk


def convert_dense_mixer_checkpoint_grouped(
    src_state: dict[str, torch.Tensor],
    band_edges: Iterable[int],
    top_k_freqs: int,
    depth: int,
    ordering: str | None = "causal",
) -> dict[str, torch.Tensor]:
    """Translate dense spectral-conv mixer keys to ordered group-mixer keys."""
    edges = validate_band_edges(band_edges, int(top_k_freqs))
    G = len(edges) - 1
    dep = group_dependency_mask(G, ordering)
    out: dict[str, torch.Tensor] = {}
    dense_keys_handled: set[str] = set()

    for block_idx in range(int(depth)):
        w_key = f"blocks.{block_idx}.freq_mixer.weight"
        b_key = f"blocks.{block_idx}.freq_mixer.bias"
        alt_w_key = f"trunk.blocks.{block_idx}.freq_mixer.weight"
        alt_b_key = f"trunk.blocks.{block_idx}.freq_mixer.bias"
        W = src_state.get(w_key, src_state.get(alt_w_key))
        b = src_state.get(b_key, src_state.get(alt_b_key))
        if W is None:
            continue
        dense_keys_handled.update({w_key, b_key, alt_w_key, alt_b_key})

        for out_group in range(G):
            out_start, out_end = edges[out_group], edges[out_group + 1]
            allowed = [idx for idx in range(G) if bool(dep[out_group, idx])]
            cols = []
            for in_group in allowed:
                in_start, in_end = edges[in_group], edges[in_group + 1]
                cols.append(W[out_start:out_end, in_start:in_end])
            new_prefix = f"trunk.blocks.{block_idx}.freq_mixer.group_mixers.{out_group}"
            out[f"{new_prefix}.weight"] = torch.cat(cols, dim=1).clone()
            if b is not None:
                out[f"{new_prefix}.bias"] = b[out_start:out_end].clone()

    for key, value in src_state.items():
        if key in dense_keys_handled:
            continue
        out_key = key if key.startswith("trunk.") else f"trunk.{key}"
        out[out_key] = value
    return out
