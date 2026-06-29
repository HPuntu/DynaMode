"""
Frequency-band parsing helpers for spectral models.

All band edges are half-open intervals [edge_i, edge_{i+1}) over DCT mode
indices. The canonical SpecConv grouping keeps DC separate:
DC | 1-8 | 9-32 | 33-128 | 129+.
"""

from __future__ import annotations

import re
from typing import Iterable


def default_band_edges(top_k_freqs: int, scheme: str = "block_mix") -> tuple[int, ...]:
    """Return a default frequency grouping for top_k_freqs modes."""
    K = int(top_k_freqs)
    if K <= 0:
        raise ValueError(f"top_k_freqs must be positive, got {top_k_freqs}")

    scheme = str(scheme).strip().lower().replace("-", "_")
    if scheme in {"low_k", "lowk", "dct_low", "dct"}:
        candidates = (0, 1, 5, 17, 65, K)
    elif scheme in {"block_mix", "physical", "spec_conv"}:
        candidates = (0, 1, 9, 33, 129, K)
    else:
        raise ValueError(
            f"Unknown default band scheme {scheme!r}. Use 'block_mix', "
            "'physical', 'spec_conv', or 'low_k'."
        )

    edges: list[int] = []
    for edge in candidates:
        edge = min(max(int(edge), 0), K)
        if not edges or edge > edges[-1]:
            edges.append(edge)
    if edges[0] != 0:
        edges.insert(0, 0)
    if edges[-1] != K:
        edges.append(K)
    return validate_band_edges(edges, K)


def validate_band_edges(band_edges: Iterable[int], top_k_freqs: int) -> tuple[int, ...]:
    """Validate and normalize a half-open frequency band specification."""
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
        return int(match.group(1)), K

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
    """
    Parse a frequency band spec into validated half-open edges.

    Supported forms:
    - None -> :func:`default_band_edges`
    - iterable of integer edges, e.g. (0, 1, 9, 33, 129, 256)
    - named default: "block_mix", "physical", "spec_conv", or "low_k"
    - edge string: "0,1,9,33,129,256"
    - inclusive ranges: "DC,1-4,5-16,17+"
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
            "block_mix",
            "physical",
            "spec_conv",
            "low_k",
            "lowk",
            "dct_low",
            "dct",
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
