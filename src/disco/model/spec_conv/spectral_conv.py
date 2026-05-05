'''
CNO2 — Cosine Neural Operator with spectral convolution and cross-frequency mixer.

Difference from the baseline transformer
-----------------------------------------
The FFN in each block is replaced by an FNO-style spectral operation that
explicitly respects the K (frequency) structure of the token:

    SpectralConv1d(h_freq, h_freq, modes)   — learns cross-mode mixing
  + nn.Conv1d(h_freq, h_freq, 1)            — pointwise channel mix
  + nn.Linear(K, K)                         — dense cross-frequency mixer
  + GELU

The spectral convolution operates on the per-frequency hidden dimension as the
"channel" axis, with the K DCT modes forming the spatial axis. This is the FNO
formulation applied to the frequency dimension of the spectral volume.

The freq_mixer (K → K linear) enables full cross-mode energy redistribution,
allowing the model to learn that low-frequency unfolding drift and high-frequency
jitter are not independent.

DC contamination note
---------------------
The freq_mixer can, in principle, project k=0 (DC, mean position, ~tens of Å)
into high-frequency modes. This is suppressed by per-frequency normalisation:
after normalisation, all modes have comparable unit variance, so the mixer
operates on a well-conditioned, isotropic input rather than a wildly skewed one.

All other design choices (per-freq normalisation, three-term conditioning,
v-prediction, AdaLN-Zero, RoPE) are identical to the transformer baseline.
'''

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.spectral.adapters import SpectralAdapter, DCT, DFT
from src.models.conditioned_freq_scale import size_bin_label

from src.models.modules import (
    RotaryEmbedding,
    ScalarEmbedding,
    SmoothScalarEmbedding,
    WindowContextEmbedding,
    apply_rotary_pos_emb,
)


# ---------------------------------------------------------------------------
# Hilbert spatial envelope
# ---------------------------------------------------------------------------


def _parse_low_k_correction_spec(spec, top_k_freqs: int) -> tuple[list[tuple[int, int]], list[int]]:
    """Parse correction-head specs into non-overlapping inclusive mode ranges.

    Supported forms:
    - legacy int ``n`` -> range ``0..n-1``
    - bare digit string ``"4"`` -> legacy count ``0..3``
    - ``"DC"`` -> ``0..0``
    - ``"1-4"`` -> inclusive range ``1..4``
    - comma-separated combinations like ``"DC,1-4"``
    """
    if spec is None:
        return [], []

    ranges: list[tuple[int, int]] = []

    if isinstance(spec, int):
        width = min(max(int(spec), 1), int(top_k_freqs))
        ranges = [(0, width - 1)]
    elif isinstance(spec, str):
        text = spec.strip()
        if not text:
            return [], []
        if re.fullmatch(r"\d+", text):
            width = min(max(int(text), 1), int(top_k_freqs))
            ranges = [(0, width - 1)]
        else:
            for raw_token in text.split(","):
                token = raw_token.strip().upper()
                if not token:
                    continue
                if token == "DC":
                    start, end = 0, 0
                elif re.fullmatch(r"\d+", token):
                    start = end = int(token)
                else:
                    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
                    if match is None:
                        raise ValueError(
                            f"Invalid low_k_correction_modes token {raw_token!r}. "
                            "Use forms like 'DC', '0-4', or 'DC,1-4'."
                        )
                    start, end = int(match.group(1)), int(match.group(2))
                if start > end:
                    raise ValueError(
                        f"Invalid low_k_correction_modes token {raw_token!r}: start > end."
                    )
                if start < 0 or end >= int(top_k_freqs):
                    raise ValueError(
                        f"Invalid low_k_correction_modes token {raw_token!r}: "
                        f"modes must lie within [0, {int(top_k_freqs) - 1}]."
                    )
                ranges.append((start, end))
    else:
        raise TypeError(
            f"low_k_correction_modes must be int or str, got {type(spec).__name__}"
        )

    if not ranges:
        return [], []

    ranges = sorted(ranges, key=lambda item: (item[0], item[1]))
    for (prev_start, prev_end), (start, end) in zip(ranges, ranges[1:]):
        if start <= prev_end:
            raise ValueError(
                "low_k_correction_modes defines overlapping correction heads. "
                f"Got ranges {(prev_start, prev_end)} and {(start, end)}."
            )

    target_modes: list[int] = []
    for start, end in ranges:
        target_modes.extend(range(start, end + 1))
    return ranges, target_modes

class HilbertSpatialEnvelope(nn.Module):
    '''Per-mode spatial amplitude envelope via Hilbert transform along L.

    For each DCT mode k and hidden channel h, computes the analytic signal
    along the residue axis L and returns the amplitude envelope |z(l)|.
    This gives the model explicit access to which contiguous spatial regions
    co-participate in each frequency mode — information that the per-residue
    spectral conv cannot derive (it operates within a single residue) and
    that full-token attention only represents implicitly.

    Injected as a zero-init residual at the start of Phase 2 in
    SpectralConvBlock so it is harmless at initialisation:
        xs ← xs + gate * envelope(xs)
    where gate is a learned scalar initialised to 0.

    Args:
        freq_hidden_size: Per-frequency hidden width H.
        n_freqs: Number of DCT modes K.
    '''

    def __init__(self, freq_hidden_size: int, n_freqs: int):
        super().__init__()
        # Scalar gate per hidden channel, zero-initialised → identity at start
        self.gate = nn.Parameter(torch.zeros(freq_hidden_size))

    @staticmethod
    def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
        '''Amplitude envelope of real signal x along its last dimension.

        Uses the standard analytic-signal construction:
            1. fft along last dim
            2. double positive frequencies, zero negative frequencies
            3. ifft → complex analytic signal
            4. amplitude = |analytic|

        Args:
            x: (..., L) real tensor.

        Returns:
            Amplitude envelope (..., L), same shape and dtype as x.
        '''
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.to(dtype=torch.float32)
            N = x_fp32.shape[-1]
            X = torch.fft.fft(x_fp32, dim=-1)

            h = torch.zeros(N, dtype=x_fp32.dtype, device=x_fp32.device)
            h[0] = 1.0
            if N % 2 == 0:
                h[N // 2] = 1.0
                h[1:N // 2] = 2.0
            else:
                h[1:(N + 1) // 2] = 2.0

            analytic = torch.fft.ifft(X * h, dim=-1)
            env = torch.abs(analytic)
        return env.to(dtype=x.dtype)

    def forward(self, xs: torch.Tensor, B: int, L: int) -> torch.Tensor:
        '''
        Args:
            xs: (B*L, H, K) — Phase 2 hidden state before spectral conv.
            B, L: batch size and residue count.

        Returns:
            (B*L, H, K) envelope tensor scaled by learned gate.
        '''
        H, K = xs.shape[1], xs.shape[2]
        # Reshape to (B, H, K, L) so L is the last axis for Hilbert
        x_spatial = xs.view(B, L, H, K).permute(0, 2, 3, 1)  # (B, H, K, L)
        env = self._hilbert_envelope(x_spatial)                 # (B, H, K, L)
        env = env.permute(0, 3, 1, 2).reshape(B * L, H, K)     # (B*L, H, K)
        return env * self.gate.view(1, H, 1)


class HilbertSpatialEnvelopeDCT(nn.Module):
    '''Boundary-safe spatial Hilbert envelope via even-symmetric extension.

    Drop-in replacement for :class:`HilbertSpatialEnvelope` that avoids the
    periodic-boundary Gibbs artifact of the straight FFT-Hilbert. The input
    signal along the residue axis L is mirrored to a length-2L even extension
    (the same implicit extension used by the DCT-II), the standard FFT-based
    analytic signal is computed on the now-continuous periodic signal, and
    the envelope of the first L samples is returned. Mathematically equivalent
    to constructing the analytic signal in the DCT/DST basis.

    Shares the same interface and the same single learnable parameter
    ``gate`` (shape ``(freq_hidden_size,)``) as :class:`HilbertSpatialEnvelope`,
    so a checkpoint trained with the FFT-based envelope loads into this
    module without modification — the stored ``gate`` values are carried over
    and the underlying transform is silently upgraded to the boundary-safe
    version.

    Args:
        freq_hidden_size: Per-frequency hidden width H.
        n_freqs: Number of DCT modes K (unused; kept for interface parity).
    '''

    def __init__(self, freq_hidden_size: int, n_freqs: int):
        super().__init__()
        # Scalar gate per hidden channel, zero-initialised → identity at start.
        # Name and shape match HilbertSpatialEnvelope exactly so state_dicts
        # interoperate.
        self.gate = nn.Parameter(torch.zeros(freq_hidden_size))

    @staticmethod
    def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
        '''Amplitude envelope of real signal x along its last dimension on
        the even-symmetric extension.

        The signal is mirrored to length 2L:
            ``x_ext = [x[0], ..., x[L-1], x[L-1], ..., x[0]]``
        which is continuous across the periodic wrap-around by construction
        (no jump at either seam). The standard FFT analytic-signal recipe
        then runs on a signal for which the periodicity assumption is valid,
        eliminating the central-amplification artifact of the raw FFT-Hilbert
        on a non-periodic residue chain.

        Args:
            x: (..., L) real tensor.

        Returns:
            Amplitude envelope (..., L), same shape and dtype as x.
        '''
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.to(dtype=torch.float32)
            L = x_fp32.shape[-1]
            N = 2 * L

            # Even-symmetric extension: mirror x along the last axis and append.
            # Matches the DCT-II implicit extension (reflection around L - 1/2).
            # `.contiguous()` avoids a PyTorch rfft backend warning about an
            # internal zero-size `out` tensor being resized on non-contiguous input.
            x_ext = torch.cat([x_fp32, torch.flip(x_fp32, dims=[-1])], dim=-1).contiguous()  # (..., 2L)

            X = torch.fft.fft(x_ext, dim=-1)                          # (..., 2L)

            h = torch.zeros(N, dtype=x_fp32.dtype, device=x_fp32.device)
            h[0] = 1.0
            h[L] = 1.0
            h[1:L] = 2.0

            analytic_ext = torch.fft.ifft(X * h, dim=-1)             # (..., 2L)
            analytic = analytic_ext[..., :L]
            env = torch.abs(analytic)
        return env.to(dtype=x.dtype)

    def forward(self, xs: torch.Tensor, B: int, L: int) -> torch.Tensor:
        '''
        Args:
            xs: (B*L, H, K) — Phase 2 hidden state before spectral conv.
            B, L: batch size and residue count.

        Returns:
            (B*L, H, K) envelope tensor scaled by learned gate.
        '''
        H, K = xs.shape[1], xs.shape[2]
        x_spatial = xs.view(B, L, H, K).permute(0, 2, 3, 1).contiguous()  # (B, H, K, L)
        env = self._hilbert_envelope(x_spatial)                            # (B, H, K, L)
        env = env.permute(0, 3, 1, 2).reshape(B * L, H, K)                # (B*L, H, K)
        return env * self.gate.view(1, H, 1)


# ---------------------------------------------------------------------------
# Spectral convolution
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    '''Learned mixing of the first `modes` DCT frequency modes.

    Weight shape: (C_in, C_out, modes). Modes beyond `modes` are zeroed
    (hard low-pass), so the spectral path cannot represent high-frequency
    content that wasn't already in the input.
    '''

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, K)
        B, C, K = x.shape
        modes = min(self.modes, K)
        out = torch.zeros_like(x)
        out[:, :, :modes] = torch.einsum(
            "bcm,com->bom", x[:, :, :modes], self.weight[:, :, :modes]
        )
        return out


# ---------------------------------------------------------------------------
# SpectralConv Block
# ---------------------------------------------------------------------------

class SpectralConvBlock(nn.Module):
    '''Residue attention + spectral conv + cross-frequency mixer.

    Phase 1 — Residue self-attention: inter-residue correlations.
    Phase 2 — Spectral FNO: SpectralConv + pointwise + freq_mixer + GELU.

    Both phases use AdaLN-Zero conditioning.

    Args:
        d_model: Total token dimension D = K * freq_hidden_size.
        freq_hidden_size: Per-frequency latent width H.
        n_freqs: Number of DCT modes K.
        num_heads: Attention heads; must divide D.
        spectral_modes: Number of modes processed by SpectralConv.
        attn_dropout: Attention output dropout for regularisation.
    '''

    def __init__(
        self,
        d_model: int,
        cond_dim: int,
        freq_hidden_size: int,
        n_freqs: int,
        num_heads: int,
        spectral_modes: int,
        attn_dropout: float = 0.0,
        use_hilbert: bool = False,
        use_hilbert_dct: bool = False,
    ):
        super().__init__()
        self.num_heads        = num_heads
        self.freq_hidden_size = freq_hidden_size
        self.n_freqs          = n_freqs

        # Phase 1: residue attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv   = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj  = nn.Linear(d_model, d_model, bias=False)
        self.drop  = nn.Dropout(attn_dropout)

        # Phase 2: spectral FNO
        self.norm2         = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.spectral_conv = SpectralConv1d(freq_hidden_size, min(spectral_modes, n_freqs))
        self.pointwise     = nn.Conv1d(freq_hidden_size, freq_hidden_size, 1)
        # Dense K×K mixer — enables cross-mode energy redistribution
        self.freq_mixer    = nn.Linear(n_freqs, n_freqs)

        # Optional Hilbert spatial envelope injection (zero-init gate → identity at init).
        # Two variants share the attribute name `hilbert` and the parameter name
        # `hilbert.gate`, so a checkpoint trained with one loads cleanly into the other.
        if use_hilbert_dct:
            self.hilbert = HilbertSpatialEnvelopeDCT(freq_hidden_size, n_freqs)
        elif use_hilbert:
            self.hilbert = HilbertSpatialEnvelope(freq_hidden_size, n_freqs)
        else:
            self.hilbert = None

        # AdaLN-Zero: cond_dim → 6 * d_model (decoupled from token width)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model, bias=True),
        )
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias,   0)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=1)

        # --- Phase 1: Residue attention ---
        xn = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        qkv = self.qkv(xn).reshape(B, L, 3, self.num_heads, D // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        attn_mask = None
        if mask is not None:
            mb = mask.bool()
            attn_mask = (mb.unsqueeze(1) & mb.unsqueeze(2)).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask, dropout_p=0.0,
        ).transpose(1, 2).flatten(2)
        x = x + gate1.unsqueeze(1) * self.drop(self.proj(out))
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # --- Phase 2: Spectral FNO ---
        xn = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)

        # Unflatten: (B, L, D) → (B*L, H, K)
        xs = xn.view(B, L, self.n_freqs, self.freq_hidden_size)
        xs = xs.permute(0, 1, 3, 2).reshape(B * L, self.freq_hidden_size, self.n_freqs)

        # Optional Hilbert spatial envelope: inject per-mode spatial coherence signal
        # before spectral conv. Zero-init gate means no effect at initialisation.
        if self.hilbert is not None:
            xs = xs + self.hilbert(xs, B, L)

        # SpectralConv + pointwise + freq_mixer + activation
        xs = self.spectral_conv(xs) + self.pointwise(xs)
        xs = self.freq_mixer(xs)
        xs = F.gelu(xs)

        # Reflatten: (B*L, H, K) → (B, L, D)
        xs = xs.reshape(B, L, self.freq_hidden_size, self.n_freqs)
        xs = xs.permute(0, 1, 3, 2).reshape(B, L, D)

        x = x + gate2.unsqueeze(1) * xs
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


VALID_HILBERT_MODES = {"every_block", "every_3_blocks", "input_only", "off"}


def _hilbert_enabled_for_block(hilbert_mode: str, block_idx: int) -> bool:
    if hilbert_mode == "every_block":
        return True
    if hilbert_mode == "every_3_blocks":
        return block_idx % 3 == 0
    if hilbert_mode == "input_only":
        return block_idx == 0
    if hilbert_mode == "off":
        return False
    raise ValueError(
        f"Invalid hilbert_mode={hilbert_mode!r}. "
        f"Expected one of {sorted(VALID_HILBERT_MODES)}."
    )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SpectralConvDiT(nn.Module):
    '''SpectralConv: spectral diffusion transformer with FNO-style frequency convolutions.

    Input: raw DCT coefficients of coordinate deviations (x - native), shape (B, L, K*C_in).
    Normalisation (÷ freq_scale) is applied internally; output is denormalised before return
    so the diffusion process and loss stay in raw coefficient scale.

    Args:
        top_k_freqs: Number of retained DCT modes K.
        in_channels: Coordinate channels per mode.
        cond_channels: Native conditioning channels.
        freq_hidden_size: Per-frequency latent width H; D = K * H.
        depth: Number of SpectralConvBlocks.
        num_heads: Attention heads; must divide D.
        spectral_modes: Number of modes used in SpectralConv (K_modes ≤ K).
                        Focus on low-frequency modes — recommend K/4.
        attn_dropout: Attention output dropout for regularisation. Recommend 0.1
            for small datasets (<500 proteins); 0.0 at full scale.
        freq_scale: Per-feature 95th-pct amplitudes (K*C_in,). Required for
                    correct training — without it, DC contamination via
                    freq_mixer will catastrophically amplify low modes.
        cfg_dropout: Enable classifier-free guidance conditioning dropout.
            Randomly nulls native_coords/temp/win_pos during training so the
            model learns both conditional and unconditional distributions.
            Unrelated to attn_dropout.
        prediction_target: "v", "x_0", or "noise".
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
        freq_scale: torch.Tensor = None,
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
    ):
        super().__init__()
        assert prediction_target in ("v", "x_0", "noise")
        assert not (use_hilbert and use_hilbert_dct), (
            "Enable at most one of use_hilbert (FFT-based) and use_hilbert_dct "
            "(boundary-safe DCT-based) — they share the same attribute name "
            "and parameter, so both cannot be active simultaneously."
        )
        if hilbert_mode not in VALID_HILBERT_MODES:
            raise ValueError(
                f"Invalid hilbert_mode={hilbert_mode!r}. "
                f"Expected one of {sorted(VALID_HILBERT_MODES)}."
            )
        self.use_rmsf_prior_gain = use_rmsf_prior_gain
        self.use_low_k_correction_head = use_low_k_correction_head

        self.prediction_target = prediction_target
        self.cfg_dropout       = cfg_dropout
        self.top_k_freqs       = top_k_freqs
        self.in_channels       = in_channels
        self.freq_hidden_size  = freq_hidden_size
        self.d_model           = top_k_freqs * freq_hidden_size
        self.cond_dim          = cond_dim
        self.flat_out_dim      = top_k_freqs * in_channels
        self.is_dct            = is_dct
        self.hilbert_mode      = hilbert_mode
        (
            self.low_k_correction_ranges,
            self.low_k_correction_target_modes,
        ) = _parse_low_k_correction_spec(low_k_correction_modes, top_k_freqs)
        self.low_k_correction_modes = len(self.low_k_correction_target_modes)
        self.low_k_context_modes = (
            max((end for _, end in self.low_k_correction_ranges), default=-1) + 1
        )
        self.use_seq_conditioning = use_seq_conditioning
        self.seq_embed_dim = seq_embed_dim if use_seq_conditioning else 0
        self.use_ss_conditioning = use_ss_conditioning
        self.ss_embed_dim = ss_embed_dim if use_ss_conditioning else 0
        self.num_res_types = 21
        self.num_dssp_states = 8
        self.default_size_bins = [100, 200, 300, 400, 500, 600]

        if self.use_low_k_correction_head and self.prediction_target != "x_0":
            raise ValueError(
                "use_low_k_correction_head currently requires prediction_target='x_0' "
                "so the additive correction head can operate directly in clean-spectrum space."
            )
        if self.use_low_k_correction_head and not self.low_k_correction_ranges:
            raise ValueError(
                "use_low_k_correction_head=True requires a non-empty low_k_correction_modes spec."
            )

        if freq_scale is not None:
            self.register_buffer("freq_scale", freq_scale.float())
        else:
            self.register_buffer("freq_scale", None)

        self.spectral_adapter = SpectralAdapter(
            transform_engine=DCT if is_dct else DFT,
            scale_factors=freq_scale,
            conditioned_freq_scale=conditioned_freq_scale,
            channels=in_channels,
            coord_channels=min(cond_channels, in_channels),
        )
        self.spectral_adapter.bind_freq_scale(lambda: self.freq_scale)
        self.conditioned_freq_scale_lookup = self.spectral_adapter.conditioned_freq_scale_lookup
        self.use_scale_conditioning = self.spectral_adapter.use_scale_conditioning
        self.scale_cond_dim = 0
        self.size_bins = (
            list(self.conditioned_freq_scale_lookup.size_bins)
            if self.conditioned_freq_scale_lookup is not None
            and getattr(self.conditioned_freq_scale_lookup, "size_bins", None)
            else list(self.default_size_bins)
        )
        self.num_size_bins = len(self.size_bins) + 1
        self.size_embed_dim = max(8, min(32, freq_hidden_size * 2))

        x_embed_in = in_channels + cond_channels + self.seq_embed_dim + self.ss_embed_dim
        self.x_embed = nn.Linear(x_embed_in, freq_hidden_size)
        # Learnable per-frequency position embedding: gives x_embed a distinct
        # identity for each DCT mode so the model can learn mode-specific responses.
        self.freq_pos_embed = nn.Parameter(
            torch.randn(top_k_freqs, freq_hidden_size) * 0.02
        )

        if self.use_seq_conditioning:
            self.residue_embed = nn.Embedding(self.num_res_types, self.seq_embed_dim)
            self.seq_global_proj = nn.Linear(self.seq_embed_dim, cond_dim)
            nn.init.normal_(self.residue_embed.weight, std=0.02)
        if self.use_ss_conditioning:
            self.ss_local_proj = nn.Linear(self.num_dssp_states, self.ss_embed_dim)
            self.ss_global_proj = nn.Linear(self.ss_embed_dim, cond_dim)
        if self.use_scale_conditioning:
            modes = self.spectral_adapter.scale_condition_modes + 1
            chans = self.spectral_adapter.scale_condition_channels
            self.scale_cond_dim = modes * chans
            self.scale_global_proj = nn.Linear(self.scale_cond_dim, cond_dim)
        self.size_embed = nn.Embedding(self.num_size_bins, self.size_embed_dim)
        self.size_global_proj = nn.Linear(self.size_embed_dim, cond_dim)
        nn.init.normal_(self.size_embed.weight, std=0.02)

        # cond_dim is decoupled from d_model — scalar signals don't need d_model dimensions
        self.time_embedder = ScalarEmbedding(cond_dim, max_period=10000)
        self.temp_embedder = SmoothScalarEmbedding(cond_dim)
        self.pos_embedder  = SmoothScalarEmbedding(cond_dim)
        # Additive enrichment: joint MLP over (win_pos, temp, size).
        # Zero-init final layer => step-0 output is 0, preserving warm-start parity.
        self.window_ctx_mlp = WindowContextEmbedding(cond_dim)

        if cfg_dropout:
            self.null_cond = nn.Parameter(torch.zeros(1, 1, cond_channels))
            self.null_temp = nn.Parameter(torch.zeros(1, cond_dim))

        self.rope   = RotaryEmbedding(self.d_model // num_heads)
        self.blocks = nn.ModuleList([
            SpectralConvBlock(
                d_model=self.d_model,
                cond_dim=cond_dim,
                freq_hidden_size=freq_hidden_size,
                n_freqs=top_k_freqs,
                num_heads=num_heads,
                spectral_modes=spectral_modes,
                attn_dropout=attn_dropout,
                use_hilbert=use_hilbert and _hilbert_enabled_for_block(hilbert_mode, i),
                use_hilbert_dct=use_hilbert_dct and _hilbert_enabled_for_block(hilbert_mode, i),
            )
            for i in range(depth)
        ])

        self.final_norm  = nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * self.d_model),
        )
        self.final_proj = nn.Linear(freq_hidden_size, in_channels)
        if self.use_low_k_correction_head:
            low_k_head_in = (
                self.low_k_context_modes * freq_hidden_size
                + cond_dim
                + self.seq_embed_dim
                + self.ss_embed_dim
            )
            heads = []
            for start, end in self.low_k_correction_ranges:
                width = end - start + 1
                head = nn.Sequential(
                    nn.Linear(low_k_head_in, freq_hidden_size),
                    nn.SiLU(),
                    nn.Linear(freq_hidden_size, width * in_channels),
                )
                nn.init.constant_(head[-1].weight, 0)
                nn.init.constant_(head[-1].bias, 0)
                heads.append(head)
            self.low_k_correction_heads = nn.ModuleList(heads)

        if prediction_target in ("v", "noise"):
            nn.init.constant_(self.final_proj.weight, 0)
            nn.init.constant_(self.final_proj.bias,   0)

        # Optional NMA RMSF prior output gain (Mechanism A from NMA_Fourier.md).
        # Scalar, zero-init → gain = rmsf_prior^0 = 1 everywhere at start, so the
        # model's output is bit-identical to a non-prior model at step 0. As the
        # gate grows during fine-tuning, output is multiplied by
        #     (rmsf_prior / mean_rmsf_prior)^gate
        # per residue, biasing the predicted amplitude toward the structure-
        # derived flexibility envelope. State-dict compatible: when loading a
        # checkpoint trained without this flag, use strict=False and the new
        # `rmsf_gate` parameter stays at its zero init → identity.
        if use_rmsf_prior_gain:
            self.rmsf_gate = nn.Parameter(torch.zeros(1))

    def _resolve_scale(self, x: torch.Tensor, freq_scale_override: torch.Tensor | None = None):
        return self.spectral_adapter.resolve_scale(
            x, freq_scale_override=freq_scale_override, base_scale=self.freq_scale
        )

    def _normalise(self, x, freq_scale_override: torch.Tensor | None = None):
        return self.spectral_adapter.normalise(
            x, freq_scale_override=freq_scale_override, base_scale=self.freq_scale
        )

    def _denormalise(self, x, freq_scale_override: torch.Tensor | None = None):
        return self.spectral_adapter.denormalise(
            x, freq_scale_override=freq_scale_override, base_scale=self.freq_scale
        )

    def _mode_scale_override(
        self,
        start_mode: int,
        num_modes: int,
        freq_scale_override: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        scale = freq_scale_override if freq_scale_override is not None else self.freq_scale
        if scale is None:
            return None

        start = start_mode * self.in_channels
        end = (start_mode + num_modes) * self.in_channels
        if scale.ndim == 1:
            return scale[start:end]
        if scale.ndim == 2:
            return scale[:, start:end]
        raise ValueError(f"freq scale must be 1D or 2D, got {tuple(scale.shape)}")

    def _denormalise_modes(self, x, freq_scale_override: torch.Tensor | None = None):
        return self.spectral_adapter.denormalise_modes(
            x,
            top_k_freqs=self.top_k_freqs,
            in_channels=self.in_channels,
            freq_scale_override=freq_scale_override,
            base_scale=self.freq_scale,
        )

    def _build_cond(
        self,
        t,
        temp,
        win_pos,
        cond_drop_mask=None,
        size_global: torch.Tensor | None = None,
        seq_global: torch.Tensor | None = None,
        ss_global: torch.Tensor | None = None,
        scale_global: torch.Tensor | None = None,
        size_scalar: torch.Tensor | None = None,
    ):
        t_emb   = self.time_embedder(t)
        tau_emb = self.temp_embedder(temp)
        s_emb   = self.pos_embedder(win_pos)
        # CFG dropout: null only temperature (and native_coords, handled
        # upstream in ``forward``). Window position, size, sequence, DSSP,
        # and scale-conditioning signals are metadata / identity signals
        # without a useful "unconditional" branch, so we keep them always on.
        if self.cfg_dropout and cond_drop_mask is not None:
            dm      = cond_drop_mask.view(-1, 1)
            tau_emb = torch.where(dm, self.null_temp, tau_emb)
        if size_scalar is None:
            size_scalar = torch.zeros_like(win_pos)
        # Recover Kelvin: train passes norm_temps = clip((T - 250) / 200, 0, 1).
        temp_k = temp.float() * 200.0 + 250.0
        ctx_emb = self.window_ctx_mlp(win_pos, temp_k, size_scalar)
        c = t_emb + tau_emb + s_emb + ctx_emb
        if size_global is not None:
            c = c + size_global
        if seq_global is not None:
            c = c + seq_global
        if ss_global is not None:
            c = c + ss_global
        if scale_global is not None:
            c = c + scale_global
        return c

    def _size_features(
        self,
        mask: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            lengths = mask.float().sum(dim=1).long().tolist()
        else:
            lengths = [int(L)] * B

        idxs: list[int] = []
        for length in lengths:
            label = size_bin_label(int(length), self.size_bins)
            if label.startswith("le_"):
                cutoff = int(label.split("_", 1)[1])
                try:
                    idx = self.size_bins.index(cutoff)
                except ValueError:
                    idx = len(self.size_bins)
            else:
                idx = len(self.size_bins)
            idxs.append(idx)

        size_idx = torch.tensor(idxs, device=device, dtype=torch.long)
        size_local = self.size_embed(size_idx)
        size_global = self.size_global_proj(size_local)
        return size_local, size_global

    def _sequence_features(
        self,
        res_type: torch.Tensor | None,
        mask: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_seq_conditioning:
            raise RuntimeError("_sequence_features called with use_seq_conditioning=False")

        if res_type is None:
            seq_local = torch.zeros(B, L, self.seq_embed_dim, device=device)
        else:
            if res_type.ndim == 3:
                res_idx = res_type.argmax(dim=-1)
            elif res_type.ndim == 2:
                res_idx = res_type
            else:
                raise ValueError(f"res_type must be 2D or 3D, got {tuple(res_type.shape)}")
            res_idx = res_idx.long().clamp(min=0, max=self.num_res_types - 1)
            seq_local = self.residue_embed(res_idx)

        if mask is not None:
            seq_local = seq_local * mask.unsqueeze(-1).float()
            denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            seq_pooled = seq_local.sum(dim=1) / denom
        else:
            seq_pooled = seq_local.mean(dim=1)

        seq_global = self.seq_global_proj(seq_pooled)
        return seq_local, seq_global

    def _ss_features(
        self,
        dssp: torch.Tensor | None,
        mask: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_ss_conditioning:
            raise RuntimeError("_ss_features called with use_ss_conditioning=False")

        if dssp is None:
            ss_base = torch.zeros(B, L, self.num_dssp_states, device=device)
            ss_base[..., -1] = 1.0
        else:
            if dssp.ndim == 2:
                ss_base = F.one_hot(
                    dssp.long().clamp(min=0, max=self.num_dssp_states - 1),
                    num_classes=self.num_dssp_states,
                ).float()
            elif dssp.ndim == 3:
                ss_base = dssp.float()
            else:
                raise ValueError(f"dssp must be 2D or 3D, got {tuple(dssp.shape)}")
        ss_local = self.ss_local_proj(ss_base)
        if mask is not None:
            ss_local = ss_local * mask.unsqueeze(-1).float()
            denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            ss_pooled = ss_local.sum(dim=1) / denom
        else:
            ss_pooled = ss_local.mean(dim=1)
        ss_global = self.ss_global_proj(ss_pooled)
        return ss_local, ss_global

    def _rmsf_gain(self, rmsf_prior: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        '''Compute the per-residue output gain from the NMA RMSF prior.

        Returns a tensor of shape (B, L, 1, 1) to broadcast over (K, C_in) on
        the post-projection output. At init ``rmsf_gate = 0`` the gain is 1.0
        everywhere (identity). As the gate grows the gain approaches the
        protein-specific relative flexibility envelope
        ``rmsf_prior / mean_rmsf_prior`` per residue.

        Args:
            rmsf_prior: (B, L) per-residue unitless ANM RMSF.
            mask:       (B, L) optional validity mask used to compute the
                        per-protein mean only over valid residues.

        Returns:
            (B, L, 1, 1) gain tensor.
        '''
        prior = torch.nan_to_num(
            rmsf_prior.float(),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        ).clamp(min=1e-8, max=1e8)
        if mask is not None:
            m = mask.float()
            denom = m.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean = (prior * m).sum(dim=-1, keepdim=True) / denom
        else:
            mean = prior.mean(dim=-1, keepdim=True)
        mean = mean.clamp(min=1e-8)
        log_rel = torch.log(prior / mean).clamp(min=-8.0, max=8.0)       # (B, L)
        log_gain = (self.rmsf_gate.float() * log_rel).clamp(min=-8.0, max=8.0)
        gain = torch.exp(log_gain).to(rmsf_prior.dtype)                 # (B, L)
        return gain.unsqueeze(-1).unsqueeze(-1)                         # (B, L, 1, 1)

    def apply_rmsf_prior_gain(
        self,
        x: torch.Tensor,
        rmsf_prior: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        '''Apply the NMA RMSF gain to an x_0-like tensor.

        The prior is physically a clean-sample amplitude prior, so it should
        act on predicted x_0 rather than directly on v/noise targets.

        Args:
            x: Either flattened spectral features ``(B, L, D)`` or unflattened
                ``(B, L, K, C)`` clean-sample-like tensor.
            rmsf_prior: (B, L) per-residue unitless ANM RMSF.
            mask: Optional (B, L) validity mask used for the per-protein mean.

        Returns:
            Tensor with the same shape as ``x``.
        '''
        if not self.use_rmsf_prior_gain or rmsf_prior is None:
            return x

        gain = self._rmsf_gain(rmsf_prior, mask=mask)
        if x.ndim == 4:
            return x * gain
        if x.ndim == 3:
            return x * gain.view(x.shape[0], x.shape[1], 1)
        raise ValueError(f"apply_rmsf_prior_gain expected 3D or 4D input, got {tuple(x.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        temp: torch.Tensor,
        native_coords: torch.Tensor,
        mask: torch.Tensor = None,
        win_pos: torch.Tensor = None,
        cond_drop_mask: torch.Tensor = None,
        native_angles: torch.Tensor = None,
        res_type: torch.Tensor = None,
        dssp: torch.Tensor = None,
        rmsf_prior: torch.Tensor = None,
        freq_scale_override: torch.Tensor = None,
        scale_cond: torch.Tensor = None,
        extra_global_cond: torch.Tensor = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | int]:
        B, L, _ = x.shape

        # Normalise to unit-ish variance internally (÷ freq_scale).
        # Output is denormalised before return so loss and diffusion stay in raw coefficient scale.
        x = self._normalise(x, freq_scale_override=freq_scale_override)

        if win_pos is None:
            win_pos = torch.zeros(B, device=x.device, dtype=x.dtype)

        native = native_coords
        if self.cfg_dropout and cond_drop_mask is not None:
            dm     = cond_drop_mask.view(-1, 1, 1)
            native = torch.where(dm, self.null_cond.expand(B, L, -1), native)
            if mask is not None:
                native = native * mask.unsqueeze(-1)

        # CFG dropout scope: native_coords (above) and temperature
        # (inside _build_cond) only. size / seq / dssp / scale are
        # metadata / identity signals whose unconditional branch is not
        # semantically meaningful, so they are always on.
        seq_local = None
        seq_global = None
        if self.use_seq_conditioning:
            seq_local, seq_global = self._sequence_features(
                res_type, mask, B, L, x.device
            )

        ss_local = None
        ss_global = None
        if self.use_ss_conditioning:
            ss_local, ss_global = self._ss_features(
                dssp, mask, B, L, x.device
            )

        scale_global = None
        if self.use_scale_conditioning:
            if scale_cond is None:
                scale_cond = torch.zeros(B, self.scale_cond_dim, device=x.device, dtype=x.dtype)
            scale_cond = torch.log(scale_cond.clamp(min=1e-8)).float()
            scale_global = self.scale_global_proj(scale_cond)

        _, size_global = self._size_features(mask, B, L, x.device)

        x_kc    = x.view(B, L, self.top_k_freqs, self.in_channels)
        # Normalise native coords to ~unit scale before concatenation.
        nat_exp = native.unsqueeze(2).expand(-1, -1, self.top_k_freqs, -1) / 10.0
        embed_inputs = [x_kc, nat_exp]
        if seq_local is not None:
            seq_exp = seq_local.unsqueeze(2).expand(-1, -1, self.top_k_freqs, -1)
            embed_inputs.append(seq_exp)
        if ss_local is not None:
            ss_exp = ss_local.unsqueeze(2).expand(-1, -1, self.top_k_freqs, -1)
            embed_inputs.append(ss_exp)
        per_mode = self.x_embed(torch.cat(embed_inputs, dim=-1))  # (B, L, K, H)
        # Add frequency position embedding so the model can distinguish DCT modes.
        per_mode = per_mode + self.freq_pos_embed.view(1, 1, self.top_k_freqs, self.freq_hidden_size)
        tokens  = per_mode.view(B, L, self.d_model)

        # Raw per-sample size (number of valid residues) for the context MLP.
        if mask is not None:
            size_scalar = mask.float().sum(dim=1)
        else:
            size_scalar = torch.full((B,), float(L), device=x.device, dtype=x.dtype)

        c = self._build_cond(
            t, temp, win_pos, cond_drop_mask,
            size_global=size_global,
            seq_global=seq_global,
            ss_global=ss_global,
            scale_global=scale_global,
            size_scalar=size_scalar,
        )
        if extra_global_cond is not None:
            c = c + extra_global_cond.to(device=c.device, dtype=c.dtype)
        rope_freqs = self.rope(tokens)

        for block in self.blocks:
            tokens = block(tokens, c, rope_freqs, mask=mask)

        tokens = self.final_norm(tokens)
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        token_grid = tokens.view(B, L, self.top_k_freqs, self.freq_hidden_size)
        out = self.final_proj(token_grid)
        low_k_correction = None
        low_k_correction_dc = None
        if self.use_low_k_correction_head:
            context_tokens = token_grid[:, :, :self.low_k_context_modes, :].reshape(B, L, -1)
            head_inputs = [context_tokens, c.unsqueeze(1).expand(-1, L, -1)]
            if seq_local is not None:
                head_inputs.append(seq_local)
            if ss_local is not None:
                head_inputs.append(ss_local)
            head_input = torch.cat(head_inputs, dim=-1)
            correction_chunks = []
            for (start, end), head in zip(self.low_k_correction_ranges, self.low_k_correction_heads):
                width = end - start + 1
                band_correction = head(head_input).view(B, L, width, self.in_channels)
                out[:, :, start:end + 1, :] = out[:, :, start:end + 1, :] + band_correction
                correction_chunks.append(band_correction)
                if start == 0:
                    low_k_correction_dc = band_correction[:, :, 0, :]
            if correction_chunks:
                low_k_correction = torch.cat(correction_chunks, dim=2)
        # Optional NMA RMSF prior gain. This is a clean-sample amplitude prior,
        # so for v/noise prediction it is applied later, after recovering x_0
        # from the target-space prediction inside the training / sampling code.
        if self.use_rmsf_prior_gain and rmsf_prior is not None and self.prediction_target == "x_0":
            out = self.apply_rmsf_prior_gain(out, rmsf_prior, mask=mask)
        out = self._denormalise_modes(out, freq_scale_override=freq_scale_override).view(B, L, self.flat_out_dim)
        if low_k_correction is not None:
            low_k_correction = self._denormalise_modes(
                low_k_correction, freq_scale_override=freq_scale_override
            ).reshape(
                B, L, self.low_k_correction_modes * self.in_channels
            )
        if low_k_correction_dc is not None:
            low_k_correction_dc = self._denormalise(
                low_k_correction_dc,
                freq_scale_override=self._mode_scale_override(
                    start_mode=0,
                    num_modes=1,
                    freq_scale_override=freq_scale_override,
                ),
            )

        if mask is not None:
            out = out * mask.unsqueeze(-1)
            if low_k_correction is not None:
                low_k_correction = low_k_correction * mask.unsqueeze(-1)
            if low_k_correction_dc is not None:
                low_k_correction_dc = low_k_correction_dc * mask.unsqueeze(-1)

        if return_aux:
            result = {"pred": out}
            if low_k_correction is not None:
                result["low_k_correction"] = low_k_correction
                result["low_k_correction_modes"] = self.low_k_correction_modes
                result["low_k_correction_specs"] = list(self.low_k_correction_ranges)
            if low_k_correction_dc is not None:
                result["low_k_correction_dc"] = low_k_correction_dc
            return result

        return out
