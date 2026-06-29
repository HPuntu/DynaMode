import torch
import torch.nn as nn
import torch.nn.functional as F



class HilbertSpatialEnvelope(nn.Module):
    '''
    Per-mode spatial amplitude envelope via Hilbert transform along L.

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

    freq_hidden_size = Per-frequency hidden width H.
    n_freqs = Number of DCT modes K.
    '''

    def __init__(self, freq_hidden_size: int, n_freqs: int):
        super().__init__()
        # Scalar gate per hidden channel, zero-initialised → identity at start
        self.gate = nn.Parameter(torch.zeros(freq_hidden_size))

    @staticmethod
    def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
        '''
        Amplitude envelope of real signal x along its last dimension.

        Uses the standard analytic-signal construction:
            1. fft along last dim
            2. double positive frequencies, zero negative frequencies
            3. ifft → complex analytic signal
            4. amplitude = |analytic|

        Returns amplitude envelope (..., L), same shape and dtype as x.
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
    '''
    Boundary-safe spatial Hilbert envelope via even-symmetric extension.

    Drop-in replacement for :class:`HilbertSpatialEnvelope` that avoids the
    periodic-boundary Gibbs artifact of the straight FFT-Hilbert. The input
    signal along the residue axis L is mirrored to a length-2L even extension
    (the same implicit extension used by the DCT-II), the standard FFT-based
    analytic signal is computed on the now-continuous periodic signal, and
    the envelope of the first L samples is returned. Mathematically equivalent
    to constructing the analytic signal in the DCT/DST basis.

    Shares the same interface and the same single learnable parameter
    gate (shape (freq_hidden_size,)) as :class:`HilbertSpatialEnvelope`,
    so a checkpoint trained with the FFT-based envelope loads into this
    module without modification — the stored gate values are carried over
    and the underlying transform is silently upgraded to the boundary-safe
    version.

    freq_hidden_size = Per-frequency hidden width H.
    n_freqs = Number of DCT modes K (unused; kept for interface parity).
    '''

    def __init__(self, freq_hidden_size: int, n_freqs: int):
        super().__init__()
        # Scalar gate per hidden channel, zero-initialised → identity at start.
        # Name and shape match HilbertSpatialEnvelope exactly so state_dicts
        # interoperate.
        self.gate = nn.Parameter(torch.zeros(freq_hidden_size))

    @staticmethod
    def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
        '''
        Amplitude envelope of real signal x along its last dimension on
        the even-symmetric extension.

        The signal is mirrored to length 2L:
            x_ext = [x[0], ..., x[L-1], x[L-1], ..., x[0]]
        which is continuous across the periodic wrap-around by construction
        (no jump at either seam). The standard FFT analytic-signal recipe
        then runs on a signal for which the periodicity assumption is valid,
        eliminating the central-amplification artifact of the raw FFT-Hilbert
        on a non-periodic residue chain.

        Returns amplitude envelope (..., L), same shape and dtype as x.
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
        xs = (B*L, H, K) — Phase 2 hidden state before spectral conv.
        B, L = batch size and residue count.

        Returns (B*L, H, K) envelope tensor scaled by learned gate.
        '''
        H, K = xs.shape[1], xs.shape[2]
        x_spatial = xs.view(B, L, H, K).permute(0, 2, 3, 1).contiguous()  # (B, H, K, L)
        env = self._hilbert_envelope(x_spatial)                            # (B, H, K, L)
        env = env.permute(0, 3, 1, 2).reshape(B * L, H, K)                # (B*L, H, K)
        return env * self.gate.view(1, H, 1)
