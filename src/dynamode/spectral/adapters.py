'''
Adapter machinery for taking a set of coordinates (raw or displacement from native, or unit chain)
and converting them to a spectral representation (DCT or DFT) and back, with optional normalization.
'''


import torch
import torch.nn.functional as F
import math

from dynamode.spectral.conditioned_freq_scale import ConditionedFreqScaleLookup



def normalize_adaptive(x_flat, scale_factors, global_scale=1.0, eps=1e-8): 
    '''
    Normalize spectral features. Auto-adapts to truncated volumes.
    global_scale = multiplicative scaling of all values
    eps = small constant to prevent division by zero
    '''

    D = x_flat.shape[-1]
    scale_safe = torch.clamp(scale_factors[:D].view(1, 1, -1), min=eps)

    x_norm = (x_flat / scale_safe) * global_scale # apply scale factors normalisation
    x_norm = torch.clamp(x_norm, -10.0, 10.0) 
    
    return x_norm

def denormalize_adaptive(x_norm, scale_factors, global_scale=1.0, eps=1e-8): 
    '''
    Inverse of normalize_adaptive. Auto-adapts to truncated volumes
    global_scale = multiplicative scaling of all values
    eps = small constant to prevent division by zero
    '''

    D = x_norm.shape[-1]
    scale_safe = torch.clamp(scale_factors[:D].view(1, 1, -1), min=eps)

    x_unscaled = torch.clamp(x_norm / global_scale, -10.0, 10.0)
    x_original = x_unscaled * scale_safe # apply scale factors normalisation
    
    return x_original

def compute_frequency_stats(loader, time_to_spectral_fn, samples=1000, top_k=None, device='cuda', whitening_strength=0.0, is_dct=True, use_angles=True, coords_type='ca', representation=None):
    '''
    Computes 95th percentile amplitude for EACH feature index.

    samples = Number of sequences to process before stopping.
    top_k = Truncation limit for frequencies.
    whitening_strength = Exponent applied to scale factors. 0.0 means NO whitening (exponent=1.0 conceptually).
    is_dct = Boolean. If True, assumes purely real DCT input (skips imaginary physics patch). If False, assumes standard DFT and applies DC/Nyquist imaginary patches.
    coords_type = 'ca' (3ch), 'bb' (12ch). Controls coord_channels used for DFT physics patch.
    '''
    coord_channels = 12 if coords_type == 'bb' else 3
    n_channels = (coord_channels + 4) if use_angles else coord_channels
    print(f"Computing Adaptive Frequency Normalization stats (Top-K={top_k}, DCT={is_dct}, Whitening={whitening_strength}, coords_type={coords_type}, n_channels={n_channels})...")
    valid_chunks = []
    count = 0
    observed_channels = None

    with torch.no_grad():
        for batch in loader:
            coords = batch['coords'].to(device)
            mask = batch['mask'].to(device)
            native = batch.get('native_coords', None)
            if representation is not None:
                if native is None:
                    raise ValueError("representation-aware frequency stats require batch['native_coords']")
                coords = representation.forward(
                    coords,
                    native.to(device),
                    mask=mask,
                    return_context=False,
                )

            if use_angles:
                angles = batch['angles'].to(device)
                x_time = torch.cat([coords, angles], dim=-1)
            else:
                x_time = coords
            observed_channels = int(x_time.shape[-1])

            x_masked = x_time * mask.unsqueeze(1).unsqueeze(-1)

            # Use respective transform (FFT/DCT + Slice)
            x_flat = time_to_spectral_fn(x_masked, top_k=top_k)
            
            x_valid = x_flat[mask] 
            if x_valid.numel() > 0:
                valid_chunks.append(x_valid.abs().cpu())
                count += x_time.shape[0]
                
            if count >= samples: break
            
    if not valid_chunks: raise ValueError("No data found!")
        
    combined = torch.cat(valid_chunks, dim=0) 
    
    # Compute 95th percentile
    scale_factors = torch.quantile(combined.float(), 0.95, dim=0)
    scale_factors = torch.clamp(scale_factors, min=0.01)

    # OPTIONAL Apply Partial Whitening
    # If whitening_strength is > 0 (e.g. 0.5), we compress the physical energy hierarchy.
    if whitening_strength is not None and whitening_strength > 0.0:
        scale_factors = scale_factors ** whitening_strength

    # Apply Physics Patch (ONLY for old DFT)
    if not is_dct:
        n_channels = observed_channels or n_channels
        D = scale_factors.shape[0]
        n_freqs = D // (n_channels * 2)

        scales_view = scale_factors.view(n_freqs, n_channels, 2).clone()
        # Fix DC Imaginary scales to 1.0
        scales_view[0, :, 1] = 1.0 
        # Fix Nyquist Imaginary scales to 1.0 (if not truncated)
        if top_k is None: 
            scales_view[-1, :, 1] = 1.0 
            
        scale_factors = scales_view.view(-1)
    
    return scale_factors.to(device)


# TRANSFORM ENGINES
class DFT:
    ''' Discrete Fourier Transform'''
    @staticmethod
    def time_to_spectral(x_time, top_k=None):
        '''(B, T, L, C) -> (B, L, D)'''
        x_freq = torch.fft.rfft(x_time, dim=1, norm='ortho') 

        if top_k is not None:
            K = min(top_k, x_freq.shape[1])
            x_freq = x_freq[:, :K, :, :]

        x_freq = x_freq.permute(0, 2, 1, 3)
        x_real_imag = torch.view_as_real(x_freq)
        
        B, L, F, C, _ = x_real_imag.shape
        return x_real_imag.reshape(B, L, F * C * 2)
    
    @staticmethod
    def spectral_to_time(x_flat, n_time_steps, n_channels=None):
        '''(B, L, D) -> (B, T, L, C)'''
        assert n_channels is not None, "n_channels must be provided explicitly"
        B, L, D = x_flat.shape
        F_total = n_time_steps // 2 + 1
        K = D // (n_channels * 2)

        x_view = x_flat.view(B, L, K, n_channels, 2)
        x_complex = torch.complex(x_view[..., 0], x_view[..., 1])
        x_complex[:, :, 0, :].imag = 0.0
        
        if K < F_total:
            pad_amount = F_total - K
            x_complex = F.pad(x_complex, (0, 0, 0, pad_amount))
        
        x_complex = x_complex.permute(0, 2, 1, 3)
        x_time = torch.fft.irfft(x_complex, n=n_time_steps, dim=1, norm='ortho')
        return x_time


class DCT:
    ''' Discrete Cosine Transform (DCT-II)'''
    _DCT_CACHE = {}

    @staticmethod
    def get_dct_matrix(N, device):
        '''Creates an orthogonal DCT-II matrix in Float64 for precision.'''
        key = (int(N), torch.device(device).type, torch.device(device).index)
        cached = DCT._DCT_CACHE.get(key)
        if cached is not None:
            return cached

        n = torch.arange(N, device=device, dtype=torch.float32)
        k = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(1)

        W = torch.cos(math.pi / N * (n + 0.5) * k)
        W[0] *= 1.0 / math.sqrt(2.0)
        W *= math.sqrt(2.0 / N)
        W = W.to(torch.float32)
        DCT._DCT_CACHE[key] = W
        return W
    
    @staticmethod
    def get_idct_matrix(N, device):
        return DCT.get_dct_matrix(N, device).t()

    @staticmethod
    def time_to_spectral(x_time, top_k=None):
        '''(B, T, L, C) -> (B, L, D)'''
        if x_time.ndim == 5:
            B, T, L, A, XYZ = x_time.shape
            x_time = x_time.reshape(B, T, L, A * XYZ)
        elif x_time.ndim == 4:
            B, T, L, C = x_time.shape
        else:
            raise ValueError(f"time_to_spectral expected 4D or 5D input, got shape {x_time.shape}")

        B, T, L, C = x_time.shape
        W = DCT.get_dct_matrix(T, x_time.device)
        x_freq = torch.einsum('btlc,kt->bklc', x_time, W)

        if top_k is not None:
            K = min(top_k, x_freq.shape[1])
            x_freq = x_freq[:, :K, :, :]

        x_freq = x_freq.permute(0, 2, 1, 3)
        B, L, F, C = x_freq.shape
        return x_freq.reshape(B, L, F * C)

    @staticmethod
    def spectral_to_time(x_flat, n_time_steps, n_channels=None):
        '''(B, L, D) -> (B, T, L, C)'''
        assert n_channels is not None, "n_channels must be provided explicitly"
        B, L, D = x_flat.shape
        K = D // n_channels

        x_freq = x_flat.view(B, L, K, n_channels)   # (B, L, K, C)
        x_freq = x_freq.permute(0, 2, 1, 3)         # (B, K, L, C)

        # Zero-pad missing higher frequency modes
        if K < n_time_steps:
            pad = torch.zeros(
                B, n_time_steps - K, L, n_channels,
                device=x_flat.device,
                dtype=x_flat.dtype,
            )
            x_freq = torch.cat([x_freq, pad], dim=1) # (B, T, L, C)
        elif K > n_time_steps:
            raise ValueError(
                f"Got K={K} frequency modes but n_time_steps={n_time_steps}. "
                "Input has more modes than the target time resolution."
            )

        W_inv = DCT.get_idct_matrix(n_time_steps, x_flat.device)
        x_time = torch.einsum('bklc,tk->btlc', x_freq, W_inv)
        return x_time


# CORE ADAPTER
class SpectralAdapter:
    '''
    Unified spectral representation helper.
    '''

    def __init__(
        self,
        transform_engine,
        scale_factors=None,
        channels=7,
        scale_multiplier=1.0,
        conditioned_freq_scale: dict | None = None,
        coord_channels: int | None = None,
        device="cpu",
        **legacy_kwargs,
    ):
        if transform_engine is None:
            raise ValueError("transform_engine is required")

        self.transform_engine = transform_engine
        self.is_dft = self.transform_engine.__name__ == "DFT"
        self.complex_multiplier = 2 if self.is_dft else 1
        self.channels = int(channels)
        self.coord_channels = int(coord_channels or min(3, self.channels))
        self.device = device
        self.scale_multiplier = float(scale_multiplier)
        self._ignored_legacy_kwargs = tuple(sorted(legacy_kwargs))

        if scale_factors is not None:
            self._freq_scale = torch.as_tensor(scale_factors).detach().float().contiguous()
        else:
            self._freq_scale = None
        self._freq_scale_getter = None

        self.conditioned_freq_scale_lookup = (
            ConditionedFreqScaleLookup(conditioned_freq_scale)
            if conditioned_freq_scale is not None
            else None
        )
        self.use_scale_conditioning = self.conditioned_freq_scale_lookup is not None
        if self.use_scale_conditioning:
            self.scale_condition_modes = self.conditioned_freq_scale_lookup.scale_condition_modes
            self.scale_condition_channels = self.conditioned_freq_scale_lookup.scale_condition_channels
        else:
            self.scale_condition_modes = 0
            self.scale_condition_channels = 0

    @property
    def freq_scale(self):
        if self._freq_scale_getter is not None:
            return self._freq_scale_getter()
        return self._freq_scale

    def bind_freq_scale(self, getter):
        self._freq_scale_getter = getter
        return self

    def time_to_spectral(self, x_time, top_k=None):
        return self.transform_engine.time_to_spectral(x_time, top_k=top_k)

    def spectral_to_time(self, x_flat, n_time_steps, n_channels=None):
        channels = self.channels if n_channels is None else n_channels
        return self.transform_engine.spectral_to_time(x_flat, n_time_steps, n_channels=channels)

    def resolve_scale(
        self,
        x: torch.Tensor,
        *,
        freq_scale_override: torch.Tensor | None = None,
        base_scale: torch.Tensor | None = None,
    ):
        scale = freq_scale_override
        if scale is None:
            scale = base_scale if base_scale is not None else self.freq_scale
        if scale is None:
            return None
        scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-8)
        if scale.ndim == 1:
            return scale.view(1, 1, -1)
        if scale.ndim == 2:
            return scale.unsqueeze(1)
        raise ValueError(f"freq scale must be 1D or 2D, got {tuple(scale.shape)}")

    def normalise(
        self,
        x: torch.Tensor,
        *,
        freq_scale_override: torch.Tensor | None = None,
        base_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.resolve_scale(x, freq_scale_override=freq_scale_override, base_scale=base_scale)
        if scale is None:
            return x
        return (x / scale) * self.scale_multiplier

    def denormalise(
        self,
        x: torch.Tensor,
        *,
        freq_scale_override: torch.Tensor | None = None,
        base_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.resolve_scale(x, freq_scale_override=freq_scale_override, base_scale=base_scale)
        if scale is None:
            return x
        return (x / self.scale_multiplier) * scale

    def denormalise_modes(
        self,
        x: torch.Tensor,
        *,
        top_k_freqs: int,
        in_channels: int,
        freq_scale_override: torch.Tensor | None = None,
        base_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"denormalise_modes expected 4D input, got {tuple(x.shape)}")
        flat_view = x.view(x.shape[0], x.shape[1], -1)
        scale = self.resolve_scale(flat_view, freq_scale_override=freq_scale_override, base_scale=base_scale)
        if scale is None:
            return x
        if scale.shape[0] == 1:
            scale = scale.view(1, top_k_freqs, in_channels)[:, : x.shape[2], :]
            return (x / self.scale_multiplier) * scale.view(1, 1, x.shape[2], in_channels)
        scale = scale.squeeze(1).view(x.shape[0], top_k_freqs, in_channels)[:, : x.shape[2], :]
        return (x / self.scale_multiplier) * scale.view(x.shape[0], 1, x.shape[2], in_channels)

    def lookup_scales(self, temp, mask, *, seq_len=None, device=None):
        if self.conditioned_freq_scale_lookup is None:
            return None
        return self.conditioned_freq_scale_lookup.lookup_scales(
            temp, mask, seq_len=seq_len, device=device
        )

    def lookup_scale_features(self, temp, mask, *, seq_len=None, device=None):
        if self.conditioned_freq_scale_lookup is None:
            return None
        return self.conditioned_freq_scale_lookup.lookup_scale_features(
            temp, mask, seq_len=seq_len, device=device
        )

    def lookup_model_conditioning(self, temp, mask, *, seq_len=None, device=None):
        return (
            self.lookup_scales(temp, mask, seq_len=seq_len, device=device),
            self.lookup_scale_features(temp, mask, seq_len=seq_len, device=device),
        )

    def lookup_dc_baselines(self, temp, mask, *, coord_channels=None, seq_len=None, device=None):
        if self.conditioned_freq_scale_lookup is None:
            return None
        return self.conditioned_freq_scale_lookup.lookup_dc_baselines(
            temp,
            mask,
            coord_channels=coord_channels or self.coord_channels,
            seq_len=seq_len,
            device=device,
        )

    def residualise_dc(self, x, temp, mask, *, coord_channels=None, seq_len=None, per_residue_baseline=None):
        if self.conditioned_freq_scale_lookup is None:
            return x, None
        return self.conditioned_freq_scale_lookup.residualise_dc(
            x,
            temp,
            mask,
            coord_channels=coord_channels or self.coord_channels,
            seq_len=seq_len,
            per_residue_baseline=per_residue_baseline,
        )

    def restore_dc(self, x, baseline, *, coord_channels=None):
        if self.conditioned_freq_scale_lookup is None or baseline is None:
            return x
        return self.conditioned_freq_scale_lookup.restore_dc(
            x,
            baseline,
            coord_channels=coord_channels or self.coord_channels,
        )

    def forward_transform(self, x_time, mask, top_k=None):
        if mask is not None:
            x_time = x_time * mask.unsqueeze(1).unsqueeze(-1)
        x_flat = self.time_to_spectral(x_time, top_k=top_k)
        if mask is not None:
            x_flat = x_flat * mask.unsqueeze(-1)
        return self.normalise(x_flat)

    def inverse_transform(
        self,
        x_norm,
        n_time_steps,
        mask=None,
        dynamic_gain=1.0,
        gain_start_k=None,
        freq_scale_override: torch.Tensor | None = None,
    ):
        del dynamic_gain, gain_start_k
        x_spec = self.denormalise(x_norm, freq_scale_override=freq_scale_override)

        if not torch.isfinite(x_spec).all():
            x_spec = torch.nan_to_num(x_spec, nan=0.0, posinf=100.0, neginf=-100.0)

        x_time = self.spectral_to_time(x_spec, n_time_steps, n_channels=self.channels)
        if mask is not None:
            x_time = x_time * mask.unsqueeze(1).unsqueeze(-1)

        result = {"coords": x_time[..., : self.coord_channels]}
        if self.channels > self.coord_channels:
            result["angles"] = x_time[..., self.coord_channels :]
        return result
    

# USAGE
# -----
# # for DFT
# adapter = SpectralAdapter(
#     transform_engine=DFT, 
#     scale_factors=dft_scales
# )

# # for DCT
# adapter = SpectralAdapter(
#     transform_engine=DCT, 
#     scale_factors=dct_scales
# )
