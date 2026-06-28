'''
Provides a central class object for diffusion. Functionality for noising and
denoising and associated hyperparameters.
'''


import torch
import torch.nn as nn
import numpy as np
import math



def make_aniso_weights(
    freq_scales, gamma=0.5, top_k=None, channels=None, legacy_direction=False
):
    '''Derive per-bin anisotropic noise multipliers from freq_scales.

    Current direction (default). w_k = (scale_k / min_scale) gamma.
    Low-k bins (with the largest scale_k) receive the largest multiplier:
    low frequencies are perturbed more heavily than high during forward
    diffusion. This is the intended, physically-motivated direction.

    Legacy direction (legacy_direction=True). w_k = (min_scale / scale_k) gamma.
    High-k bins receive the largest multiplier. This flag exists solely to
    reproduce the noise schedule that was in place before the direction was
    corrected: checkpoints trained under the legacy schedule must be evaluated
    with the same schedule, because the denoiser learned to invert one
    specific noise distribution and swapping direction at inference time
    initialises the ODE sampler from a distribution the model has never seen.

    Normalised so mean(w_k^2) = 1, keeping total noise power identical to
    the isotropic case in both directions.

    Args:
        freq_scales: 1-D tensor of shape (D,).
        gamma: Exponent controlling anisotropy strength (0 = isotropic).
        top_k: If not None, use only first top_k * channels elements.
        channels: Channels per frequency bin.
        legacy_direction: If True, invert the direction for backward
            compatibility with pre-flip checkpoints.

    Returns:
        Tensor of shape (D,) with per-bin noise multipliers.
    '''
    D = (top_k * channels) if (top_k is not None and channels is not None) else len(freq_scales)
    s = freq_scales[:D].float().clamp(min=1e-8)
    if legacy_direction:
        w = (s.min() / s) ** gamma
    else:
        w = (s / s.min()) ** gamma
    # Normalise: keep total noise power equal to isotropic case
    w = w / w.pow(2).mean().sqrt()
    return w


class SpectralDiffusion:
    def __init__(self, T=1000, device='cuda', schedule="cosine", min_snr_gamma=None, shift_value=2.0, aniso_weights=None):
        
        self.T = T
        self.device = device
        self.schedule = schedule
        self.min_snr_gamma = min_snr_gamma
        self.shift_value = shift_value
        # Optional per-bin noise multipliers for anisotropic diffusion.
        # Shape (D,); None = standard isotropic schedule.
        if aniso_weights is not None:
            self.register_aniso(aniso_weights, device)
        else:
            self.aniso_weights = None
        
        if schedule == 'linear': # Linear Schedule (Standard for standard diffusion)
            self.beta = torch.linspace(1e-4, 0.02, T).to(device)
            self.alpha = 1. - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        elif schedule == 'cosine': # Cosine schedule as proposed by Nichol & Dhariwal (2021) BUT SHIFTED SO LOW_T SNR AT t=500!!
            # 1. Get the shifted alpha_bar schedule
            alpha_bars = self.get_shifted_cosine_schedule(T, shift=shift_value).to(device)
            self.alpha_bar = alpha_bars
            
            # 2. Reverse-engineer alpha_t from alpha_bar_t
            # alpha_t = alpha_bar_t / alpha_bar_{t-1}
            alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]])
            self.alpha = alpha_bars / alpha_bar_prev
            
            # 3. Reverse-engineer beta_t
            self.beta = 1.0 - self.alpha
            
            # 4. Clamp beta for numerical stability in the reverse process variance
            self.beta = torch.clamp(self.beta, 0.0001, 0.9999)
            
            # Re-sync alpha and alpha_bar just in case the clamp altered anything
            self.alpha = 1.0 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def get_shifted_cosine_schedule(self, T, shift=2.0, s=0.008):
        '''
        Creates a Cosine noise schedule with an adjustable Log-SNR shift.
        
        Args:
            T (int): Total diffusion timesteps.
            shift (float): The Log-SNR shift. Positive values push the SNR curve UP 
                        (keeping the signal intact for more timesteps).
            s (float): Small offset to prevent singularity at t=0.
            
        Returns:
            torch.Tensor: alpha_bar_t schedule of shape (T,)
        '''
        steps = torch.arange(T + 1, dtype=torch.float32)
        
        # 1. Standard cosine schedule formulation
        f_t = torch.cos(((steps / T) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
        alpha_bars = f_t / f_t[0]
        
        # Clamp to prevent mathematical singularities
        alpha_bars = torch.clamp(alpha_bars, min=1e-5, max=0.99999)
        
        # 2. Apply the Log-SNR Shift
        if shift != 0.0:
            # Convert to Log-SNR space: log( SNR ) = log( a / (1-a) )
            log_snr = torch.log(alpha_bars / (1.0 - alpha_bars))
            
            # Shift the curve up
            shifted_log_snr = log_snr + shift
            
            # Convert back to alpha_bar space using the sigmoid function
            alpha_bars = torch.sigmoid(shifted_log_snr)
            
        # Drop the t=0 state to return exactly T steps
        return alpha_bars[1:]

    def register_aniso(self, aniso_weights, device):
        '''Store aniso_weights as a plain tensor (not nn.Parameter — diffusion is not an nn.Module).'''
        self.aniso_weights = aniso_weights.float().to(device)

    def to(self, device):
        '''Move schedule tensors and aniso_weights to a new device.'''
        self.device = device
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha     = self.alpha.to(device)
        self.beta      = self.beta.to(device)
        if self.aniso_weights is not None:
            self.aniso_weights = self.aniso_weights.to(device)
        return self

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)

    def sample_initial_noise(self, shape, device=None):
        '''Sample the prior noise x_T ~ N(0, w^2) for DDIM initialisation.

        With anisotropic weights each bin k is scaled by w_k.  The schedule
        resolver decides whether those weights are high-k-heavy, low-k-heavy,
        or targeted to model-space SNR crossings.
        For isotropic diffusion this is identical to torch.randn(shape).
        '''
        dev = device or self.device
        noise = torch.randn(shape, device=dev)
        if self.aniso_weights is not None:
            w = self.aniso_weights.to(dev)
            # Broadcast over (B, L, D) — w has shape (D,)
            noise = noise * w.view(1, 1, -1)[:, :, :shape[-1]]
        return noise

    def get_snr(self, t):
        '''Scalar SNR = alpha_bar / (1 - alpha_bar) at timestep(s) t.'''
        ab = self.alpha_bar[t]
        return ab / (1 - ab + 1e-8)

    def q_sample(self, x0, t, noise=None):
        '''Forward diffusion: add (optionally anisotropic) noise to spectral x0.

        With aniso_weights the effective per-bin SNR at timestep t is:
            SNR_k(t) = alpha_bar[t] / ((1 - alpha_bar[t]) * w_k^2)
        so bins with larger w_k corrupt faster in raw coefficient space.

        The returned noise is the *scaled* noise w * eps that the model
        is trained to predict — no changes to the loss are required.
        '''
        if noise is None:
            noise = torch.randn_like(x0)
        if self.aniso_weights is not None:
            w = self.aniso_weights.to(x0.device)
            noise = noise * w.view(1, 1, -1)[:, :, :x0.shape[-1]]

        sqrt_ab        = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1)
        sqrt_one_minus = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_one_minus * noise
        return x_t, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        '''
        Reconstruct x_0 (clean data) from x_t (noisy data) and predicted noise.
        Formula: x0 = (xt - sqrt(1-ab)*noise) / sqrt(ab)
        '''
        # Reshape for broadcasting
        if self.schedule == "cosine":
            no_div_zero = 1e-3
        else: no_div_zero = 1e-6
        sqrt_recip_ab = torch.sqrt(1.0 / (self.alpha_bar[t] + no_div_zero)).view(-1, 1, 1)
        sqrt_recipm1_ab = torch.sqrt(1.0 / self.alpha_bar[t] - 1).view(-1, 1, 1)
        
        return sqrt_recip_ab * x_t - sqrt_recipm1_ab * noise

    def _broadcast_alpha_terms(self, t, x_like):
        '''Return sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t) broadcast to x_like.'''
        ab = self.alpha_bar[t].to(device=x_like.device, dtype=x_like.dtype)
        if ab.ndim == 0:
            ab = ab.view(1)
        shape = (ab.shape[0],) + (1,) * (x_like.ndim - 1) if ab.ndim == 1 else ab.shape
        sqrt_ab = torch.sqrt(ab).view(shape)
        sqrt_one_minus = torch.sqrt(1.0 - ab).view(shape)
        return sqrt_ab, sqrt_one_minus

    def extract_x0_eps_from_prediction(self, pred, x_t, t, prediction_target):
        '''Recover x_0 and epsilon from a model prediction in any target space.'''
        sqrt_ab, sqrt_one_minus = self._broadcast_alpha_terms(t, x_t)

        if prediction_target == "v":
            x_0_pred = sqrt_ab * x_t - sqrt_one_minus * pred
            eps_pred = sqrt_ab * pred + sqrt_one_minus * x_t
        elif prediction_target == "x_0":
            x_0_pred = pred
            eps_pred = (x_t - sqrt_ab * x_0_pred) / sqrt_one_minus.clamp(min=1e-6)
        elif prediction_target == "noise":
            eps_pred = pred
            x_0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_ab.clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

        return x_0_pred, eps_pred

    def prediction_from_x0(self, x_0_pred, x_t, t, prediction_target):
        '''Convert an x_0 prediction back into the model's configured target space.'''
        sqrt_ab, sqrt_one_minus = self._broadcast_alpha_terms(t, x_t)

        if prediction_target == "x_0":
            return x_0_pred
        if prediction_target == "noise":
            return (x_t - sqrt_ab * x_0_pred) / sqrt_one_minus.clamp(min=1e-6)
        if prediction_target == "v":
            return (sqrt_ab * x_t - x_0_pred) / sqrt_one_minus.clamp(min=1e-6)
        raise ValueError(f"Unknown prediction_target: {prediction_target}")
    
    @torch.no_grad()
    def denoise_ode(
        self,
        model,
        input_noise,
        native_coords,
        native_angles,
        temps,
        mask,
        torsion_mask=None,
        verbose=False,
        is_dct=True,
        num_steps=50,  # Default to 50 for fast evaluation
        win_pos=None,
        rmsf_prior=None,
        res_type=None,
        dssp=None,
        feature_dim=None,
        spectral_mask=None,
        model_kwargs=None,
    ):
        '''Universal DDIM/ODE sampler for v, x_0, or noise prediction.'''
        coord_channels = native_coords.shape[-1]
        angle_channels = native_angles.shape[-1] if native_angles is not None else 0
        channels = coord_channels + angle_channels
        B, L, D = input_noise.shape
        complex_mult = 1 if is_dct else 2
        if feature_dim is not None:
            channels = int(feature_dim)
            coord_channels = int(feature_dim)
            angle_channels = 0
            torsion_mask = None
        K = D // (channels * complex_mult)

        model.eval()

        # 1. GENERATE THE FULL SPECTRAL MASK
        # Callers with non-Cartesian coordinate representations can provide an
        # already-flattened mask so sampling exactly matches the training loss
        # mask (e.g. unit-chain anchor/bond channels).
        if spectral_mask is not None:
            if spectral_mask.numel() != B * L * D:
                raise RuntimeError(
                    "spectral_mask has incompatible size for denoise_ode: "
                    f"got shape={tuple(spectral_mask.shape)} numel={spectral_mask.numel()}, "
                    f"expected {(B, L, D)} numel={B * L * D}"
                )
            full_spectral_mask = spectral_mask.reshape(B, L, D).to(self.device)
        else:
            mask_coords_expand = mask.unsqueeze(-1).expand(-1, -1, coord_channels)

            if angle_channels > 0:
                if torsion_mask is None:
                    torsion_mask = mask.unsqueeze(-1).expand(-1, -1, angle_channels)
                elif torsion_mask.dim() == 2:
                    torsion_mask = torsion_mask.unsqueeze(-1).expand(-1, -1, angle_channels)
                feature_mask = torch.cat([mask_coords_expand, torsion_mask], dim=-1)
            else:
                feature_mask = mask_coords_expand

            if is_dct:
                full_mask = feature_mask.unsqueeze(2).expand(-1, -1, K, -1)
            else:
                full_mask = feature_mask.unsqueeze(2).unsqueeze(-1).expand(-1, -1, K, -1, 2)

            full_spectral_mask = full_mask.reshape(B, L, D).to(self.device)

        x_t = input_noise * full_spectral_mask
        #print(full_spectral_mask.shape, x_t.shape)
        if num_steps is None:
            num_steps = self.T
            
        times = torch.linspace(self.T - 1, 0, num_steps, dtype=torch.long, device=self.device)
        base_model = getattr(model, "base_model", model)
        inner = getattr(base_model, "model", base_model)
        model_kwargs = model_kwargs or {}

        # 3. ODE INTEGRATION LOOP
        for i in range(len(times)):
            t_current = times[i]
            t_batch = torch.full((B,), t_current, device=self.device, dtype=torch.long)
            
            # Forward Pass
            pred = model(
                x_t, t_batch, temps, native_coords, native_angles,
                mask=mask, win_pos=win_pos,
                rmsf_prior=rmsf_prior, res_type=res_type, dssp=dssp,
                **model_kwargs,
            )

            x_0_pred, eps_pred = self.extract_x0_eps_from_prediction(
                pred, x_t, t_batch, model.prediction_target
            )

            # The NMA RMSF prior is a clean-sample amplitude prior. For
            # v/noise prediction we therefore apply it after converting to x_0,
            # then recompute epsilon so the DDIM update stays self-consistent.
            if model.prediction_target != "x_0" and hasattr(inner, "apply_rmsf_prior_gain"):
                x_0_pred = inner.apply_rmsf_prior_gain(x_0_pred, rmsf_prior, mask=mask)
                eps_pred = self.prediction_from_x0(x_0_pred, x_t, t_batch, "noise")

            if i == len(times) - 1:
                x_t = x_0_pred * full_spectral_mask
                break
                
            t_next = times[i+1]
            sqrt_ab_next = torch.sqrt(self.alpha_bar[t_next])
            sqrt_one_minus_ab_next = torch.sqrt(1.0 - self.alpha_bar[t_next])

            # Take the step
            x_t = (sqrt_ab_next * x_0_pred) + (sqrt_one_minus_ab_next * eps_pred)
            x_t = x_t * full_spectral_mask

        return x_t
