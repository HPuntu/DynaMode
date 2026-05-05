'''
Baseline Spectral DiT — standard DiT with per-frequency normalisation.

Input representation
--------------------
Each residue is represented as a flattened spectral volume of shape
(K * C_in,) where K = top_k DCT modes and C_in = coordinate channels.
Per-frequency normalisation (95th-percentile amplitudes from training data,
passed as `freq_scale`) maps this to unit-ish variance across all modes before
the model sees it. Without this normalisation the dynamic range across modes is
~5e4, making gradient flow impossible in practice.

Block structure (per layer)
---------------------------
1. Residue self-attention  + AdaLN-Zero
2. SwiGLU feed-forward     + AdaLN-Zero

Conditioning
------------
Three-term conditioning vector: c = f_t(t_diff) + f_τ(τ) + f_s(s)
  t_diff : diffusion timestep  (ScalarEmbedding — sinusoidal)
  τ      : MD temperature      (SmoothScalarEmbedding — smooth MLP)
  s      : normalised window position n_0/N ∈ [0,1]  (SmoothScalarEmbedding)

s is essential for non-equilibrium training: p(X | τ, s=0) ≠ p(X | τ, s=0.9)
at high temperature. When not provided it defaults to zeros (s=0 everywhere),
which is correct for equilibrium-only datasets.

Prediction target
-----------------
v-prediction: v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0.
Approximately constant magnitude across all diffusion timesteps, giving more
uniform loss weighting than noise or x_0 prediction. Zero-init of the final
projection ensures near-identity behaviour at initialisation.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.spectral.adapters import SpectralAdapter, DCT, DFT
from src.models.conditioned_freq_scale import size_bin_label

from src.models.modules import (
    RotaryEmbedding,
    ScalarEmbedding,
    SmoothScalarEmbedding,
    SwiGLU,
    apply_rotary_pos_emb,
)
from src.models.models.spectral_conv import _parse_low_k_correction_spec


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    '''Standard DiT block: residue attention + SwiGLU FFN with AdaLN-Zero.

    Args:
        d_model: Token dimension D = K * freq_hidden_size.
        cond_dim: Conditioning vector dimension (decoupled from d_model).
        num_heads: Attention heads; must divide d_model.
        mlp_ratio: FFN hidden width = mlp_ratio * d_model.
        dropout: Attention output dropout.
    '''

    def __init__(self, d_model: int, cond_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        mlp_hidden = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.qkv   = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj  = nn.Linear(d_model, d_model, bias=False)
        self.drop  = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn   = SwiGLU(d_model, mlp_hidden)

        # AdaLN-Zero: cond_dim → 6 * d_model (decoupled from token width)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model, bias=True),
        )
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias,   0)

    def forward(
        self,
        x: torch.Tensor,           # (B, L, D)
        c: torch.Tensor,           # (B, D)  conditioning vector
        rope_freqs: torch.Tensor,  # RoPE frequencies
        mask: torch.Tensor = None, # (B, L) validity mask
    ) -> torch.Tensor:
        B, L, D = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=1)

        # --- Attention ---
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

        # --- FFN ---
        xn = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.ffn(xn)
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SpectralDiT(nn.Module):
    '''Baseline spectral diffusion transformer for protein trajectory diffusion.

    Input: raw DCT coefficients of coordinate deviations (x - native), shape (B, L, K*C_in).
    Normalisation (÷ freq_scale) is applied internally so the model operates in unit-variance
    space; the output is denormalised before return so the diffusion process and loss both
    stay in raw coefficient scale.
    Each residue token = flattened (K, C_in) coefficients projected per-frequency
    to (K, freq_hidden_size) then flattened to D = K * freq_hidden_size.

    Args:
        top_k_freqs: Number of retained DCT modes K.
        in_channels: Coordinate channels per mode (3 for xyz, 7 with torsion).
        cond_channels: Native structure conditioning channels (same as in_channels).
        freq_hidden_size: Per-frequency latent width H; D = K * H.
        depth: Number of DiTBlocks.
        num_heads: Attention heads; must divide D.
        mlp_ratio: FFN expansion factor.
        attn_dropout: Attention output dropout for regularisation. Applied inside
            each DiTBlock regardless of CFG. Recommend 0.1 for small datasets
            (<500 proteins); 0.0 for large-scale training where data is the
            primary regulariser.
        freq_scale: Per-feature 95th-percentile amplitudes, shape (K*C_in,).
                    If None, no per-frequency normalisation is applied.
        cfg_dropout: Enable classifier-free guidance conditioning dropout.
            Randomly nulls native_coords/temp/win_pos during training so the
            model learns both conditional and unconditional distributions,
            enabling guided sampling at inference. Unrelated to attn_dropout.
        prediction_target: "v", "x_0", or "noise". Recommend "v".
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
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        freq_scale: torch.Tensor = None,
        conditioned_freq_scale: dict | None = None,
        cfg_dropout: bool = False,
        prediction_target: str = "v",
        is_dct: bool = True,
        use_seq_conditioning: bool = False,
        seq_embed_dim: int = 16,
        use_ss_conditioning: bool = False,
        ss_embed_dim: int = 8,
        use_low_k_correction_head: bool = False,
        low_k_correction_modes: int | str = 1,
        cond_dim: int = 512,
    ):
        super().__init__()
        assert prediction_target in ("v", "x_0", "noise")

        self.prediction_target = prediction_target
        self.cfg_dropout      = cfg_dropout
        self.use_low_k_correction_head = use_low_k_correction_head
        self.top_k_freqs      = top_k_freqs
        self.in_channels      = in_channels
        self.freq_hidden_size = freq_hidden_size
        self.d_model          = top_k_freqs * freq_hidden_size
        self.cond_dim         = cond_dim
        self.flat_out_dim     = top_k_freqs * in_channels
        self.is_dct           = is_dct  # compatibility flag for shared inference helpers
        self.use_seq_conditioning = use_seq_conditioning
        self.seq_embed_dim = seq_embed_dim if use_seq_conditioning else 0
        self.use_ss_conditioning = use_ss_conditioning
        self.ss_embed_dim = ss_embed_dim if use_ss_conditioning else 0
        self.num_res_types = 21
        self.num_dssp_states = 8
        self.default_size_bins = [100, 200, 300, 400, 500, 600]
        (
            self.low_k_correction_ranges,
            self.low_k_correction_target_modes,
        ) = _parse_low_k_correction_spec(low_k_correction_modes, top_k_freqs)
        self.low_k_correction_modes = len(self.low_k_correction_target_modes)
        self.low_k_context_modes = (
            max((end for _, end in self.low_k_correction_ranges), default=-1) + 1
        )

        if self.use_low_k_correction_head and self.prediction_target != "x_0":
            raise ValueError(
                "use_low_k_correction_head currently requires prediction_target='x_0' "
                "so the additive correction head can operate directly in clean-spectrum space."
            )
        if self.use_low_k_correction_head and not self.low_k_correction_ranges:
            raise ValueError(
                "use_low_k_correction_head=True requires a non-empty low_k_correction_modes spec."
            )

        # Per-frequency normalisation scale
        if freq_scale is not None:
            assert freq_scale.shape[0] == top_k_freqs * in_channels, (
                f"freq_scale must have shape ({top_k_freqs * in_channels},), "
                f"got {freq_scale.shape}"
            )
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

        # Per-frequency per-channel → per-frequency latent
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

        # Three-term conditioning: diffusion step + MD temperature + window position
        # cond_dim is decoupled from d_model — scalar signals don't need d_model dimensions
        self.time_embedder = ScalarEmbedding(cond_dim, max_period=10000)
        self.temp_embedder = SmoothScalarEmbedding(cond_dim)
        self.pos_embedder  = SmoothScalarEmbedding(cond_dim)

        if cfg_dropout:
            self.null_cond   = nn.Parameter(torch.zeros(1, 1, cond_channels))
            self.null_temp   = nn.Parameter(torch.zeros(1, cond_dim))
            self.null_winpos = nn.Parameter(torch.zeros(1, cond_dim))
            self.null_size_global = nn.Parameter(torch.zeros(1, cond_dim))
            if self.use_seq_conditioning:
                self.null_seq_local = nn.Parameter(torch.zeros(1, 1, self.seq_embed_dim))
                self.null_seq_global = nn.Parameter(torch.zeros(1, cond_dim))
            if self.use_ss_conditioning:
                self.null_ss_local = nn.Parameter(torch.zeros(1, 1, self.ss_embed_dim))
                self.null_ss_global = nn.Parameter(torch.zeros(1, cond_dim))
            if self.use_scale_conditioning:
                self.null_scale_global = nn.Parameter(torch.zeros(1, cond_dim))

        self.rope   = RotaryEmbedding(self.d_model // num_heads)
        self.blocks = nn.ModuleList([
            DiTBlock(self.d_model, cond_dim, num_heads, mlp_ratio, dropout=attn_dropout)
            for _ in range(depth)
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_scale(self, x: torch.Tensor, freq_scale_override: torch.Tensor | None = None):
        return self.spectral_adapter.resolve_scale(
            x, freq_scale_override=freq_scale_override, base_scale=self.freq_scale
        )

    def _normalise(self, x: torch.Tensor, freq_scale_override: torch.Tensor | None = None) -> torch.Tensor:
        return self.spectral_adapter.normalise(
            x, freq_scale_override=freq_scale_override, base_scale=self.freq_scale
        )

    def _denormalise(self, x: torch.Tensor, freq_scale_override: torch.Tensor | None = None) -> torch.Tensor:
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
        t: torch.Tensor,
        temp: torch.Tensor,
        win_pos: torch.Tensor,
        cond_drop_mask: torch.Tensor = None,
        size_global: torch.Tensor | None = None,
        seq_global: torch.Tensor | None = None,
        ss_global: torch.Tensor | None = None,
        scale_global: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb    = self.time_embedder(t)
        tau_emb  = self.temp_embedder(temp)
        s_emb    = self.pos_embedder(win_pos)

        if self.cfg_dropout and cond_drop_mask is not None:
            dm = cond_drop_mask.view(-1, 1)
            tau_emb = torch.where(dm, self.null_temp,   tau_emb)
            s_emb   = torch.where(dm, self.null_winpos, s_emb)

        c = t_emb + tau_emb + s_emb
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                    # (B, L, K*C_in)  raw DCT coefficients of deviations
        t: torch.Tensor,                    # (B,)  diffusion timestep  [long]
        temp: torch.Tensor,                 # (B,)  MD temperature, normalised to [0,1]
        native_coords: torch.Tensor,        # (B, L, cond_channels)
        mask: torch.Tensor = None,          # (B, L)
        win_pos: torch.Tensor = None,       # (B,)  window position s ∈ [0,1]
        cond_drop_mask: torch.Tensor = None,
        native_angles: torch.Tensor = None, # unused; accepted for API compatibility
        res_type: torch.Tensor = None,
        dssp: torch.Tensor = None,
        freq_scale_override: torch.Tensor = None,
        scale_cond: torch.Tensor = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        # 1. Normalise to unit-ish variance internally (÷ freq_scale).
        #    The model operates in this normalised space; output is denormalised
        #    before return so loss and diffusion both stay in raw coefficient scale.
        x = self._normalise(x, freq_scale_override=freq_scale_override)

        # 2. Default win_pos to 0 (s=0 ≡ trajectory start, equilibrium)
        if win_pos is None:
            win_pos = torch.zeros(B, device=x.device, dtype=x.dtype)

        # 3. CFG dropout on native conditioning
        native = native_coords
        if self.cfg_dropout and cond_drop_mask is not None:
            dm = cond_drop_mask.view(-1, 1, 1)
            native = torch.where(dm, self.null_cond.expand(B, L, -1), native)
            if mask is not None:
                native = native * mask.unsqueeze(-1)

        seq_local = None
        seq_global = None
        if self.use_seq_conditioning:
            seq_local, seq_global = self._sequence_features(
                res_type, mask, B, L, x.device
            )
            if self.cfg_dropout and cond_drop_mask is not None:
                dm_local = cond_drop_mask.view(B, 1, 1)
                dm_global = cond_drop_mask.view(B, 1)
                seq_local = torch.where(dm_local, self.null_seq_local.expand(B, L, -1), seq_local)
                seq_global = torch.where(dm_global, self.null_seq_global.expand(B, -1), seq_global)
                if mask is not None:
                    seq_local = seq_local * mask.unsqueeze(-1)

        ss_local = None
        ss_global = None
        if self.use_ss_conditioning:
            ss_local, ss_global = self._ss_features(
                dssp, mask, B, L, x.device
            )
            if self.cfg_dropout and cond_drop_mask is not None:
                dm_local = cond_drop_mask.view(B, 1, 1)
                dm_global = cond_drop_mask.view(B, 1)
                ss_local = torch.where(dm_local, self.null_ss_local.expand(B, L, -1), ss_local)
                ss_global = torch.where(dm_global, self.null_ss_global.expand(B, -1), ss_global)
                if mask is not None:
                    ss_local = ss_local * mask.unsqueeze(-1)

        scale_global = None
        if self.use_scale_conditioning:
            if scale_cond is None:
                scale_cond = torch.zeros(B, self.scale_cond_dim, device=x.device, dtype=x.dtype)
            scale_cond = torch.log(scale_cond.clamp(min=1e-8)).float()
            scale_global = self.scale_global_proj(scale_cond)
            if self.cfg_dropout and cond_drop_mask is not None:
                dm_global = cond_drop_mask.view(B, 1)
                scale_global = torch.where(dm_global, self.null_scale_global.expand(B, -1), scale_global)

        _, size_global = self._size_features(mask, B, L, x.device)
        if self.cfg_dropout and cond_drop_mask is not None:
            dm_global = cond_drop_mask.view(B, 1)
            size_global = torch.where(dm_global, self.null_size_global.expand(B, -1), size_global)

        # 4. Embed: (B, L, K, C_in + C_cond) → (B, L, K, H) → (B, L, D)
        x_kc    = x.view(B, L, self.top_k_freqs, self.in_channels)
        # Normalise native coords to ~unit scale before concatenation.
        # Raw native coords have std ≈ 10–30 Å vs normalised spectral std ≈ 1;
        # without this the linear learns to ignore the spectral input entirely.
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

        # 5. Conditioning vector
        c = self._build_cond(
            t, temp, win_pos, cond_drop_mask,
            size_global=size_global,
            seq_global=seq_global, ss_global=ss_global, scale_global=scale_global,
        )

        # 6. Transformer blocks
        rope_freqs = self.rope(tokens)
        for block in self.blocks:
            tokens = block(tokens, c, rope_freqs, mask=mask)

        # 7. Final norm + adaLN
        tokens = self.final_norm(tokens)
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 8. Project back: (B, L, K, H) → (B, L, K, C_in), with optional
        # additive low-k specialist correction in normalised coefficient space.
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

        # 9. Denormalise to spectral-coefficient scale
        out = self._denormalise_modes(out, freq_scale_override=freq_scale_override).view(B, L, self.flat_out_dim)
        if low_k_correction is not None:
            low_k_correction = self._denormalise_modes(
                low_k_correction, freq_scale_override=freq_scale_override
            ).reshape(B, L, self.low_k_correction_modes * self.in_channels)
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

    
