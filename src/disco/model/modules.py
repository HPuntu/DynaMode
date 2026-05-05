import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :]

def apply_rotary_pos_emb(x, freqs):
    # x: [B, L, H, D]
    # freqs: [1, L, 1, rot_dim] (Where rot_dim matches x_rot size)
    
    rot_dim = freqs.shape[-1]
    # Split x into the part to be rotated and the pass-through part
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    
    # Helper to swap halves: [x1, x2] -> [-x2, x1]
    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)

    c, s = freqs.cos(), freqs.sin()
    
    # Standard RoPE application
    # This works because c and s are already duplicated [cos, cos] in RotaryEmbedding
    x_rotated = (x_rot * c) + (rotate_half(x_rot) * s)
    
    return torch.cat([x_rotated, x_pass], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ScalarEmbedding(nn.Module):
    '''
    Sinusoidal feature embedding -> MLP.
    '''
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP Projection
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        
        # 1. Create Sinusoidal Frequencies
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device)
        
        # 2. Create Embeddings
        # (Batch, 1) * (1, Half) -> (Batch, Half)
        args = x[:, None].float() * freqs[None]
        
        # Concatenate sin/cos -> (Batch, Dim)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        # 3. Project
        return self.mlp(embedding)
    

class SmoothScalarEmbedding(nn.Module):
    '''
    Use continuous functional mapping over temperatures for inference using temperatures not
    seen in training data.
    '''
    def __init__(self, dim):
        super().__init__()
        # Direct projection: Scalar -> Vector
        # We avoid Sine/Cosine to prevent high-frequency "hashing"
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        # x: (Batch,)
        # Expects x to be normalized to roughly [0, 1] range!
        return self.net(x.unsqueeze(-1))


class WindowContextEmbedding(nn.Module):
    """Joint MLP embedding of ``(win_pos, temperature, protein_size)``.

    The scalar sinusoidal window-position embedding cannot express
    interactions between window position and thermodynamic / structural
    context. This module projects a concatenation of the three raw scalars
    through a 2-layer MLP, letting the model learn joint responses such as
    "terminal windows of large proteins at high T" without a separate
    interaction term.

    The final linear is zero-initialised so the module's output is zero at
    step 0 — safe to add on top of an existing scalar window embedding when
    warm-starting from an older checkpoint.

    Inputs at call time (all ``(B,)``):
        win_pos: ``[0, 1]`` window start fraction.
        temp_k:  Temperature in Kelvin (normalised internally).
        size:    Number of valid residues (log-normalised internally).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        in_dim = 3  # win_pos, temp_norm, log_size_norm
        hidden = max(self.dim, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        win_pos: torch.Tensor,
        temp_k: torch.Tensor,
        size: torch.Tensor,
    ) -> torch.Tensor:
        B = win_pos.shape[0]
        wp = win_pos.float().view(B, 1)
        temp_norm = (temp_k.float().view(B, 1) - 320.0) / 130.0  # ~[0, 1] over [320, 450]
        size_norm = torch.log1p(size.float().view(B, 1)) / math.log(1000.0)  # ~[0, 1]
        feats = torch.cat([wp, temp_norm, size_norm], dim=-1)
        return self.net(feats)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, context, mask=None):
        B, L, D = x.shape
        B_c, L_c, D_c = context.shape

        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, L_c, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Standard SDPA (Simplified for brevity)
        # Note: If context has a different mask than x, apply context mask here
        attn_mask = None
        if mask is not None:
            # # Masking the context (K,V) positions
            # attn_mask = mask.view(B, 1, 1, L_c)
            # 1. Create a 2D grid of valid interactions
            # (B, L, 1) * (B, 1, L) -> (B, L, L)
            mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
            
            # 2. Expand for Multi-Head Attention: (B, 1, L, L)
            attn_mask = mask_2d.unsqueeze(1).bool()

        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        x_attn = x_attn.transpose(1, 2).reshape(B, L, D)
        return self.proj(x_attn)


class FrequencyAttention(nn.Module):
    '''  
    Frequency Coordinated attention spectral volume learning implemented from original Google Paper (https://arxiv.org/pdf/2309.07906)

    ONLY USE with two-stage training as described in Google paper.
    '''
    def __init__(self, hidden_size, num_heads, num_freqs):
        super().__init__()
        self.num_heads = num_heads
        self.num_freqs = num_freqs
        
        # Standard Attention Components
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Frequency Positional Embedding (Crucial: K=0 is different from K=64)
        self.freq_pos_emb = nn.Parameter(torch.randn(1, num_freqs, hidden_size))
        
        # Norm specifically for this block
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Modulation for this block (Scale/Shift/Gate)
        # We use a separate modulation to keep it distinct from spatial attention
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
        # Zero init the gate
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t_emb):
        '''
        Input x: (B * K, L, D) - The flattened batch from the spatial layers
        '''
        total_batch, L, D = x.shape
        K = self.num_freqs
        B = total_batch // K
        
        # 1. Modulation
        shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
        
        # 2. Reshape for Frequency Attention
        # We want to attend across K, so we treat (B, L) as the "Batch" dimension for this layer
        # (B*K, L, D) -> (B, K, L, D) -> (B, L, K, D) -> (B*L, K, D)
        x_reshaped = x.view(B, K, L, D).permute(0, 2, 1, 3).reshape(B*L, K, D)
        
        # 3. Add Frequency Positional Embedding
        # freq_pos_emb is (1, K, D), broadcasts to (B*L, K, D)
        x_reshaped = x_reshaped + self.freq_pos_emb
        
        # 4. Normalize & Modulate
        # We need to expand shift/scale to match the new view (B*L)
        # shift is (B*K, D). We need to align it carefully.
        # Actually, simpler to normalize first, then attend, then add residual.
        x_norm = self.norm(x_reshaped)
        
        # 5. Attention
        qkv = self.qkv(x_norm).reshape(B*L, K, 3, self.num_heads, D // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard Attention over K
        x_attn = F.scaled_dot_product_attention(q, k, v)
        x_attn = x_attn.transpose(1, 2).reshape(B*L, K, D)
        x_attn = self.proj(x_attn)
        
        # 6. Reshape back to Spatial Format (B*K, L, D)
        # (B*L, K, D) -> (B, L, K, D) -> (B, K, L, D) -> (B*K, L, D)
        x_attn_out = x_attn.view(B, L, K, D).permute(0, 2, 1, 3).reshape(B*K, L, D)
        
        # 7. Apply Gate and Residual
        x = x + gate.unsqueeze(1) * x_attn_out
        
        return x
    

class SlowToFastCrossAttention(nn.Module):
    '''One-way cross-attention from slow-branch tokens to fast-branch tokens.

    Implements the bridge described in PLAN.md §11.9: Query from fast-branch
    residue tokens, Key/Value from a projection of the slow-branch output.
    Residue-indexed on both sides (L_fast == L_slow == L), so the attention
    is effectively learned per-residue mixing of the per-residue slow context
    into the fast stream.

    The output is multiplied by a per-block learnable scalar ``gate``,
    zero-initialised so the module is an identity at step 0 — safe to add
    into a pretrained fast-branch block without perturbing warm-start
    behaviour.

    Args:
      d_fast: Fast-branch token width (queries).
      slow_context_dim: Dimension of the slow-context tokens (keys/values).
          Typically ``K_slow * in_channels`` or a pre-projected width.
      num_heads: Multi-head attention heads; must divide ``d_fast``.
      dropout: Output dropout.

    Shape contract:
      Inputs:
        fast_tokens: (B, L, d_fast)
        slow_context: (B, L, slow_context_dim)
        mask: (B, L) or None
      Output:
        (B, L, d_fast), scaled by the (initially zero) gate.
    '''

    def __init__(
        self,
        d_fast: int,
        slow_context_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_fast % num_heads != 0:
            raise ValueError(
                f'd_fast={d_fast} must be divisible by num_heads={num_heads}'
            )
        self.num_heads = int(num_heads)
        self.head_dim = d_fast // self.num_heads

        self.norm_q = nn.LayerNorm(d_fast, elementwise_affine=False, eps=1e-6)
        self.kv_proj = nn.Linear(slow_context_dim, 2 * d_fast, bias=False)
        self.q_proj = nn.Linear(d_fast, d_fast, bias=False)
        self.out_proj = nn.Linear(d_fast, d_fast, bias=False)
        self.drop = nn.Dropout(dropout)

        # Zero-init gate — bridge is effectively absent at step 0, letting
        # the fast branch start from a warm-start checkpoint unchanged.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        fast_tokens: torch.Tensor,
        slow_context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = fast_tokens.shape

        q_in = self.norm_q(fast_tokens)
        q = self.q_proj(q_in).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(slow_context).view(B, L, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_mask = None
        if mask is not None:
            mb = mask.bool()
            attn_mask = (mb.unsqueeze(1) & mb.unsqueeze(2)).unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.drop(self.out_proj(out))
        return self.gate * out


class SpectralBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, use_cross_attn=False, use_freq_coords=False, num_freq_coords=64):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5
        self.use_cross_attn = use_cross_attn
        self.use_freq_coords = use_freq_coords
        self.enable_freq_attn = False

        # Attention
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio * 2 / 3) # SwiGLU ratio
        self.mlp = SwiGLU(hidden_size, mlp_hidden)
        
        # Norms
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # AdaLN-Zero Modulation
        # Input: t_emb (Global)
        # Output: 6 parameters (shift, scale, gate) x 2
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        # )

        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads, dropout)
            # Add a 7th parameter to adaLN for the cross-attn gate
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 9 * hidden_size, bias=True) # 6 -> 9 params
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

        # Zero Init (Crucial for DiT stability)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # OPTIONAL: Frequency Coordination Layer
        if self.use_freq_coords:
            self.freq_attn = FrequencyAttention(hidden_size, num_heads, num_freq_coords)

    def forward(self, x, t_emb, rope_freqs, context=None, mask=None):
        B, L, D = x.shape

        # Modulation Params
        mods = self.adaLN_modulation(t_emb).chunk(9 if self.use_cross_attn else 6, dim=1)
        
        # Self-Attention (standard DiT logic)
        shift_msa, scale_msa, gate_msa = mods[0:3]
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        #     self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Attention Block
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # QKV
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, D // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # RoPE
        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)
        
        attn_mask = None
        if mask is not None:
            # Mask input is (B, L) where True=Valid, False=Pad
            # We need to reshape to (B, 1, 1, L) to broadcast over Heads and Query Positions
            # SDPA expects: (B, Heads, Q_Len, K_Len)
            #attn_mask = mask.view(B, 1, 1, L)
            # (B, 1, L, L) ensures padding doesn't talk to protein AND vice-versa
            # Convert the float mask to boolean first
            mask_bool = mask.to(torch.bool)

            # Now apply the bitwise AND
            attn_mask = (mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)).unsqueeze(1)
            #attn_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1)
            
            # Ensure it is boolean (for SDPA, True=Keep, False=Ignore)
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.bool()

        # Self-Attention
        x_attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0 # Add dropout if needed
        )
        x_attn = x_attn.transpose(1, 2).flatten(2)
        x_attn = self.proj(x_attn)

        x_attn = self.attn_drop(x_attn)
        
        # Gate + Residual
        x = x + gate_msa.unsqueeze(1) * x_attn
        if mask is not None:
            x = x * mask.unsqueeze(-1) # <--- Hygiene Mask

        # 1. OPTIONAL: Frequency Coordination
        # This happens AFTER Spatial Attention but BEFORE the MLP
        if self.use_freq_coords and self.enable_freq_attn:
            # x is currently (B*K, L, D)
            # t_emb needs to be (B*K, D) - it already is, passed from main model
            x = self.freq_attn(x, t_emb)

        # 2. OPTIONAL Cross-Attention
        if self.use_cross_attn and context is not None:
            shift_ca, scale_ca, gate_ca = mods[3:6]
            # Use separate modulation for Cross-Attention
            x_norm_c = self.norm_cross(x) * (1 + scale_ca.unsqueeze(1)) + shift_ca.unsqueeze(1)
            x = x + gate_ca.unsqueeze(1) * self.cross_attn(x_norm_c, context, mask=mask)
            x = x * mask.unsqueeze(-1) # <--- Hygiene Mask
            
            # Re-index MLP mods
            shift_mlp, scale_mlp, gate_mlp = mods[6:9]
        else:
            shift_mlp, scale_mlp, gate_mlp = mods[3:6]

        # 3. MLP Block
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        if mask is not None:
            x = x * mask.unsqueeze(-1) # <--- Hygiene Mask

        return x


