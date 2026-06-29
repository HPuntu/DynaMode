
'''
Auxiliary transformer blocks used by SpecConv heads.
'''


from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamode.model.modules import SwiGLU, apply_rotary_pos_emb



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
        x = (B * K, L, D) - The flattened batch from the spatial layers
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

class AuxSpectralTransformerBlock(nn.Module):
    '''
    AdaLN transformer block for auxiliary SpecConv heads.

    This is intentionally separate from SpectralConvBlock in
    spectral_conv.py. It operates on ordinary residue tokens and does not
    perform the FNO-style convolution over frequency modes used by the main
    SpecConv trunk.
    '''

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

