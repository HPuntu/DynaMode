'''
Collection of shared modules used across the models e.g. for position embedding and
other conditioning features.
'''

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
    '''
    Joint MLP embedding of ``(win_pos, temperature, protein_size)``.

    The scalar sinusoidal window-position embedding cannot express
    interactions between window position and thermodynamic / structural
    context. This module projects a concatenation of the three raw scalars
    through a 2-layer MLP, letting the model learn joint responses such as
    "terminal windows of large proteins at high T" without a separate
    interaction term.
    '''

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

