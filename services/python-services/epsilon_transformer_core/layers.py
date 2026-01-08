"""
Transformer layers - PyTorch implementation with autograd
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))


class RoPE(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute frequencies
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        # x: (batch, n_heads, seq_len, head_dim)
        if seq_len is None:
            seq_len = x.shape[-2]
        
        head_dim = x.shape[-1]
        cos = self.cos_cached[:, :, :seq_len, :head_dim]
        sin = self.sin_cached[:, :, :seq_len, :head_dim]
        
        # Handle case where head_dim doesn't match cached dimensions
        if head_dim != self.dim:
            # Recompute RoPE for this dimension if needed
            if head_dim % 2 != 0:
                # If odd dimension, pad to even
                x = F.pad(x, (0, 1), mode='constant', value=0)
                head_dim = head_dim + 1
            
            # Recompute frequencies for this dimension
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype).float() / head_dim))
            t = torch.arange(seq_len, device=x.device, dtype=x.dtype).float()
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
        
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE"""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.use_rope = config.use_rope
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if self.use_rope:
            self.rope = RoPE(self.head_dim, config.max_seq_len)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        # Output projection
        output = self.o_proj(attn_output)
        return output


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Activation function
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "swiglu":
            # SwiGLU: Swish(xW) * (xV)
            self.gate_proj = nn.Linear(config.d_model, config.d_ff)
            self.up_proj = nn.Linear(config.d_model, config.d_ff)
            self.down_proj = nn.Linear(config.d_ff, config.d_model)
            self.activation = None  # SwiGLU is custom
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'gate_proj'):  # SwiGLU
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(gate * up)
        else:
            return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

