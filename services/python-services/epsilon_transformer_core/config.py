"""
Transformer configuration
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Transformer model configuration"""
    vocab_size: int = 50000
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 510
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    use_rope: bool = True  # Rotary Position Embedding
    use_alibi: bool = False  # ALiBi positional bias
    attention_type: str = "standard"  # standard, grouped_query, sliding_window
    activation: str = "gelu"  # gelu, swiglu, relu
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'vocab_size': self.vocab_size,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'use_rope': self.use_rope,
            'use_alibi': self.use_alibi,
            'attention_type': self.attention_type,
            'activation': self.activation,
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(**data)

