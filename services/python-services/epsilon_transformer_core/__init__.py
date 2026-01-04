"""
Epsilon AI Transformer - PyTorch Implementation
Custom transformer architecture with PyTorch autograd
"""

from .model import EpsilonTransformerLM
from .config import TransformerConfig

__all__ = ['EpsilonTransformerLM', 'TransformerConfig']

