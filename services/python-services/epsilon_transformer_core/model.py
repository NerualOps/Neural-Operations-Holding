"""
Epsilon Transformer Language Model - PyTorch Implementation
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import TransformerConfig
from .layers import TransformerBlock, RMSNorm


class EpsilonTransformerLM(nn.Module):
    """Epsilon AI Transformer Language Model"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights (optional - share embedding and output weights)
        # self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs for loss computation
            mask: (batch, seq_len) attention mask
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss if targets provided, else None
        """
        # Embeddings
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.9,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> torch.Tensor:
        """
        Generate text using the model
        
        Args:
            input_ids: (batch, seq_len) starting tokens
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            top_k: top-k sampling
            repetition_penalty: penalty for repeating tokens (1.0 = no penalty, >1.0 = penalty)
        
        Returns:
            generated_ids: (batch, seq_len + max_new_tokens) generated token IDs
        """
        self.eval()
        generated = input_ids.clone()
        
        # Track recent tokens for repetition penalty
        recent_tokens = generated[0].tolist()[-10:] if generated.shape[0] > 0 else []
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = self.forward(generated)
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(recent_tokens) > 0:
                for token_id in set(recent_tokens[-5:]):  # Penalize last 5 unique tokens
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Avoid sampling padding/invalid tokens
            if probs.numel() > 0 and not torch.isnan(probs).any():
                # Ensure we have valid probabilities
                if probs.sum() > 0:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # All probabilities are zero/inf - use argmax as fallback
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Fallback if all tokens filtered out or NaN
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update recent tokens (keep last 10)
            recent_tokens.append(next_token[0, 0].item())
            if len(recent_tokens) > 10:
                recent_tokens.pop(0)
        
        return generated

