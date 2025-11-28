"""
Mini-GPT: GPT-2 Style Language Model for DFCB Validation
Decoder-only Transformer optimized for GPU training
Target: 10-25M parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)

        # Causal mask (prevent attending to future tokens)
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.gelu(x)  # GELU activation (GPT-2 style)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer decoder block"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor):
        # Pre-norm architecture (more stable)
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    Mini-GPT: Decoder-only Transformer for language modeling

    Architecture:
    - Token + positional embeddings
    - N Transformer blocks
    - Layer norm
    - Output projection (language modeling head)

    Parameters:
    - vocab_size: 50257 (GPT-2 BPE)
    - embed_dim: 256-512
    - num_layers: 6-12
    - num_heads: 8
    - ff_dim: 4 * embed_dim
    - context_length: 256-512
    - dropout: 0.1

    Total params: ~10-25M
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = 1536,
        context_length: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings (learned)
        self.position_embedding = nn.Embedding(context_length, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying (share token embedding and output projection)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        self.num_params = self.count_parameters()

    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass

        Args:
            idx: Token indices (B, T)
            targets: Target tokens for loss computation (B, T)

        Returns:
            logits: (B, T, vocab_size)
            loss: (optional) cross-entropy loss
        """
        B, T = idx.shape
        assert T <= self.context_length, f"Sequence length {T} exceeds context length {self.context_length}"

        # Token embeddings
        token_emb = self.token_embedding(idx)  # (B, T, C)

        # Positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T, C)

        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1  # Ignore padding
            )

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ):
        """
        Generate new tokens autoregressively

        Args:
            idx: Initial context (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)

        Returns:
            Generated sequence (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]

            # Forward pass
            logits, _ = self.forward(idx_cond)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params(self):
        """Get number of parameters in millions"""
        return self.num_params / 1e6


def create_mini_gpt_10M():
    """Create a ~10M parameter Mini-GPT"""
    return MiniGPT(
        vocab_size=50257,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        context_length=256,
        dropout=0.1
    )


def create_mini_gpt_25M():
    """Create a ~25M parameter Mini-GPT"""
    return MiniGPT(
        vocab_size=50257,
        embed_dim=384,
        num_layers=8,
        num_heads=8,
        ff_dim=1536,
        context_length=256,
        dropout=0.1
    )


# Test
if __name__ == "__main__":
    # Create model
    model = create_mini_gpt_10M()
    print(f"Model parameters: {model.get_num_params():.2f}M")

    # Test forward pass
    batch_size = 4
    seq_len = 128
    idx = torch.randint(0, 50257, (batch_size, seq_len))
    targets = torch.randint(0, 50257, (batch_size, seq_len))

    logits, loss = model(idx, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    context = torch.randint(0, 50257, (1, 10))
    generated = model.generate(context, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")

    print("\n✅ Mini-GPT architecture test passed!")
