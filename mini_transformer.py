"""
Mini-Transformer with VQ bottleneck for ZD-CBE v2
NanoGPT-style transformer with Vector Quantization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)  # [batch, seq, 3*embed]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous()  # [batch, seq, heads, head_dim]
        out = out.reshape(batch_size, seq_len, embed_dim)

        # Output projection
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and FFN"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual connection
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for concept emergence"""
    def __init__(self, num_embeddings: int = 32, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity for monitoring
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, {'loss': loss, 'perplexity': perplexity, 'encodings': encodings}


class MiniTransformerWithVQ(nn.Module):
    """
    Mini-Transformer with VQ bottleneck for ZD-CBE

    Architecture:
    - Input embedding
    - 2 Transformer blocks (4 heads each)
    - VQ bottleneck (optional)
    - Output projection
    """

    def __init__(
        self,
        vocab_size: int = 28,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_vq_codes: int = 32,
        use_vq: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 128
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.use_vq = use_vq

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # VQ bottleneck (optional)
        if use_vq:
            self.vq = VectorQuantizer(num_vq_codes, embed_dim)
        else:
            self.vq = None

        # Output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len = x.shape

        # Token + positional embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = token_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # VQ bottleneck (optional)
        vq_info = {}
        if self.use_vq and self.vq is not None:
            x, vq_info = self.vq(x)

        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.output_proj(x)

        return logits, vq_info

    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backwards compatibility wrapper
class CoreAgentWithVQ(MiniTransformerWithVQ):
    """Backwards compatible wrapper for existing code"""

    def __init__(
        self,
        vocab_size: int = 28,
        embed_dim: int = 64,
        hidden_dim: int = 128,  # Used for ff_dim
        num_embeddings: int = 32,  # num_vq_codes
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=4,  # Fixed at 4
            ff_dim=hidden_dim * 2,  # 2x hidden_dim
            num_vq_codes=num_embeddings,
            use_vq=True,
            dropout=dropout
        )

        self.hidden_state = None  # For compatibility

    def reset_hidden_state(self):
        """Compatibility method"""
        self.hidden_state = None

    def get_concept_usage(self):
        """Get VQ code usage statistics"""
        if self.vq is not None and hasattr(self.vq, 'embeddings'):
            return self.vq.embeddings.weight.detach().cpu().numpy()
        return None


# Test the model
if __name__ == "__main__":
    print("Testing Mini-Transformer with VQ...")

    # Create model
    model_with_vq = MiniTransformerWithVQ(
        vocab_size=28,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        num_vq_codes=32,
        use_vq=True
    )

    model_without_vq = MiniTransformerWithVQ(
        vocab_size=28,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        use_vq=False
    )

    print(f"Model with VQ: {model_with_vq.count_parameters():,} parameters")
    print(f"Model without VQ: {model_without_vq.count_parameters():,} parameters")

    # Test forward pass
    x = torch.randint(0, 28, (2, 10))

    logits_vq, info_vq = model_with_vq(x)
    print(f"With VQ - Logits shape: {logits_vq.shape}, VQ loss: {info_vq.get('loss', 'N/A')}")

    logits_no_vq, info_no_vq = model_without_vq(x)
    print(f"Without VQ - Logits shape: {logits_no_vq.shape}, VQ info: {info_no_vq}")

    print("\nBackwards compatibility test:")
    old_style_model = CoreAgentWithVQ(
        vocab_size=28,
        embed_dim=64,
        hidden_dim=128,
        num_embeddings=32,
        num_layers=2
    )
    print(f"Old-style model: {old_style_model.count_parameters():,} parameters")
    logits_old, info_old = old_style_model(x)
    print(f"Old-style - Logits shape: {logits_old.shape}")

    print("\n✅ All tests passed!")
