import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class VectorQuantizer(nn.Module):
    """Vector Quantization layer for concept emergence"""
    def __init__(self, num_embeddings: int = 64, embedding_dim: int = 128, commitment_cost: float = 0.25):
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

class CoreAgentWithVQ(nn.Module):
    """Core cognitive agent with Vector Quantization bottleneck"""
    def __init__(self, 
                 vocab_size: int = 28,  # A-Z + 0-1
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_embeddings: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Recurrent core (GRU)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Vector Quantization bottleneck
        self.vq = VectorQuantizer(num_embeddings, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_state = None
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass through the network"""
        batch_size, seq_len = x.shape
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # GRU forward pass
        gru_output, hidden = self.gru(embedded, hidden)
        
        # Vector Quantization
        quantized, vq_info = self.vq(gru_output)
        
        # Output projection
        logits = self.output_proj(quantized)
        
        return logits, hidden, vq_info
    
    def generate_next_token(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate next token given current token"""
        logits, hidden, vq_info = self.forward(x, self.hidden_state)
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample next token
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        self.hidden_state = hidden
        
        return next_token, vq_info
    
    def reset_hidden_state(self):
        """Reset the hidden state"""
        self.hidden_state = None
    
    def get_concept_usage(self) -> np.ndarray:
        """Get VQ code usage statistics"""
        if hasattr(self.vq, 'embeddings'):
            # Return embedding weights as concept representation
            return self.vq.embeddings.weight.detach().cpu().numpy()
        return np.array([])
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DreamEnvironment:
    """Environment for the dreaming phase"""
    def __init__(self, model: CoreAgentWithVQ, vocab_size: int = 28):
        self.model = model
        self.vocab_size = vocab_size
        self.reset()
        
    def reset(self):
        """Reset the dream environment"""
        self.model.reset_hidden_state()
        self.sequence = []
        
    def step(self, temperature: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """Take one step in the dream environment"""
        # Start with random token if sequence is empty
        if len(self.sequence) == 0:
            current_token = torch.randint(0, self.vocab_size, (1, 1))
        else:
            current_token = torch.tensor([[self.sequence[-1]]], dtype=torch.long)
            
        # Generate next token
        next_token, vq_info = self.model.generate_next_token(current_token, temperature)
        
        # Add to sequence
        self.sequence.append(next_token.item())
        
        return next_token, vq_info
    
    def get_sequence(self, max_length: int = 100) -> str:
        """Get recent sequence as string"""
        recent_seq = self.sequence[-max_length:]
        return self.tokens_to_string(recent_seq)
    
    def tokens_to_string(self, tokens: list) -> str:
        """Convert token indices to string"""
        chars = []
        for token in tokens:
            if token < 26:
                chars.append(chr(ord('A') + token))
            elif token == 26:
                chars.append('0')
            elif token == 27:
                chars.append('1')
        return ''.join(chars)
    
    def string_to_tokens(self, s: str) -> list:
        """Convert string to token indices"""
        tokens = []
        for c in s.upper():
            if c.isalpha():
                tokens.append(ord(c) - ord('A'))
            elif c == '0':
                tokens.append(26)
            elif c == '1':
                tokens.append(27)
        return tokens