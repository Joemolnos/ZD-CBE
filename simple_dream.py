"""
Simple Dream Phase without Evolutionary Agent
For v2 experiments with Transformer architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from mini_transformer import MiniTransformerWithVQ


def simple_dream_phase(
    model: MiniTransformerWithVQ,
    vocab_size: int = 28,
    dream_steps: int = 1000,
    batch_size: int = 32,
    seq_len: int = 20,
    learning_rate: float = 0.003,
    save_path: str = "dreamer_model.pth"
):
    """
    Simple dreaming phase: auto-regressive training on random noise
    No evolutionary agent, just pure self-supervised learning
    """
    print(f"Starting Simple Dream Phase for {dream_steps} steps...")
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    losses = []
    perplexities = []

    for step in range(dream_steps):
        # Generate random sequences
        random_sequences = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create input and target (shift by 1)
        inputs = random_sequences[:, :-1]
        targets = random_sequences[:, 1:]

        # Forward pass
        logits, vq_info = model(inputs)

        # Calculate prediction loss
        pred_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        # Add VQ loss if available
        if 'loss' in vq_info:
            total_loss = pred_loss + 0.1 * vq_info['loss']
        else:
            total_loss = pred_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        losses.append(pred_loss.item())
        if 'perplexity' in vq_info:
            perplexity_val = vq_info['perplexity']
            if torch.is_tensor(perplexity_val):
                perplexity_val = perplexity_val.item()
            perplexities.append(perplexity_val)

        # Print progress
        if step % 100 == 0 or step == dream_steps - 1:
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            avg_perp = np.mean(perplexities[-100:]) if perplexities else 0.0
            print(f"Step {step}/{dream_steps} - Loss: {avg_loss:.4f} - Perplexity: {avg_perp:.4f}")

    # Save model
    checkpoint = {
        'model_state': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'embed_dim': model.embed_dim,
            'num_layers': len(model.blocks),
            'num_heads': model.blocks[0].attention.num_heads if len(model.blocks) > 0 else 4,
            'use_vq': model.use_vq
        },
        'dream_steps': dream_steps,
        'final_loss': losses[-1] if losses else 0.0,
        'final_perplexity': perplexities[-1] if perplexities else 0.0
    }

    torch.save(checkpoint, save_path)
    print(f"\nDream phase completed!")
    print(f"Model saved to: {save_path}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Final perplexity: {perplexities[-1]:.4f}" if perplexities else "N/A")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description='Simple Dream Phase')
    parser.add_argument('--vocab_size', type=int, default=28)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=256)
    parser.add_argument('--num_vq_codes', type=int, default=32)
    parser.add_argument('--dream_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--save_path', type=str, default='dreamer_model.pth')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create model
    model = MiniTransformerWithVQ(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_vq_codes=args.num_vq_codes,
        use_vq=True
    )

    # Run dream phase
    simple_dream_phase(
        model=model,
        vocab_size=args.vocab_size,
        dream_steps=args.dream_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
