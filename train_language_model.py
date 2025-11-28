"""
Language Modeling Training with Perplexity Tracking
Compare Random vs Dreamer initialization
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import math

from mini_gpt import MiniGPT, create_mini_gpt_10M, create_mini_gpt_25M
from tinystories_loader import create_tinystories_dataloader


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(min(loss, 20))  # Cap to prevent overflow


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str, max_batches: int = 50):
    """Evaluate model and calculate perplexity"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            if num_batches >= max_batches:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits, loss = model(input_ids, labels)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)

    return avg_loss, perplexity


def train_language_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_tokens: int = 100000,
    eval_interval: int = 1000,
    learning_rate: float = 3e-4,
    device: str = 'cuda',
    model_name: str = 'model',
    save_dir: str = 'checkpoints'
):
    """
    Train language model and track perplexity convergence

    Args:
        model: Mini-GPT model
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        max_tokens: Maximum training tokens
        eval_interval: Evaluate every N tokens
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
        model_name: Model identifier (e.g., 'random', 'dreamer')
        save_dir: Directory to save checkpoints

    Returns:
        dict: Training metrics with perplexity convergence
    """
    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Tracking
    tokens_seen = 0
    training_curve = []  # List of (tokens_seen, train_loss, val_loss, perplexity)
    dataloader_iter = iter(train_dataloader)

    print(f"\n📚 Training {model_name} on TinyStories")
    print(f"Max tokens: {max_tokens:,}")
    print(f"Eval interval: {eval_interval:,} tokens")
    print(f"Model parameters: {model.get_num_params():.2f}M")

    # Initial evaluation
    val_loss, val_perplexity = evaluate_model(model, val_dataloader, device)
    training_curve.append({
        'tokens': 0,
        'train_loss': None,
        'val_loss': val_loss,
        'perplexity': val_perplexity
    })
    print(f"Initial perplexity: {val_perplexity:.2f}")

    pbar = tqdm(total=max_tokens, desc=f"Training {model_name}")

    while tokens_seen < max_tokens:
        try:
            input_ids, labels = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            input_ids, labels = next(dataloader_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        batch_tokens = input_ids.numel()

        # Forward pass
        logits, loss = model(input_ids, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tokens_seen += batch_tokens

        # Evaluate periodically
        if tokens_seen % eval_interval < batch_tokens or tokens_seen >= max_tokens:
            model.eval()
            val_loss, val_perplexity = evaluate_model(model, val_dataloader, device)
            model.train()

            training_curve.append({
                'tokens': tokens_seen,
                'train_loss': loss.item(),
                'val_loss': val_loss,
                'perplexity': val_perplexity
            })

            pbar.set_postfix({
                'ppl': f'{val_perplexity:.2f}',
                'loss': f'{val_loss:.4f}'
            })

        pbar.update(batch_tokens)

    pbar.close()

    # Final evaluation
    model.eval()
    final_loss, final_perplexity = evaluate_model(model, val_dataloader, device)

    print(f"\n✅ Training complete for {model_name}")
    print(f"Final perplexity: {final_perplexity:.2f}")
    print(f"Tokens processed: {tokens_seen:,}")

    # Save checkpoint
    save_path = Path(save_dir) / f'{model_name}_final.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokens_seen': tokens_seen,
        'final_perplexity': final_perplexity,
        'training_curve': training_curve
    }, save_path)

    print(f"💾 Model saved to {save_path}")

    return {
        'model_name': model_name,
        'tokens_seen': tokens_seen,
        'final_perplexity': final_perplexity,
        'training_curve': training_curve
    }


def find_convergence_tokens(training_curve: list, target_perplexity: float = 50.0):
    """Find how many tokens needed to reach target perplexity"""
    for point in training_curve:
        if point['perplexity'] is not None and point['perplexity'] <= target_perplexity:
            return point['tokens']
    return None  # Did not converge


def main():
    parser = argparse.ArgumentParser(description='Language Model Training')
    parser.add_argument('--model_size', type=str, default='10M', choices=['10M', '25M'])
    parser.add_argument('--dreamer_checkpoint', type=str, default=None,
                       help='Path to dreamer checkpoint (for dreamer model)')
    parser.add_argument('--max_tokens', type=int, default=100000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='model')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create model
    print(f"Creating {args.model_size} Mini-GPT...")
    if args.model_size == '10M':
        model = create_mini_gpt_10M()
    else:
        model = create_mini_gpt_25M()

    # Load dreamer checkpoint if provided
    if args.dreamer_checkpoint:
        print(f"Loading dreamer checkpoint from {args.dreamer_checkpoint}")
        checkpoint = torch.load(args.dreamer_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Dreamer weights loaded!")

    # Create dataloaders
    print("Creating TinyStories dataloaders...")
    train_dataloader, tokenizer = create_tinystories_dataloader(
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Use same dataset for validation (small corpus)
    val_dataloader = train_dataloader

    # Train
    metrics = train_language_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        max_tokens=args.max_tokens,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        device=args.device,
        model_name=args.model_name,
        save_dir=args.save_dir
    )

    # Save metrics
    metrics_path = Path(args.save_dir) / f'{args.model_name}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Metrics saved to {metrics_path}")

    # Check convergence
    convergence_50 = find_convergence_tokens(metrics['training_curve'], 50.0)
    if convergence_50:
        print(f"✅ Reached 50 perplexity at {convergence_50:,} tokens")
    else:
        print(f"⚠️ Did not reach 50 perplexity within {args.max_tokens:,} tokens")


if __name__ == "__main__":
    main()
