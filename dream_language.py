"""
Dream Phase for Language Models
Self-supervised pretraining on random token sequences
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from mini_gpt import MiniGPT, create_mini_gpt_10M, create_mini_gpt_25M
from tinystories_loader import create_random_dataloader


def dream_phase(
    model: nn.Module,
    dataloader: DataLoader,
    num_steps: int = 5000,
    learning_rate: float = 3e-4,
    device: str = 'cuda',
    save_path: str = 'dreamer_gpt.pth',
    log_interval: int = 100
):
    """
    Dream phase: Self-supervised pretraining on random tokens

    Args:
        model: Mini-GPT model
        dataloader: Random token dataloader
        num_steps: Number of training steps
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
        save_path: Path to save model checkpoint
        log_interval: Logging interval

    Returns:
        dict: Training metrics
    """
    model = model.to(device)
    model.train()

    # Optimizer (AdamW like GPT-2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Learning rate scheduler (cosine decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=learning_rate * 0.1
    )

    # Training loop
    losses = []
    step = 0
    dataloader_iter = iter(dataloader)

    print(f"🌙 Starting Dream Phase: {num_steps} steps on random tokens")
    print(f"Device: {device}")
    print(f"Model parameters: {model.get_num_params():.2f}M")

    pbar = tqdm(total=num_steps, desc="Dreaming")

    while step < num_steps:
        try:
            input_ids, labels = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            input_ids, labels = next(dataloader_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass
        logits, loss = model(input_ids, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Log metrics
        losses.append(loss.item())

        if step % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:])
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        pbar.update(1)
        step += 1

    pbar.close()

    # Save model
    print(f"\n💾 Saving dreamer model to {save_path}")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'losses': losses,
        'config': {
            'vocab_size': model.vocab_size,
            'embed_dim': model.embed_dim,
            'context_length': model.context_length,
            'num_params': model.get_num_params()
        }
    }
    torch.save(checkpoint, save_path)

    # Metrics
    final_loss = np.mean(losses[-100:])
    initial_loss = np.mean(losses[:100])
    improvement = ((initial_loss - final_loss) / initial_loss) * 100

    metrics = {
        'num_steps': step,
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'improvement_pct': float(improvement),
        'all_losses': losses
    }

    print(f"\n✅ Dream Phase Complete!")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Dream Phase for Language Models')
    parser.add_argument('--model_size', type=str, default='10M', choices=['10M', '25M'])
    parser.add_argument('--num_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='dreamer_gpt.pth')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)

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

    print(f"Model parameters: {model.get_num_params():.2f}M")

    # Create random dataloader
    print(f"Creating random token dataloader...")
    dataloader = create_random_dataloader(
        vocab_size=model.vocab_size,
        batch_size=args.batch_size,
        num_samples=args.num_steps * 2,  # Enough samples
        max_length=args.max_length
    )

    # Dream phase
    metrics = dream_phase(
        model=model,
        dataloader=dataloader,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        save_path=args.save_path,
        log_interval=args.log_interval
    )

    # Save metrics
    metrics_path = Path(args.save_path).with_suffix('.json')
    with open(metrics_path, 'w') as f:
        # Don't save all losses (too large)
        metrics_summary = {
            'num_steps': metrics['num_steps'],
            'initial_loss': metrics['initial_loss'],
            'final_loss': metrics['final_loss'],
            'improvement_pct': metrics['improvement_pct']
        }
        json.dump(metrics_summary, f, indent=2)

    print(f"\n💾 Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
