import torch
import torch.nn as nn
import time
import argparse
import json
import os
from typing import Dict

from core_agent import CoreAgentWithVQ
from mini_lm_loader import MiniLanguageModelDataLoader, calculate_perplexity


class MiniLMTrainer:
    """Train a mini language model using a dreamer-initialized model"""

    def __init__(self, model: CoreAgentWithVQ, config: Dict):
        self.model = model
        self.config = config
        self.data_loader = MiniLanguageModelDataLoader(
            vocab_size=config['vocab_size'],
            max_seq_length=config.get('max_seq_length', 50)
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lm_learning_rate', 0.001)
        )

        self.training_history = []

    def train_epoch(self, batch_size: int = 32) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()

        # Generate batch
        inputs, targets = self.data_loader.generate_batch(batch_size)

        # Forward pass
        predictions, _, vq_info = self.model(inputs)

        # Calculate loss
        loss = nn.CrossEntropyLoss(ignore_index=0)(
            predictions.view(-1, self.config['vocab_size']),
            targets.view(-1)
        )

        # Add VQ loss if available
        if 'loss' in vq_info:
            total_loss = loss + 0.1 * vq_info['loss']  # Weighted VQ loss
        else:
            total_loss = loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Calculate metrics
        self.model.eval()
        with torch.no_grad():
            metrics = self.data_loader.evaluate_batch(self.model, batch_size)
            metrics['train_loss'] = loss.item()

        return metrics

    def train(self, num_epochs: int = 100, batch_size: int = 32,
              eval_interval: int = 10) -> Dict:
        """Train the mini language model"""
        print(f"Training Mini Language Model for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")

        start_time = time.time()
        best_loss = float('inf')

        for epoch in range(num_epochs):
            # Train
            metrics = self.train_epoch(batch_size)
            self.training_history.append(metrics)

            # Periodic evaluation and logging
            if epoch % eval_interval == 0 or epoch == num_epochs - 1:
                perplexity = calculate_perplexity(
                    self.model, self.data_loader, num_batches=5, batch_size=batch_size
                )

                print(f"Epoch {epoch}/{num_epochs}")
                print(f"  Loss: {metrics['train_loss']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Perplexity: {perplexity:.2f}")

                # Generate sample
                sample = self.data_loader.generate_from_model(
                    self.model, start_text="A", max_length=30, temperature=0.8
                )
                print(f"  Sample: {sample}")
                print("-" * 50)

                # Save best model
                if metrics['train_loss'] < best_loss:
                    best_loss = metrics['train_loss']
                    self.save_model('best_mini_lm.pth')

        training_time = time.time() - start_time

        # Final evaluation
        final_perplexity = calculate_perplexity(
            self.model, self.data_loader, num_batches=20, batch_size=batch_size
        )

        results = {
            'num_epochs': num_epochs,
            'training_time': training_time,
            'final_loss': self.training_history[-1]['train_loss'],
            'final_accuracy': self.training_history[-1]['accuracy'],
            'final_perplexity': final_perplexity,
            'best_loss': best_loss,
            'training_history': self.training_history
        }

        # Save final model
        self.save_model('final_mini_lm.pth')

        # Save results
        with open('mini_lm_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nMini LM Training completed in {training_time:.2f} seconds")
        print(f"Final perplexity: {final_perplexity:.2f}")
        print(f"Final accuracy: {results['final_accuracy']:.4f}")

        return results

    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filename)

    def generate_samples(self, num_samples: int = 10, max_length: int = 30):
        """Generate multiple samples from the model"""
        print("\nGenerated samples:")
        for i in range(num_samples):
            start_char = self.data_loader.idx_to_char[i % 26]
            sample = self.data_loader.generate_from_model(
                self.model, start_text=start_char, max_length=max_length, temperature=0.8
            )
            print(f"{i+1}. {sample}")


def compare_dreamer_vs_random(config: Dict):
    """Compare dreamer-initialized vs random-initialized mini LM training"""
    print("=" * 60)
    print("MINI LANGUAGE MODEL COMPARISON")
    print("Dreamer-Initialized vs Random-Initialized")
    print("=" * 60)

    # Load dreamer model
    print("\n1. Training with DREAMER initialization...")
    dreamer_model = CoreAgentWithVQ(
        vocab_size=config['vocab_size'],
        embed_dim=config.get('embed_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_embeddings=config.get('num_vq_codes', 32),
        num_layers=config.get('num_layers', 1)
    )

    # Load dreamer weights if available
    if os.path.exists('dreamer_model.pth'):
        checkpoint = torch.load('dreamer_model.pth', map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            dreamer_model.load_state_dict(checkpoint['model_state'])
        else:
            dreamer_model.load_state_dict(checkpoint)
        print("Loaded dreamer weights from dreamer_model.pth")
    else:
        print("Warning: dreamer_model.pth not found, using random initialization")

    dreamer_trainer = MiniLMTrainer(dreamer_model, config)
    dreamer_results = dreamer_trainer.train(
        num_epochs=config.get('lm_epochs', 100),
        batch_size=config.get('batch_size', 32),
        eval_interval=10
    )

    # Generate samples from dreamer model
    print("\nDreamer model samples:")
    dreamer_trainer.generate_samples(num_samples=5, max_length=30)

    # Train random model
    print("\n2. Training with RANDOM initialization...")
    random_model = CoreAgentWithVQ(
        vocab_size=config['vocab_size'],
        embed_dim=config.get('embed_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_embeddings=config.get('num_vq_codes', 32),
        num_layers=config.get('num_layers', 1)
    )

    random_trainer = MiniLMTrainer(random_model, config)
    random_results = random_trainer.train(
        num_epochs=config.get('lm_epochs', 100),
        batch_size=config.get('batch_size', 32),
        eval_interval=10
    )

    # Generate samples from random model
    print("\nRandom model samples:")
    random_trainer.generate_samples(num_samples=5, max_length=30)

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    comparison = {
        'dreamer': {
            'final_loss': dreamer_results['final_loss'],
            'final_accuracy': dreamer_results['final_accuracy'],
            'final_perplexity': dreamer_results['final_perplexity'],
            'training_time': dreamer_results['training_time']
        },
        'random': {
            'final_loss': random_results['final_loss'],
            'final_accuracy': random_results['final_accuracy'],
            'final_perplexity': random_results['final_perplexity'],
            'training_time': random_results['training_time']
        },
        'advantage': {
            'loss_improvement': random_results['final_loss'] - dreamer_results['final_loss'],
            'accuracy_improvement': dreamer_results['final_accuracy'] - random_results['final_accuracy'],
            'perplexity_improvement': random_results['final_perplexity'] - dreamer_results['final_perplexity']
        }
    }

    print(f"\nDreamer Model:")
    print(f"  Final Loss: {comparison['dreamer']['final_loss']:.4f}")
    print(f"  Final Accuracy: {comparison['dreamer']['final_accuracy']:.4f}")
    print(f"  Final Perplexity: {comparison['dreamer']['final_perplexity']:.2f}")

    print(f"\nRandom Model:")
    print(f"  Final Loss: {comparison['random']['final_loss']:.4f}")
    print(f"  Final Accuracy: {comparison['random']['final_accuracy']:.4f}")
    print(f"  Final Perplexity: {comparison['random']['final_perplexity']:.2f}")

    print(f"\nDreamer Advantage:")
    print(f"  Loss Improvement: {comparison['advantage']['loss_improvement']:.4f}")
    print(f"  Accuracy Improvement: {comparison['advantage']['accuracy_improvement']:.4f}")
    print(f"  Perplexity Improvement: {comparison['advantage']['perplexity_improvement']:.2f}")

    # Save comparison
    with open('mini_lm_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Train Mini Language Model')
    parser.add_argument('--config', type=str, default='config_fast.json',
                       help='Configuration file path')
    parser.add_argument('--lm_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--compare', action='store_true',
                       help='Compare dreamer vs random initialization')

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'vocab_size': 28,
            'embed_dim': 64,
            'hidden_dim': 128,
            'num_vq_codes': 32,
            'num_layers': 1,
            'lm_learning_rate': 0.003,
            'batch_size': 32,
            'max_seq_length': 50
        }

    config['lm_epochs'] = args.lm_epochs

    if args.compare:
        compare_dreamer_vs_random(config)
    else:
        # Train only dreamer model
        model = CoreAgentWithVQ(
            vocab_size=config['vocab_size'],
            embed_dim=config.get('embed_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            num_embeddings=config.get('num_vq_codes', 32),
            num_layers=config.get('num_layers', 1)
        )

        # Load dreamer weights if available
        if os.path.exists('dreamer_model.pth'):
            checkpoint = torch.load('dreamer_model.pth', map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
            print("Loaded dreamer weights from dreamer_model.pth")

        trainer = MiniLMTrainer(model, config)
        results = trainer.train(
            num_epochs=config['lm_epochs'],
            batch_size=config['batch_size']
        )

        # Generate samples
        trainer.generate_samples(num_samples=10, max_length=40)


if __name__ == "__main__":
    main()
