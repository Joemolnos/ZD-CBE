import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import random

class MiniLanguageModelDataLoader:
    """Data loader for training a mini language model after dreaming phase"""

    def __init__(self, vocab_size: int = 28, max_seq_length: int = 50):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # Character mapping (A-Z + 0-1)
        self.char_to_idx = {}
        for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            self.char_to_idx[char] = i
        self.char_to_idx['0'] = 26
        self.char_to_idx['1'] = 27
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # Simple vocabulary of words using A-Z characters
        self.words = self._create_simple_vocabulary()
        self.templates = self._create_sentence_templates()

    def _create_simple_vocabulary(self) -> List[str]:
        """Create a simple vocabulary using A-Z characters"""
        # Simple 2-3 letter "words"
        words = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'AB', 'AC', 'AD', 'BA', 'BC', 'CA', 'CB', 'DA', 'DB',
            'ABC', 'ABD', 'BAC', 'BCA', 'CAB', 'CBA', 'DAB', 'DBA',
            'AAA', 'BBB', 'CCC', 'ABA', 'BAB', 'CAC', 'DAD',
        ]
        return words

    def _create_sentence_templates(self) -> List[str]:
        """Create simple sentence-like patterns"""
        # Patterns that create simple sequences
        templates = [
            lambda: f"{random.choice(self.words)}{random.choice(self.words)}",
            lambda: f"{random.choice(self.words)}{random.choice(['0', '1'])}{random.choice(self.words)}",
            lambda: f"{random.choice(self.words)}{random.choice(self.words)}{random.choice(self.words)}",
            lambda: f"{random.choice(self.words[:10])}" * 3,  # Repetition
            lambda: self._generate_pattern(),
        ]
        return templates

    def _generate_pattern(self) -> str:
        """Generate a simple repeating pattern"""
        pattern = random.choice(self.words[:15])
        repetitions = random.randint(2, 4)
        return pattern * repetitions

    def generate_sequence(self) -> str:
        """Generate a single training sequence"""
        template = random.choice(self.templates)
        sequence = template()

        # Truncate if too long
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]

        return sequence

    def generate_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of training sequences"""
        sequences = []

        for _ in range(batch_size):
            seq = self.generate_sequence()
            sequences.append(seq)

        # Convert to tensors
        input_tensors = []
        target_tensors = []

        max_len = min(max(len(seq) for seq in sequences), self.max_seq_length)

        for seq in sequences:
            # Convert to indices
            indices = [self.char_to_idx.get(char, 0) for char in seq]

            # Pad if necessary
            if len(indices) < max_len:
                indices = indices + [0] * (max_len - len(indices))
            else:
                indices = indices[:max_len]

            # Input: all but last
            # Target: all but first (shifted by 1)
            input_indices = indices[:-1] if len(indices) > 1 else [0]
            target_indices = indices[1:] if len(indices) > 1 else [0]

            # Ensure same length
            if len(input_indices) < max_len - 1:
                input_indices = input_indices + [0] * (max_len - 1 - len(input_indices))
                target_indices = target_indices + [0] * (max_len - 1 - len(target_indices))

            input_tensors.append(torch.tensor(input_indices[:max_len-1], dtype=torch.long))
            target_tensors.append(torch.tensor(target_indices[:max_len-1], dtype=torch.long))

        return torch.stack(input_tensors), torch.stack(target_tensors)

    def decode_sequence(self, indices: torch.Tensor) -> str:
        """Decode indices back to string"""
        if indices.dim() > 1:
            indices = indices[0]  # Take first in batch

        chars = []
        for idx in indices:
            char = self.idx_to_char.get(idx.item(), 'A')
            chars.append(char)

        return ''.join(chars).rstrip('\x00')

    def evaluate_batch(self, model, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate model on a batch"""
        model.eval()

        inputs, targets = self.generate_batch(batch_size)

        with torch.no_grad():
            predictions, _, _ = model(inputs)

            # Calculate loss
            loss = F.cross_entropy(
                predictions.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=0
            )

            # Calculate accuracy
            pred_indices = torch.argmax(predictions, dim=-1)
            correct = (pred_indices == targets).float()

            # Ignore padding (0)
            mask = (targets != 0).float()
            accuracy = (correct * mask).sum() / (mask.sum() + 1e-10)

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }

    def generate_from_model(self, model, start_text: str = "A",
                          max_length: int = 30, temperature: float = 1.0) -> str:
        """Generate text from the model"""
        model.eval()
        model.reset_hidden_state()

        # Convert start text to indices
        current_seq = [self.char_to_idx.get(char, 0) for char in start_text]
        generated = start_text

        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                x = torch.tensor([current_seq[-min(len(current_seq), 20):]], dtype=torch.long)

                # Forward pass
                logits, _, _ = model(x)

                # Get last prediction
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=0)

                # Sample next token
                next_idx = torch.multinomial(probs, 1).item()

                # Convert to character
                next_char = self.idx_to_char.get(next_idx, 'A')
                generated += next_char

                # Update sequence
                current_seq.append(next_idx)

                # Stop if we hit a natural break (optional)
                if len(generated) >= max_length:
                    break

        return generated


# Simple perplexity calculation
def calculate_perplexity(model, data_loader: MiniLanguageModelDataLoader,
                        num_batches: int = 10, batch_size: int = 32) -> float:
    """Calculate perplexity on test data"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = data_loader.generate_batch(batch_size)
            predictions, _, _ = model(inputs)

            loss = F.cross_entropy(
                predictions.view(-1, data_loader.vocab_size),
                targets.view(-1),
                ignore_index=0
            )
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)

    return perplexity


if __name__ == "__main__":
    # Test the data loader
    loader = MiniLanguageModelDataLoader()

    print("Generated sequences:")
    for i in range(5):
        seq = loader.generate_sequence()
        print(f"{i+1}. {seq}")

    print("\nBatch generation test:")
    inputs, targets = loader.generate_batch(4)
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Sample input: {loader.decode_sequence(inputs[0])}")
    print(f"Sample target: {loader.decode_sequence(targets[0])}")
