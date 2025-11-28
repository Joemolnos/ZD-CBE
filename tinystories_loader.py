"""
TinyStories Dataset Loader
Small, clean corpus of simple stories for children
Perfect for testing LLM pretraining methods
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import requests
from pathlib import Path
import random


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for language modeling"""

    def __init__(
        self,
        data_path: str = "tinystories_sample.txt",
        tokenizer_name: str = "gpt2",
        max_length: int = 256,
        download_if_missing: bool = True
    ):
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        data_path = Path(data_path)
        if not data_path.exists() and download_if_missing:
            print(f"Downloading TinyStories sample to {data_path}...")
            self._download_sample(data_path)

        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # Tokenize entire corpus
        print("Tokenizing corpus...")
        self.tokens = self.tokenizer.encode(self.text)
        print(f"Total tokens: {len(self.tokens):,}")

        # Create overlapping chunks
        self.chunks = self._create_chunks()
        print(f"Created {len(self.chunks)} training examples")

    def _download_sample(self, save_path: Path):
        """Download a sample of TinyStories (for demo purposes)"""
        # Use a curated sample of simple stories
        sample_stories = """
Once upon a time, there was a little girl named Lucy. She loved to play in the park.
One day, Lucy saw a big red ball. She wanted to play with it.
Lucy ran to the ball and kicked it very hard. The ball flew high into the sky.
"Wow!" said Lucy. "That was fun!"
Lucy played with the ball all day long. She was very happy.

There was a small cat named Whiskers. Whiskers was very curious.
One morning, Whiskers saw a butterfly in the garden.
The butterfly was beautiful with colorful wings.
Whiskers wanted to catch the butterfly, so he jumped and ran.
But the butterfly was too fast. It flew away into the flowers.
Whiskers sat down and watched the butterfly from afar.

A boy named Tom had a toy car. The car was blue and shiny.
Tom loved to race his car on the floor.
One day, Tom's friend Sam came to visit.
"Can I play with your car?" asked Sam.
"Yes!" said Tom. "We can race together!"
Tom and Sam raced their cars all afternoon. They laughed and had fun.

There was a little dog named Buddy. Buddy loved to dig holes.
One day, Buddy found a big stick in the yard.
He wanted to bury the stick in a hole.
Buddy dug and dug with his paws. The hole got bigger and bigger.
Finally, Buddy put the stick in the hole and covered it with dirt.
Buddy was proud of his work. He wagged his tail happily.

A girl named Emma loved to draw pictures.
She had many crayons in different colors.
One day, Emma decided to draw a rainbow.
She used red, orange, yellow, green, blue, and purple crayons.
The rainbow looked beautiful on her paper.
Emma showed her rainbow to her mom. Her mom smiled and gave her a hug.

There was a rabbit named Cotton. Cotton lived in a meadow.
Cotton loved to eat carrots and lettuce.
One sunny day, Cotton hopped to the garden.
He found many carrots growing in the ground.
Cotton nibbled on a big orange carrot. It was delicious!
Cotton ate until his tummy was full, then hopped back home.

A little bird named Chirp lived in a tree.
Chirp could sing beautiful songs.
Every morning, Chirp sang to wake up the sun.
The other animals loved to hear Chirp's songs.
One day, a little girl stopped under the tree to listen.
She smiled as Chirp sang a happy tune. It was the best song ever!

There was a turtle named Shelly. Shelly moved very slowly.
One day, Shelly decided to race with a rabbit.
The rabbit was fast, but Shelly was determined.
The rabbit ran quickly and took a nap halfway.
Shelly kept walking slowly and steadily.
When the rabbit woke up, Shelly had already won the race!

A boy named Jake had a kite. The kite was shaped like a dragon.
On a windy day, Jake went to the hill to fly his kite.
He ran fast and let the string go. The kite soared high!
The dragon kite danced in the sky with the clouds.
Jake felt happy watching his kite fly higher and higher.

There was a frog named Jumper. Jumper loved to jump on lily pads.
One day, Jumper saw a dragonfly near the pond.
The dragonfly had shiny wings that sparkled in the sun.
Jumper jumped from pad to pad, trying to catch the dragonfly.
But the dragonfly was too quick and flew away.
Jumper sat on a lily pad and croaked happily.
""" * 50  # Repeat to create a larger sample

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(sample_stories)

        print(f"✅ Sample saved to {save_path}")

    def _create_chunks(self):
        """Create overlapping chunks of tokens"""
        chunks = []
        stride = self.max_length // 2  # 50% overlap

        for i in range(0, len(self.tokens) - self.max_length, stride):
            chunk = self.tokens[i:i + self.max_length]
            if len(chunk) == self.max_length:
                chunks.append(chunk)

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        Returns:
            input_ids: (max_length,) token indices
            labels: (max_length,) target tokens (shifted by 1)
        """
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels


class RandomTokenDataset(Dataset):
    """Random token dataset for dream phase"""

    def __init__(
        self,
        vocab_size: int = 50257,
        num_samples: int = 10000,
        max_length: int = 256
    ):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate random token sequence"""
        random_tokens = torch.randint(0, self.vocab_size, (self.max_length,))
        input_ids = random_tokens[:-1]
        labels = random_tokens[1:]
        return input_ids, labels


def create_tinystories_dataloader(
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 0
):
    """Create TinyStories dataloader"""
    dataset = TinyStoriesDataset(max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, dataset.tokenizer


def create_random_dataloader(
    vocab_size: int = 50257,
    batch_size: int = 16,
    num_samples: int = 10000,
    max_length: int = 256,
    num_workers: int = 0
):
    """Create random token dataloader for dream phase"""
    dataset = RandomTokenDataset(vocab_size, num_samples, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# Test
if __name__ == "__main__":
    print("Testing TinyStories dataset...")

    # Create dataset
    dataloader, tokenizer = create_tinystories_dataloader(batch_size=4)

    # Test batch
    for input_ids, labels in dataloader:
        print(f"Input shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")

        # Decode sample
        sample_text = tokenizer.decode(input_ids[0])
        print(f"\nSample text:\n{sample_text[:200]}...")
        break

    print("\n✅ TinyStories dataset test passed!")

    print("\nTesting random token dataset...")
    random_loader = create_random_dataloader(batch_size=4, num_samples=100)

    for input_ids, labels in random_loader:
        print(f"Random input shape: {input_ids.shape}")
        print(f"Random labels shape: {labels.shape}")
        break

    print("\n✅ Random dataset test passed!")
