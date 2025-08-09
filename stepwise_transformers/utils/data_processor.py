"""Data processing utilities for transformer training.

This module provides simple data processing and tokenization for educational purposes.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class SimpleTokenizer:
    """Simple word-level tokenizer for educational purposes."""

    def __init__(
        self,
        vocab_size: int = 32000,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size.
            special_tokens: Special tokens dictionary.
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<bos>",
            "eos": "<eos>",
        }

        # Initialize vocabulary
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Add special tokens
        for token in self.special_tokens.values():
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        """Add token to vocabulary."""
        if token not in self.vocab:
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.id_to_token[token_id] = token
        return self.vocab[token]

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Simple word splitting
        for text in texts:
            words = text.lower().split()
            for word in words:
                if len(self.vocab) < self.vocab_size:
                    self._add_token(word)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab[self.special_tokens["bos"]])

        for word in words:
            token_id = self.vocab.get(word, self.vocab[self.special_tokens["unk"]])
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.vocab[self.special_tokens["eos"]])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        special_token_ids = {self.vocab[token] for token in self.special_tokens.values()}

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue
            token = self.id_to_token.get(token_id, self.special_tokens["unk"])
            tokens.append(token)

        return " ".join(tokens)

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.vocab[self.special_tokens["pad"]]

    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.vocab[self.special_tokens["unk"]]

    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.vocab[self.special_tokens["bos"]]

    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.vocab[self.special_tokens["eos"]]


class SimpleTranslationDataset(Dataset):
    """Simple translation dataset for educational purposes."""

    def __init__(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        src_tokenizer: SimpleTokenizer,
        tgt_tokenizer: SimpleTokenizer,
        max_length: int = 512,
    ) -> None:
        """Initialize dataset.

        Args:
            src_texts: Source texts.
            tgt_texts: Target texts.
            src_tokenizer: Source tokenizer.
            tgt_tokenizer: Target tokenizer.
            max_length: Maximum sequence length.
        """
        if len(src_texts) != len(tgt_texts):
            raise ValueError("Source and target texts must have same length")

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Encode texts
        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)

        # Truncate to max length
        src_ids = src_ids[: self.max_length]
        tgt_ids = tgt_ids[: self.max_length]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    # Get maximum lengths
    max_src_len = max(len(item["src_ids"]) for item in batch)
    max_tgt_len = max(len(item["tgt_ids"]) for item in batch)

    batch_size = len(batch)

    # Create padded tensors
    src_ids = torch.full((batch_size, max_src_len), pad_token_id, dtype=torch.long)
    tgt_ids = torch.full((batch_size, max_tgt_len), pad_token_id, dtype=torch.long)

    src_texts = []
    tgt_texts = []

    for i, item in enumerate(batch):
        src_len = len(item["src_ids"])
        tgt_len = len(item["tgt_ids"])

        src_ids[i, :src_len] = item["src_ids"]
        tgt_ids[i, :tgt_len] = item["tgt_ids"]

        src_texts.append(item["src_text"])
        tgt_texts.append(item["tgt_text"])

    return {
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
    }


def create_sample_data(num_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """Create sample translation data for demonstration.

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Tuple of (source_texts, target_texts).
    """
    # Simple patterns for demonstration
    templates = [
        ("hello world", "bonjour monde"),
        ("good morning", "bon matin"),
        ("thank you", "merci"),
        ("how are you", "comment allez vous"),
        ("what is your name", "quel est votre nom"),
        ("I love programming", "j'aime la programmation"),
        ("machine learning is fun", "l'apprentissage automatique est amusant"),
        ("neural networks are powerful", "les réseaux de neurones sont puissants"),
        ("transformers changed everything", "les transformers ont tout changé"),
        ("attention is all you need", "l'attention est tout ce dont vous avez besoin"),
    ]

    src_texts = []
    tgt_texts = []

    for _ in range(num_samples):
        src, tgt = random.choice(templates)

        # Add some variation
        if random.random() < 0.3:
            src = f"please {src}"
            tgt = f"s'il vous plaît {tgt}"

        if random.random() < 0.2:
            src = f"{src} please"
            tgt = f"{tgt} s'il vous plaît"

        src_texts.append(src)
        tgt_texts.append(tgt)

    return src_texts, tgt_texts


class DataProcessor:
    """Main data processor for transformer training."""

    def __init__(
        self,
        src_vocab_size: int = 32000,
        tgt_vocab_size: int = 32000,
        max_length: int = 512,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize data processor.

        Args:
            src_vocab_size: Source vocabulary size.
            tgt_vocab_size: Target vocabulary size.
            max_length: Maximum sequence length.
            special_tokens: Special tokens dictionary.
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_length = max_length

        self.src_tokenizer = SimpleTokenizer(src_vocab_size, special_tokens)
        self.tgt_tokenizer = SimpleTokenizer(tgt_vocab_size, special_tokens)

    def prepare_data(
        self,
        num_samples: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test dataloaders.

        Args:
            num_samples: Number of samples to generate.
            train_ratio: Training data ratio.
            val_ratio: Validation data ratio.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        # Generate sample data
        src_texts, tgt_texts = create_sample_data(num_samples)

        # Build vocabularies
        self.src_tokenizer.build_vocab(src_texts)
        self.tgt_tokenizer.build_vocab(tgt_texts)

        # Split data
        n_train = int(len(src_texts) * train_ratio)
        n_val = int(len(src_texts) * val_ratio)

        train_src = src_texts[:n_train]
        train_tgt = tgt_texts[:n_train]

        val_src = src_texts[n_train : n_train + n_val]
        val_tgt = tgt_texts[n_train : n_train + n_val]

        test_src = src_texts[n_train + n_val :]
        test_tgt = tgt_texts[n_train + n_val :]

        # Create datasets
        train_dataset = SimpleTranslationDataset(
            train_src, train_tgt, self.src_tokenizer, self.tgt_tokenizer, self.max_length
        )
        val_dataset = SimpleTranslationDataset(
            val_src, val_tgt, self.src_tokenizer, self.tgt_tokenizer, self.max_length
        )
        test_dataset = SimpleTranslationDataset(
            test_src, test_tgt, self.src_tokenizer, self.tgt_tokenizer, self.max_length
        )

        # Create dataloaders
        def make_collate_fn():
            return lambda batch: collate_fn(batch, self.src_tokenizer.pad_token_id)

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, collate_fn=make_collate_fn()
        )
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False, collate_fn=make_collate_fn()
        )
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, collate_fn=make_collate_fn()
        )

        return train_loader, val_loader, test_loader
