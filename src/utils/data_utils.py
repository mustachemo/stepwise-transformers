"""
Data utilities for transformer training and evaluation.

This module provides utilities for data loading, preprocessing, and
tokenization for transformer models.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration for simple tokenizer."""

    vocab_size: int = 10000
    max_seq_len: int = 512
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    start_token: str = "<START>"
    end_token: str = "<END>"


class SimpleTokenizer:
    """
    Simple word-level tokenizer for demonstration purposes.

    In practice, you would use a more sophisticated tokenizer like
    BPE or SentencePiece.
    """

    def __init__(self, config: TokenizerConfig):
        """
        Initialize tokenizer.

        Args:
            config: Tokenizer configuration
        """
        self.config = config
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        self.vocab_size = 0

        # Initialize special tokens
        special_tokens = [
            config.pad_token,
            config.unk_token,
            config.start_token,
            config.end_token,
        ]

        for token in special_tokens:
            self._add_token(token)

        self.pad_token_id = self.vocab_to_id[config.pad_token]
        self.unk_token_id = self.vocab_to_id[config.unk_token]
        self.start_token_id = self.vocab_to_id[config.start_token]
        self.end_token_id = self.vocab_to_id[config.end_token]

    def _add_token(self, token: str) -> int:
        """Add token to vocabulary."""
        if token not in self.vocab_to_id:
            token_id = self.vocab_size
            self.vocab_to_id[token] = token_id
            self.id_to_vocab[token_id] = token
            self.vocab_size += 1
            return token_id
        return self.vocab_to_id[token]

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from list of texts.

        Args:
            texts: List of text strings
        """
        # Count word frequencies
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add most frequent words to vocabulary
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        max_vocab = self.config.vocab_size - len(
            self.vocab_to_id
        )  # Reserve space for special tokens

        for word, _ in sorted_words[:max_vocab]:
            self._add_token(word)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text string to encode
            add_special_tokens: Whether to add start/end tokens

        Returns:
            List of token IDs
        """
        words = text.lower().split()
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.start_token_id)

        for word in words:
            token_id = self.vocab_to_id.get(word, self.unk_token_id)
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.end_token_id)

        # Truncate if too long
        if len(token_ids) > self.config.max_seq_len:
            token_ids = token_ids[: self.config.max_seq_len - 1] + [self.end_token_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        tokens = []
        special_token_ids = {self.pad_token_id, self.start_token_id, self.end_token_id}

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue

            if token_id == self.end_token_id:
                break

            token = self.id_to_vocab.get(token_id, self.config.unk_token)
            tokens.append(token)

        return " ".join(tokens)

    def pad_sequence(
        self, token_ids: List[int], max_length: Optional[int] = None
    ) -> List[int]:
        """
        Pad sequence to specified length.

        Args:
            token_ids: List of token IDs
            max_length: Maximum length (uses config default if None)

        Returns:
            Padded token IDs
        """
        if max_length is None:
            max_length = self.config.max_seq_len

        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        else:
            padding = [self.pad_token_id] * (max_length - len(token_ids))
            return token_ids + padding


class TextPairDataset(Dataset):
    """
    Dataset for sequence-to-sequence tasks.

    Handles pairs of source and target sequences.
    """

    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        tokenizer: SimpleTokenizer,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            source_texts: List of source text strings
            target_texts: List of target text strings
            tokenizer: Tokenizer instance
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
        """
        assert len(source_texts) == len(target_texts)

        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len or tokenizer.config.max_seq_len
        self.max_tgt_len = max_tgt_len or tokenizer.config.max_seq_len

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized source and target sequences
        """
        src_text = self.source_texts[idx]
        tgt_text = self.target_texts[idx]

        # Encode sequences
        src_tokens = self.tokenizer.encode(src_text, add_special_tokens=False)
        tgt_tokens = self.tokenizer.encode(tgt_text, add_special_tokens=True)

        # Pad sequences
        src_tokens = self.tokenizer.pad_sequence(src_tokens, self.max_src_len)
        tgt_tokens = self.tokenizer.pad_sequence(tgt_tokens, self.max_tgt_len)

        # Create input and target for decoder
        # Input: <START> + tokens[:-1]
        # Target: tokens[1:] + <END>
        tgt_input = tgt_tokens[:-1]
        tgt_output = tgt_tokens[1:]

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_output": torch.tensor(tgt_output, dtype=torch.long),
        }


def create_sample_translation_data(
    num_samples: int = 1000,
) -> Tuple[List[str], List[str]]:
    """
    Create sample translation data for demonstration.

    Args:
        num_samples: Number of samples to generate

    Returns:
        Tuple of (source_texts, target_texts)
    """
    # Simple patterns for demonstration
    patterns = [
        ("hello world", "hola mundo"),
        ("good morning", "buenos días"),
        ("thank you", "gracias"),
        ("how are you", "cómo estás"),
        ("goodbye", "adiós"),
        ("please", "por favor"),
        ("excuse me", "disculpe"),
        ("I love you", "te amo"),
        ("what time is it", "qué hora es"),
        ("where is the bathroom", "dónde está el baño"),
    ]

    source_texts = []
    target_texts = []

    for i in range(num_samples):
        src, tgt = patterns[i % len(patterns)]

        # Add some variation
        if i % 3 == 0:
            src = f"please {src}"
            tgt = f"por favor {tgt}"
        elif i % 3 == 1:
            src = f"{src} please"
            tgt = f"{tgt} por favor"

        source_texts.append(src)
        target_texts.append(tgt)

    return source_texts, target_texts


def create_dataloader(
    dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for training or evaluation.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
