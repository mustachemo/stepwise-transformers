"""
Training script for transformer model with ClearML integration.

This script demonstrates how to train a small transformer model with
comprehensive experiment tracking using ClearML.
"""

import argparse
import json
import time
from typing import Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from models.transformer import Transformer
from utils.clearml_utils import ClearMLTracker
from utils.data_utils import (
    SimpleTokenizer,
    TokenizerConfig,
    TextPairDataset,
    create_sample_translation_data,
    create_dataloader,
)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        "model": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "d_ff": 1024,
            "max_seq_len": 128,
            "dropout_rate": 0.1,
            "activation": "relu",
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 50,
            "warmup_steps": 1000,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
            "gradient_clip_norm": 1.0,
        },
        "data": {
            "num_samples": 5000,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "max_src_len": 64,
            "max_tgt_len": 64,
        },
        "tokenizer": {"vocab_size": 5000, "max_seq_len": 128},
        "logging": {"log_every": 50, "eval_every": 500, "save_every": 1000},
    }


def calculate_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, pad_token_id: int
) -> float:
    """
    Calculate token-level accuracy excluding padding tokens.

    Args:
        predictions: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Target tokens of shape (batch_size, seq_len)
        pad_token_id: Padding token ID to ignore

    Returns:
        Accuracy score
    """
    pred_tokens = predictions.argmax(dim=-1)

    # Create mask for non-padding tokens
    mask = targets != pad_token_id

    # Calculate accuracy only for non-padding tokens
    correct = (pred_tokens == targets) & mask
    total = mask.sum()

    if total == 0:
        return 0.0

    return (correct.sum().float() / total.float()).item()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    tracker: ClearMLTracker,
    epoch: int,
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device to train on
        tracker: ClearML tracker
        epoch: Current epoch number
        config: Training configuration

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)

        # Forward pass
        optimizer.zero_grad()

        _, decoder_output, attention_weights = model(src, tgt_input)

        # Calculate loss
        loss = criterion(
            decoder_output.reshape(-1, decoder_output.size(-1)), tgt_output.reshape(-1)
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["training"]["gradient_clip_norm"]
        )

        optimizer.step()

        # Calculate metrics
        batch_loss = loss.item()
        batch_accuracy = calculate_accuracy(decoder_output, tgt_output, model.pad_token)

        total_loss += batch_loss
        total_accuracy += batch_accuracy

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "acc": f"{batch_accuracy:.4f}",
        })

        # Log metrics to ClearML
        global_step = epoch * num_batches + batch_idx

        if global_step % config["logging"]["log_every"] == 0:
            tracker.log_metrics(
                {"train_loss": batch_loss, "train_accuracy": batch_accuracy},
                step=global_step,
            )

            # Log gradient norms periodically
            tracker.log_gradient_norms(model, step=global_step)

        # Log attention visualization periodically
        if global_step % config["logging"]["eval_every"] == 0 and global_step > 0:
            with torch.no_grad():
                # Get attention weights from last layer
                encoder_attn = attention_weights["encoder"][
                    f"layer_{config['model']['n_layers'] - 1}"
                ]

                tracker.log_attention_heatmap(
                    encoder_attn,
                    layer_name="encoder_final",
                    head_idx=0,
                    step=global_step,
                )

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "validation",
) -> Tuple[float, float]:
    """
    Evaluate model on validation/test set.

    Args:
        model: Transformer model
        dataloader: Validation/test data loader
        criterion: Loss criterion
        device: Device to evaluate on
        split_name: Name of the split being evaluated

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            # Move batch to device
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_output = batch["tgt_output"].to(device)

            # Forward pass
            _, decoder_output, _ = model(src, tgt_input)

            # Calculate loss
            loss = criterion(
                decoder_output.reshape(-1, decoder_output.size(-1)),
                tgt_output.reshape(-1),
            )

            # Calculate metrics
            batch_loss = loss.item()
            batch_accuracy = calculate_accuracy(
                decoder_output, tgt_output, model.pad_token
            )

            total_loss += batch_loss
            total_accuracy += batch_accuracy

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train transformer with ClearML")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--project-name",
        type=str,
        default="stepwise-transformers",
        help="ClearML project name",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="transformer_training",
        help="ClearML task name",
    )
    parser.add_argument(
        "--tags", nargs="+", default=["transformer", "training"], help="ClearML tags"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = create_default_config()

    # Initialize ClearML tracker
    tracker = ClearMLTracker(
        project_name=args.project_name, task_name=args.task_name, tags=args.tags
    )

    # Connect configuration
    tracker.connect_config(config)

    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Create sample data
    print("Creating sample translation data...")
    source_texts, target_texts = create_sample_translation_data(
        config["data"]["num_samples"]
    )

    # Initialize tokenizer
    tokenizer_config = TokenizerConfig(**config["tokenizer"])
    tokenizer = SimpleTokenizer(tokenizer_config)

    # Build vocabulary
    all_texts = source_texts + target_texts
    tokenizer.build_vocab(all_texts)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create datasets
    dataset = TextPairDataset(
        source_texts,
        target_texts,
        tokenizer,
        max_src_len=config["data"]["max_src_len"],
        max_tgt_len=config["data"]["max_tgt_len"],
    )

    # Split dataset
    train_size = int(config["data"]["train_split"] * len(dataset))
    val_size = int(config["data"]["val_split"] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = create_dataloader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )
    test_loader = create_dataloader(
        test_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Initialize model
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout_rate=config["model"]["dropout_rate"],
        activation=config["model"]["activation"],
        pad_token=tokenizer.pad_token_id,
    ).to(device)

    # Log model architecture
    tracker.log_model_architecture(
        model, (config["training"]["batch_size"], config["model"]["max_seq_len"])
    )

    # Initialize optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=config["training"]["label_smoothing"],
    )

    # Log hyperparameters
    hyperparams = {
        **config["model"],
        **config["training"],
        "vocab_size": tokenizer.vocab_size,
        "device": str(device),
    }
    tracker.log_hyperparameters(hyperparams)

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Starting training...")
    for epoch in range(config["training"]["num_epochs"]):
        start_time = time.time()

        # Train epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, tracker, epoch, config
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(
            model, val_loader, criterion, device, "validation"
        )

        # Log epoch metrics
        epoch_time = time.time() - start_time

        tracker.log_metrics(
            {
                "epoch_train_loss": train_loss,
                "epoch_train_accuracy": train_acc,
                "epoch_val_loss": val_loss,
                "epoch_val_accuracy": val_acc,
                "epoch_time": epoch_time,
            },
            step=epoch,
        )

        # Store for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s"
        )

        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = tracker.save_model_checkpoint(
                model, optimizer, epoch, val_loss
            )
            print(f"Saved best model checkpoint: {checkpoint_path}")

    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, "test")

    tracker.log_metrics({"final_test_loss": test_loss, "final_test_accuracy": test_acc})

    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Log training curves
    tracker.log_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_metrics={"accuracy": train_accuracies},
        val_metrics={"accuracy": val_accuracies},
    )

    # Finish ClearML task
    tracker.finish()

    print("Training completed!")


if __name__ == "__main__":
    main()
