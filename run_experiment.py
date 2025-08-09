#!/usr/bin/env python3
"""Main driver script for running stepwise transformer experiments with ClearML.

This script demonstrates how to use the stepwise-transformers package with
comprehensive ClearML integration for experiment tracking and visualization.

Usage:
    python run_experiment.py
    python run_experiment.py --config-name small_transformer
    python run_experiment.py model.d_model=256 training.batch_size=16
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stepwise_transformers.models import Transformer
from stepwise_transformers.utils import ClearMLTracker, DataProcessor

console = Console()


def create_model(config: DictConfig) -> Transformer:
    """Create transformer model from configuration.

    Args:
        config: Model configuration.

    Returns:
        Transformer model.
    """
    model = Transformer(
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_encoder_layers=config.model.n_encoder_layers,
        n_decoder_layers=config.model.n_decoder_layers,
        d_ff=config.model.d_ff,
        src_vocab_size=config.model.src_vocab_size,
        tgt_vocab_size=config.model.tgt_vocab_size,
        max_seq_length=config.model.max_seq_length,
        dropout=config.model.dropout,
        activation=config.model.activation,
        layer_norm_eps=config.model.layer_norm_eps,
        norm_first=config.model.norm_first,
        pad_token_id=config.model.pad_token_id,
    )

    return model


def setup_training(config: DictConfig, model: nn.Module) -> tuple[optim.Optimizer, nn.Module]:
    """Setup optimizer and loss function.

    Args:
        config: Training configuration.
        model: Model to train.

    Returns:
        Tuple of (optimizer, loss_function).
    """
    # Create optimizer
    if config.training.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.beta1, config.training.beta2),
            eps=config.training.eps,
        )
    elif config.training.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.beta1, config.training.beta2),
            eps=config.training.eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=config.model.pad_token_id)

    return optimizer, criterion


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    tracker: ClearMLTracker,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.
        tracker: ClearML tracker.
        epoch: Current epoch.

    Returns:
        Dictionary with training metrics.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Training Epoch {epoch}", total=len(dataloader))

        for batch_idx, batch in enumerate(dataloader):
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)

            # Prepare decoder input and target
            decoder_input = tgt_ids[:, :-1]  # Remove last token
            target = tgt_ids[:, 1:]  # Remove first token (BOS)

            # Create causal mask for decoder
            tgt_mask = model.generate_square_subsequent_mask(decoder_input.size(1)).to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(src_ids, decoder_input, tgt_mask=tgt_mask)

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Log metrics to ClearML
            global_step = epoch * len(dataloader) + batch_idx
            tracker.log_metric("Training", "batch_loss", loss.item(), global_step)
            tracker.log_learning_rate(optimizer.param_groups[0]["lr"], global_step)

            # Log gradients periodically
            if batch_idx % 10 == 0:
                tracker.log_gradient_norms(model, global_step, log_histogram=False)

            progress.update(task, advance=1)

    avg_loss = total_loss / num_batches
    return {"loss": avg_loss}


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation dataloader.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)

            decoder_input = tgt_ids[:, :-1]
            target = tgt_ids[:, 1:]

            tgt_mask = model.generate_square_subsequent_mask(decoder_input.size(1)).to(device)

            logits = model(src_ids, decoder_input, tgt_mask=tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return {"loss": avg_loss}


@hydra.main(config_path="config", config_name="default_config", version_base=None)
def main(config: DictConfig) -> None:
    """Main training function.

    Args:
        config: Hydra configuration.
    """
    console.print("[bold green]🚀 Starting Stepwise Transformer Experiment[/bold green]")

    # Print configuration
    console.print("\n[bold blue]📋 Configuration:[/bold blue]")
    console.print(OmegaConf.to_yaml(config))

    # Setup device
    if config.device.use_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.device.device_id}")
        console.print(f"[green]✓ Using CUDA device: {device}[/green]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]⚠ Using CPU device[/yellow]")

    # Initialize ClearML tracker
    with ClearMLTracker(
        project_name=config.logging.project_name,
        task_name=config.logging.experiment_name,
        tags=config.logging.tags,
    ) as tracker:
        console.print(f"[green]✓ ClearML experiment initialized[/green]")
        console.print(f"[blue]🔗 Experiment URL: {tracker.get_experiment_url()}[/blue]")

        # Connect configuration to ClearML
        config_dict = OmegaConf.to_container(config, resolve=True)
        tracker.connect_configuration(config_dict)

        # Create data processor and prepare data
        console.print("\n[bold blue]📊 Preparing Data...[/bold blue]")
        data_processor = DataProcessor(
            src_vocab_size=config.model.src_vocab_size,
            tgt_vocab_size=config.model.tgt_vocab_size,
            max_length=config.model.max_seq_length,
        )

        train_loader, val_loader, test_loader = data_processor.prepare_data(
            num_samples=1000,
            train_ratio=config.data.train_split,
            val_ratio=config.data.val_split,
        )

        console.print(
            f"[green]✓ Data prepared: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches[/green]"
        )

        # Create model
        console.print("\n[bold blue]🏗️ Creating Model...[/bold blue]")
        model = create_model(config).to(device)

        # Log model architecture
        tracker.log_model_architecture(model)
        param_counts = model.count_parameters()
        console.print(
            f"[green]✓ Model created with {param_counts['total_parameters']:,} parameters[/green]"
        )

        # Setup training
        optimizer, criterion = setup_training(config, model)
        console.print(f"[green]✓ Training setup complete[/green]")

        # Training loop
        console.print(
            f"\n[bold blue]🎯 Starting Training for {config.training.max_epochs} epochs...[/bold blue]"
        )

        train_losses = []
        val_losses = []

        for epoch in range(config.training.max_epochs):
            console.print(
                f"\n[bold yellow]📈 Epoch {epoch + 1}/{config.training.max_epochs}[/bold yellow]"
            )

            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, tracker, epoch
            )
            train_losses.append(train_metrics["loss"])

            # Log training metrics
            tracker.log_metric("Training", "epoch_loss", train_metrics["loss"], epoch)
            console.print(f"[green]✓ Training Loss: {train_metrics['loss']:.4f}[/green]")

            # Evaluate
            if (epoch + 1) % config.logging.eval_every_n_epochs == 0:
                console.print("[blue]🔍 Evaluating...[/blue]")
                val_metrics = evaluate_model(model, val_loader, criterion, device)
                val_losses.append(val_metrics["loss"])

                tracker.log_metric("Validation", "epoch_loss", val_metrics["loss"], epoch)
                console.print(f"[green]✓ Validation Loss: {val_metrics['loss']:.4f}[/green]")

            # Save checkpoint
            if (epoch + 1) % config.logging.save_every_n_epochs == 0:
                checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pt"
                tracker.save_model_checkpoint(
                    model, optimizer, epoch + 1, train_metrics["loss"], checkpoint_path
                )
                console.print(f"[green]✓ Checkpoint saved: {checkpoint_path}[/green]")

        # Log final training curves
        tracker.log_training_curves(train_losses, val_losses)

        # Final evaluation
        console.print("\n[bold blue]🏁 Final Evaluation...[/bold blue]")
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        tracker.log_metric("Test", "final_loss", test_metrics["loss"], 0)
        console.print(f"[green]✓ Test Loss: {test_metrics['loss']:.4f}[/green]")

        console.print(f"\n[bold green]🎉 Experiment completed successfully![/bold green]")
        console.print(f"[blue]🔗 View results: {tracker.get_experiment_url()}[/blue]")


if __name__ == "__main__":
    main()
