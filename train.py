#!/usr/bin/env python3
"""Transformer training script with MLflow monitoring.

This script demonstrates how to run transformer training and monitor
the process using MLflow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from transformer_manager import TransformerManager
from components.attention import AttentionModule
from components.positional_encoding import PositionalEncodingModule
from components.transformer_block import TransformerBlockModule

console = Console()


def train_attention_mechanism():
    """Train and demonstrate attention mechanisms."""
    console.print("[bold blue]🧠 Training Attention Mechanisms[/bold blue]")

    # Initialize manager
    manager = TransformerManager()
    attention = AttentionModule(manager)

    # Start training run
    run_id = manager.start_training_run(
        "attention_training", "Training attention mechanisms"
    )

    try:
        # Create sample data
        batch_size, seq_length, d_model = 4, 10, 64
        query = torch.randn(batch_size, seq_length, d_model)
        key = torch.randn(batch_size, seq_length, d_model)
        value = torch.randn(batch_size, seq_length, d_model)

        # Log training parameters
        manager.log_component_parameters(
            "attention",
            {
                "batch_size": batch_size,
                "seq_length": seq_length,
                "d_model": d_model,
                "num_heads": 8,
            },
        )

        # Training loop
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training attention...", total=100)

            for step in range(100):
                # Single-head attention
                single_output = attention.single_head_attention(query, key, value)

                # Multi-head attention
                multi_output = attention.multi_head_attention(
                    8, d_model, query, key, value
                )

                # Scaled dot-product attention
                scaled_output = attention.scaled_dot_product_attention(
                    query, key, value
                )

                # Log metrics
                metrics = {
                    "single_head_loss": torch.norm(single_output).item(),
                    "multi_head_loss": torch.norm(multi_output).item(),
                    "scaled_attention_loss": torch.norm(scaled_output).item(),
                    "attention_entropy": torch.softmax(
                        torch.matmul(query, key.transpose(-2, -1)), dim=-1
                    )
                    .entropy()
                    .mean()
                    .item(),
                }

                manager.log_training_metrics(metrics, step)
                progress.update(task, advance=1)

        console.print("[bold green]✅ Attention training completed![/bold green]")

    except Exception as exc:
        console.print(f"[bold red]Error during attention training: {exc}[/bold red]")
        raise


def train_positional_encoding():
    """Train and demonstrate positional encoding."""
    console.print("\n[bold blue]📍 Training Positional Encoding[/bold blue]")

    # Initialize manager
    manager = TransformerManager()
    pos_encoding = PositionalEncodingModule(manager)

    # Start training run
    run_id = manager.start_training_run(
        "positional_encoding_training", "Training positional encoding"
    )

    try:
        # Log training parameters
        manager.log_component_parameters(
            "positional_encoding",
            {
                "seq_length": 20,
                "d_model": 128,
                "encoding_types": ["sinusoidal", "learned", "relative"],
            },
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training positional encoding...", total=50)

            for step in range(50):
                # Different encoding types
                pe_sinusoidal = pos_encoding.sinusoidal_encoding(20, 128)
                pe_learned = pos_encoding.learned_encoding(20, 128)
                pe_relative = pos_encoding.relative_encoding(20, 128, 32)

                # Log metrics
                metrics = {
                    "sinusoidal_norm": torch.norm(pe_sinusoidal).item(),
                    "learned_norm": torch.norm(pe_learned).item(),
                    "relative_norm": torch.norm(pe_relative).item(),
                    "encoding_std": torch.std(pe_sinusoidal).item(),
                }

                manager.log_training_metrics(metrics, step)
                progress.update(task, advance=1)

        console.print(
            "[bold green]✅ Positional encoding training completed![/bold green]"
        )

    except Exception as exc:
        console.print(
            f"[bold red]Error during positional encoding training: {exc}[/bold red]"
        )
        raise


def train_transformer_blocks():
    """Train and demonstrate transformer blocks."""
    console.print("\n[bold blue]🏗️ Training Transformer Blocks[/bold blue]")

    # Initialize manager
    manager = TransformerManager()
    block = TransformerBlockModule(manager)

    # Start training run
    run_id = manager.start_training_run(
        "transformer_block_training", "Training transformer blocks"
    )

    try:
        # Create sample data
        batch_size, seq_length, d_model = 2, 10, 64
        input_tensor = torch.randn(batch_size, seq_length, d_model)

        # Log training parameters
        manager.log_component_parameters(
            "transformer_block",
            {
                "batch_size": batch_size,
                "seq_length": seq_length,
                "d_model": d_model,
                "num_heads": 8,
                "d_ff": 256,
                "dropout": 0.1,
            },
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training transformer blocks...", total=75)

            for step in range(75):
                # Encoder block
                encoder_output = block.encoder_block(input_tensor, d_model, 8, 256, 0.1)

                # Decoder block
                decoder_output = block.decoder_block(
                    input_tensor, encoder_output, d_model, 8, 256, 0.1
                )

                # Log metrics
                metrics = {
                    "encoder_output_norm": torch.norm(encoder_output).item(),
                    "decoder_output_norm": torch.norm(decoder_output).item(),
                    "residual_norm": torch.norm(encoder_output - input_tensor).item(),
                    "block_gradient_norm": torch.norm(
                        encoder_output.grad
                        if encoder_output.grad is not None
                        else torch.tensor(0.0)
                    ).item(),
                }

                manager.log_training_metrics(metrics, step)
                progress.update(task, advance=1)

        console.print(
            "[bold green]✅ Transformer block training completed![/bold green]"
        )

    except Exception as exc:
        console.print(
            f"[bold red]Error during transformer block training: {exc}[/bold red]"
        )
        raise


def run_complete_training():
    """Run complete transformer training pipeline."""
    console.print(
        "[bold blue]🚀 Starting Complete Transformer Training Pipeline[/bold blue]"
    )

    try:
        # Train all components
        train_attention_mechanism()
        train_positional_encoding()
        train_transformer_blocks()

        console.print(
            "\n[bold green]🎉 Complete training pipeline finished![/bold green]"
        )
        console.print("\n[bold yellow]Next Steps:[/bold yellow]")
        console.print(
            "1. Run 'python -c \"from transformer_manager import TransformerManager; TransformerManager().start_monitoring()\"' to start MLflow UI"
        )
        console.print("2. Open http://localhost:5000 in your browser")
        console.print("3. Explore your training runs and metrics")

    except Exception as exc:
        console.print(f"[bold red]Error during training: {exc}[/bold red]")
        raise


def show_monitoring_instructions():
    """Show instructions for monitoring training."""
    console.print("\n[bold blue]📊 MLflow Monitoring Instructions[/bold blue]")

    instructions = Table(title="How to Monitor Your Training")
    instructions.add_column("Step", style="cyan")
    instructions.add_column("Command", style="green")
    instructions.add_column("Description", style="yellow")

    instructions.add_row("1", "python train.py", "Run the complete training pipeline")
    instructions.add_row("2", "mlflow ui", "Start MLflow UI for monitoring")
    instructions.add_row(
        "3", "http://localhost:5000", "Open in browser to view results"
    )
    instructions.add_row("4", "Explore runs", "View metrics, parameters, and artifacts")

    console.print(instructions)

    console.print("\n[bold green]Quick Start:[/bold green]")
    console.print("```bash")
    console.print("# Terminal 1: Run training")
    console.print("python train.py")
    console.print("")
    console.print("# Terminal 2: Start monitoring")
    console.print("mlflow ui")
    console.print("```")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        show_monitoring_instructions()
    else:
        run_complete_training()
