"""
Main entry point for the stepwise transformers project.

This module provides a command-line interface for training and evaluating
transformer models with ClearML integration.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from train import main as train_main


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stepwise Transformers - Learn transformers with ClearML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python -m src.main train

  # Train with custom configuration
  python -m src.main train --config config.json

  # Train with custom ClearML settings
  python -m src.main train --project-name "my-transformers" --task-name "experiment-1"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a transformer model")
    train_parser.add_argument("--config", type=str, help="Path to config file")
    train_parser.add_argument(
        "--project-name",
        type=str,
        default="stepwise-transformers",
        help="ClearML project name",
    )
    train_parser.add_argument(
        "--task-name",
        type=str,
        default="transformer_training",
        help="ClearML task name",
    )
    train_parser.add_argument(
        "--tags", nargs="+", default=["transformer", "training"], help="ClearML tags"
    )

    args = parser.parse_args()

    if args.command == "train":
        # Import and run training
        import train

        # Set up arguments for training script
        train_args = argparse.Namespace(
            config=args.config,
            project_name=args.project_name,
            task_name=args.task_name,
            tags=args.tags,
        )

        # Override sys.argv to pass arguments to train script
        original_argv = sys.argv
        sys.argv = ["train.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        sys.argv.extend(["--project-name", args.project_name])
        sys.argv.extend(["--task-name", args.task_name])
        sys.argv.extend(["--tags"] + args.tags)

        try:
            train.main()
        finally:
            sys.argv = original_argv

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
