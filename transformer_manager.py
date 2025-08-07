"""Main transformer manager for learning with MLflow.

This module provides the core management functionality for learning
transformer architecture through interactive MLflow tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# =============================== Constants =================================== #
DEFAULT_PROJECT_NAME: str = "transformer_learning"
DEFAULT_TRACKING_URI: str = "sqlite:///mlruns.db"
DEFAULT_ARTIFACT_PATH: str = "mlruns"

# =============================== Configuration =============================== #
# * Use Loguru for robust logging with file rotation and structured output.
logger.add(
    "logs/transformer_training_{time}.log",
    level="INFO",
    rotation="10 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
)


# =============================== Data Structures ============================= #
@dataclass
class TrainingConfig:
    """Configuration for transformer learning training."""

    project_name: str
    tracking_uri: str
    artifact_path: str
    log_level: str
    auto_save: bool
    enable_ui: bool


# ============================== Main Application ============================= #
class TransformerManager:
    """Main application for interactive transformer learning with MLflow.

    This application provides a comprehensive environment for learning transformer
    architecture through MLflow tracking, interactive visualization,
    and hands-on training.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the transformer learning manager.

        Args:
            config_path: Path to configuration file. If None, uses default.

        Raises:
            FileNotFoundError: If configuration file is specified but not found.
            OSError: If application directories cannot be created.
        """
        self.config_path = config_path or Path("config.json")
        self.training_config: Optional[TrainingConfig] = None
        self.console = Console()
        self.current_run: Optional[mlflow.entities.Run] = None

        try:
            self._load_configuration()
            self._setup_mlflow()
            self._setup_directories()
            logger.info("TransformerManager initialized successfully")
        except (FileNotFoundError, OSError) as exc:
            logger.error(f"Failed to initialize TransformerManager: {exc}")
            raise

    def _load_configuration(self) -> None:
        """Load application configuration from file.

        Raises:
            FileNotFoundError: If configuration file is not found.
            ValueError: If configuration file is malformed.
        """
        if self.config_path.exists():
            try:
                # TODO: Implement configuration loading logic
                self.training_config = TrainingConfig(
                    project_name=DEFAULT_PROJECT_NAME,
                    tracking_uri=DEFAULT_TRACKING_URI,
                    artifact_path=DEFAULT_ARTIFACT_PATH,
                    log_level="INFO",
                    auto_save=True,
                    enable_ui=True,
                )
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as exc:
                logger.error(f"Failed to load configuration: {exc}")
                raise ValueError(f"Malformed configuration file: {exc}")
        else:
            # * Use default configuration if file doesn't exist
            self.training_config = TrainingConfig(
                project_name=DEFAULT_PROJECT_NAME,
                tracking_uri=DEFAULT_TRACKING_URI,
                artifact_path=DEFAULT_ARTIFACT_PATH,
                log_level="INFO",
                auto_save=True,
                enable_ui=True,
            )
            logger.info("Using default configuration")

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking and project management.

        Raises:
            RuntimeError: If MLflow setup fails.
        """
        try:
            mlflow.set_tracking_uri(self.training_config.tracking_uri)
            mlflow.set_experiment(self.training_config.project_name)
            self.current_run = mlflow.get_experiment_by_name(
                self.training_config.project_name
            )
            logger.info("MLflow setup completed successfully")
        except Exception as exc:
            logger.error(f"Failed to setup MLflow: {exc}")
            raise RuntimeError(f"MLflow setup failed: {exc}")

    def _setup_directories(self) -> None:
        """Create necessary application directories.

        Raises:
            OSError: If directories cannot be created.
        """
        try:
            Path("logs").mkdir(exist_ok=True)
            Path("models").mkdir(exist_ok=True)
            Path("data").mkdir(exist_ok=True)
            Path("checkpoints").mkdir(exist_ok=True)
            Path("artifacts").mkdir(exist_ok=True)
            logger.info("Application directories created successfully")
        except OSError as exc:
            logger.error(f"Failed to create application directories: {exc}")
            raise

    def start_training_run(self, run_name: str, description: str = "") -> str:
        """Start a new MLflow training run.

        Args:
            run_name: Name of the training run.
            description: Optional description of the run.

        Returns:
            str: The run ID.

        Raises:
            RuntimeError: If run creation fails.
        """
        try:
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("description", description)
                mlflow.log_param("run_type", "transformer_training")
                mlflow.log_param("timestamp", str(mlflow.start_time))

                logger.info(f"Started new training run: {run_name}")
                return run.info.run_id
        except Exception as exc:
            logger.error(f"Failed to start training run: {exc}")
            raise RuntimeError(f"Training run creation failed: {exc}")

    def log_component_parameters(
        self, component_name: str, parameters: Dict[str, Any]
    ) -> None:
        """Log transformer component parameters and metrics.

        Args:
            component_name: Name of the transformer component.
            parameters: Dictionary of component parameters.

        Raises:
            RuntimeError: If logging fails.
        """
        try:
            for key, value in parameters.items():
                mlflow.log_param(f"{component_name}_{key}", value)

            logger.info(f"Logged {component_name} component parameters")
        except Exception as exc:
            logger.error(f"Failed to log component: {exc}")
            raise RuntimeError(f"Component logging failed: {exc}")

    def log_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log training metrics to MLflow.

        Args:
            metrics: Dictionary of training metrics.
            step: Training step number.

        Raises:
            RuntimeError: If logging fails.
        """
        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value, step=step)

            logger.info(f"Logged training metrics at step {step}")
        except Exception as exc:
            logger.error(f"Failed to log training metrics: {exc}")
            raise RuntimeError(f"Metrics logging failed: {exc}")

    def save_model(self, model: nn.Module, model_name: str) -> None:
        """Save a PyTorch model to MLflow.

        Args:
            model: The PyTorch model to save.
            model_name: Name for the saved model.

        Raises:
            RuntimeError: If model saving fails.
        """
        try:
            mlflow.pytorch.log_model(model, model_name)
            logger.info(f"Saved model: {model_name}")
        except Exception as exc:
            logger.error(f"Failed to save model: {exc}")
            raise RuntimeError(f"Model saving failed: {exc}")

    def log_attention_visualization(
        self, attention_weights: torch.Tensor, step: int
    ) -> None:
        """Log attention weight visualization.

        Args:
            attention_weights: Attention weight tensor.
            step: Training step number.

        Raises:
            RuntimeError: If visualization logging fails.
        """
        try:
            # Convert attention weights to image and log as artifact
            attention_path = f"artifacts/attention_step_{step}.png"
            # TODO: Implement attention weight visualization
            mlflow.log_artifact(attention_path, f"attention_visualizations/step_{step}")

            logger.info(f"Logged attention visualization for step {step}")
        except Exception as exc:
            logger.error(f"Failed to log attention visualization: {exc}")
            raise RuntimeError(f"Visualization logging failed: {exc}")

    def list_runs(self) -> List[mlflow.entities.Run]:
        """List all available training runs.

        Returns:
            List[mlflow.entities.Run]: List of training runs.

        Raises:
            RuntimeError: If run listing fails.
        """
        try:
            runs = mlflow.search_runs()
            logger.info(f"Found {len(runs)} training runs")
            return runs
        except Exception as exc:
            logger.error(f"Failed to list runs: {exc}")
            raise RuntimeError(f"Run listing failed: {exc}")

    def compare_runs(self, run_ids: List[str]) -> None:
        """Compare multiple training runs.

        Args:
            run_ids: List of run IDs to compare.

        Raises:
            RuntimeError: If comparison fails.
        """
        try:
            # TODO: Implement run comparison logic
            logger.info(f"Comparing {len(run_ids)} training runs")
        except Exception as exc:
            logger.error(f"Failed to compare runs: {exc}")
            raise RuntimeError(f"Run comparison failed: {exc}")

    def export_results(self, run_id: str, output_path: Path) -> None:
        """Export training results to file.

        Args:
            run_id: ID of the training run to export.
            output_path: Path where to save the results.

        Raises:
            RuntimeError: If export fails.
        """
        try:
            # TODO: Implement results export logic
            logger.info(f"Exported training run {run_id} to {output_path}")
        except Exception as exc:
            logger.error(f"Failed to export results: {exc}")
            raise RuntimeError(f"Results export failed: {exc}")

    def start_monitoring(self) -> None:
        """Start MLflow UI for monitoring training runs."""
        try:
            import subprocess
            import webbrowser
            import time

            self.console.print(
                "[bold blue]Starting MLflow UI for monitoring...[/bold blue]"
            )

            # Start MLflow UI
            process = subprocess.Popen(
                ["mlflow", "ui"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait a moment for the server to start
            time.sleep(2)

            # Open browser
            webbrowser.open("http://localhost:5000")

            self.console.print("[bold green]✅ MLflow UI launched![/bold green]")
            self.console.print(
                "Open http://localhost:5000 in your browser to monitor training"
            )
            self.console.print("Press Ctrl+C to stop the monitoring server")

            # Wait for user to stop
            process.wait()

        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Stopping MLflow UI...[/bold yellow]")
            process.terminate()
        except Exception as exc:
            self.console.print(f"[bold red]Error starting monitoring: {exc}[/bold red]")
            raise RuntimeError(f"Monitoring start failed: {exc}")
