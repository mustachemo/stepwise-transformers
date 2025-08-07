"""Main experiment manager for transformer learning with MLflow.

This module provides the core experiment management functionality for learning
transformer architecture through interactive MLflow experiments.
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
DEFAULT_EXPERIMENT_NAME: str = "transformer_learning"
DEFAULT_TRACKING_URI: str = "sqlite:///mlruns.db"
DEFAULT_ARTIFACT_PATH: str = "mlruns"

# =============================== Configuration =============================== #
# * Use Loguru for robust logging with file rotation and structured output.
logger.add(
    "logs/transformer_experiments_{time}.log",
    level="INFO",
    rotation="10 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
)


# =============================== Data Structures ============================= #
@dataclass
class ExperimentConfig:
    """Configuration for transformer learning experiments."""

    experiment_name: str
    tracking_uri: str
    artifact_path: str
    log_level: str
    auto_save: bool
    enable_ui: bool


# ============================== Main Application ============================= #
class TransformerExperimentManager:
    """Main application for interactive transformer learning with MLflow.

    This application provides a comprehensive environment for learning transformer
    architecture through MLflow experiments, interactive visualization,
    and hands-on experimentation.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the transformer learning experiment manager.

        Args:
            config_path: Path to configuration file. If None, uses default.

        Raises:
            FileNotFoundError: If configuration file is specified but not found.
            OSError: If application directories cannot be created.
        """
        self.config_path = config_path or Path("config.json")
        self.experiment_config: Optional[ExperimentConfig] = None
        self.console = Console()
        self.current_experiment: Optional[mlflow.entities.Experiment] = None

        try:
            self._load_configuration()
            self._setup_mlflow()
            self._setup_directories()
            logger.info("TransformerExperimentManager initialized successfully")
        except (FileNotFoundError, OSError) as exc:
            logger.error(f"Failed to initialize TransformerExperimentManager: {exc}")
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
                self.experiment_config = ExperimentConfig(
                    experiment_name=DEFAULT_EXPERIMENT_NAME,
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
            self.experiment_config = ExperimentConfig(
                experiment_name=DEFAULT_EXPERIMENT_NAME,
                tracking_uri=DEFAULT_TRACKING_URI,
                artifact_path=DEFAULT_ARTIFACT_PATH,
                log_level="INFO",
                auto_save=True,
                enable_ui=True,
            )
            logger.info("Using default configuration")

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking and experiment management.

        Raises:
            RuntimeError: If MLflow setup fails.
        """
        try:
            mlflow.set_tracking_uri(self.experiment_config.tracking_uri)
            mlflow.set_experiment(self.experiment_config.experiment_name)
            self.current_experiment = mlflow.get_experiment_by_name(
                self.experiment_config.experiment_name
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
            Path("experiments").mkdir(exist_ok=True)
            Path("data").mkdir(exist_ok=True)
            Path("models").mkdir(exist_ok=True)
            Path("artifacts").mkdir(exist_ok=True)
            logger.info("Application directories created successfully")
        except OSError as exc:
            logger.error(f"Failed to create application directories: {exc}")
            raise

    def start_new_experiment(self, experiment_name: str, description: str = "") -> str:
        """Start a new MLflow experiment.

        Args:
            experiment_name: Name of the experiment.
            description: Optional description of the experiment.

        Returns:
            str: The experiment ID.

        Raises:
            RuntimeError: If experiment creation fails.
        """
        try:
            with mlflow.start_run(run_name=experiment_name) as run:
                mlflow.log_param("description", description)
                mlflow.log_param("experiment_type", "transformer_learning")
                mlflow.log_param("timestamp", str(mlflow.start_time))

                logger.info(f"Started new experiment: {experiment_name}")
                return run.info.run_id
        except Exception as exc:
            logger.error(f"Failed to start experiment: {exc}")
            raise RuntimeError(f"Experiment creation failed: {exc}")

    def log_transformer_component(
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

    def list_experiments(self) -> List[mlflow.entities.Experiment]:
        """List all available experiments.

        Returns:
            List[mlflow.entities.Experiment]: List of experiments.

        Raises:
            RuntimeError: If experiment listing fails.
        """
        try:
            experiments = mlflow.search_experiments()
            logger.info(f"Found {len(experiments)} experiments")
            return experiments
        except Exception as exc:
            logger.error(f"Failed to list experiments: {exc}")
            raise RuntimeError(f"Experiment listing failed: {exc}")

    def compare_experiments(self, experiment_ids: List[str]) -> None:
        """Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare.

        Raises:
            RuntimeError: If comparison fails.
        """
        try:
            # TODO: Implement experiment comparison logic
            logger.info(f"Comparing {len(experiment_ids)} experiments")
        except Exception as exc:
            logger.error(f"Failed to compare experiments: {exc}")
            raise RuntimeError(f"Experiment comparison failed: {exc}")

    def export_experiment_results(self, experiment_id: str, output_path: Path) -> None:
        """Export experiment results to file.

        Args:
            experiment_id: ID of the experiment to export.
            output_path: Path where to save the results.

        Raises:
            RuntimeError: If export fails.
        """
        try:
            # TODO: Implement experiment export logic
            logger.info(f"Exported experiment {experiment_id} to {output_path}")
        except Exception as exc:
            logger.error(f"Failed to export experiment: {exc}")
            raise RuntimeError(f"Experiment export failed: {exc}")
