"""Base component class for transformer learning components.

This module provides the base class for all transformer learning components
with common functionality and MLflow integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import mlflow
from loguru import logger


class BaseComponent(ABC):
    """Base class for all transformer learning components."""

    def __init__(self, manager):
        """Initialize the base component.

        Args:
            manager: The main transformer manager instance.
        """
        self.manager = manager
        self.current_run: Optional[mlflow.entities.Run] = None

    def start_run(self, run_name: str, description: str = "") -> str:
        """Start a new MLflow run.

        Args:
            run_name: Name of the run.
            description: Optional description of the run.

        Returns:
            str: The run ID.

        Raises:
            RuntimeError: If run creation fails.
        """
        try:
            self.current_run = mlflow.start_run(run_name=run_name)
            mlflow.log_param("description", description)
            mlflow.log_param("component_type", self.__class__.__name__)

            logger.info(f"Started new run: {run_name}")
            return self.current_run.info.run_id
        except Exception as exc:
            logger.error(f"Failed to start run: {exc}")
            raise RuntimeError(f"Run creation failed: {exc}")

    def end_run(self) -> None:
        """End the current MLflow run.

        Raises:
            RuntimeError: If run ending fails.
        """
        try:
            if self.current_run:
                mlflow.end_run()
                logger.info("Ended current run")
        except Exception as exc:
            logger.error(f"Failed to end run: {exc}")
            raise RuntimeError(f"Run ending failed: {exc}")

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            parameters: Dictionary of parameters to log.

        Raises:
            RuntimeError: If parameter logging fails.
        """
        try:
            for key, value in parameters.items():
                mlflow.log_param(key, value)

            logger.info(f"Logged {len(parameters)} parameters")
        except Exception as exc:
            logger.error(f"Failed to log parameters: {exc}")
            raise RuntimeError(f"Parameter logging failed: {exc}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number for the metrics.

        Raises:
            RuntimeError: If metric logging fails.
        """
        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value, step=step)

            logger.info(f"Logged {len(metrics)} metrics")
        except Exception as exc:
            logger.error(f"Failed to log metrics: {exc}")
            raise RuntimeError(f"Metric logging failed: {exc}")

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact to MLflow.

        Args:
            local_path: Path to the local file to log.
            artifact_path: Optional path within the artifact store.

        Raises:
            RuntimeError: If artifact logging fails.
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as exc:
            logger.error(f"Failed to log artifact: {exc}")
            raise RuntimeError(f"Artifact logging failed: {exc}")

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the component.

        This method should be implemented by subclasses to define
        the specific component logic.

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Returns:
            Any: The component output.
        """
        pass
