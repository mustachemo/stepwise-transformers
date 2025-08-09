"""ClearML integration utilities for transformer experiments.

This module provides comprehensive ClearML integration including experiment tracking,
model logging, visualization, and artifact management.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
import tempfile
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from clearml import Task, Logger, Model, Dataset
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


class ClearMLTracker:
    """Comprehensive ClearML tracker for transformer experiments.

    This class provides all the functionality needed to track transformer
    experiments with ClearML, including metrics, artifacts, model logging,
    and visualization.
    """

    def __init__(
        self,
        project_name: str = "stepwise-transformers",
        task_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        task_type: str = "training",
        auto_connect_frameworks: bool = True,
    ) -> None:
        """Initialize ClearML tracker.

        Args:
            project_name: ClearML project name.
            task_name: Task name. If None, auto-generated.
            tags: List of tags for the experiment.
            task_type: Type of task (training, testing, etc.).
            auto_connect_frameworks: Whether to auto-connect ML frameworks.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not project_name:
            raise ValueError("project_name cannot be empty")

        self.project_name = project_name
        self.task_name = task_name
        self.tags = tags or []

        # Initialize ClearML task
        self.task = Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            tags=self.tags,
            auto_connect_frameworks=auto_connect_frameworks,
        )

        # Get logger
        self.logger = Logger.current_logger()

        # Storage for attention weights and activations
        self._stored_attention_weights: Dict[str, torch.Tensor] = {}
        self._stored_activations: Dict[str, torch.Tensor] = {}

    def connect_configuration(self, config: Dict[str, Any]) -> None:
        """Connect configuration dictionary to ClearML.

        Args:
            config: Configuration dictionary to track.
        """
        self.task.connect(config)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to ClearML.

        Args:
            params: Dictionary of hyperparameters.
        """
        for key, value in params.items():
            self.task.set_parameter(key, value)

    def log_metric(
        self,
        title: str,
        series: str,
        value: float,
        iteration: int,
    ) -> None:
        """Log a single metric value.

        Args:
            title: Metric category/title.
            series: Metric series name.
            value: Metric value.
            iteration: Iteration number.
        """
        self.logger.report_scalar(
            title=title,
            series=series,
            value=value,
            iteration=iteration,
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        iteration: int,
        prefix: str = "",
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value.
            iteration: Iteration number.
            prefix: Optional prefix for metric names.
        """
        for name, value in metrics.items():
            title = f"{prefix}/{name}" if prefix else name
            self.log_metric(title, "value", value, iteration)

    def log_model_architecture(self, model: nn.Module) -> None:
        """Log model architecture and parameter counts.

        Args:
            model: PyTorch model to analyze.
        """
        # Get model summary
        if hasattr(model, "count_parameters"):
            param_counts = model.count_parameters()
            self.log_hyperparameters(param_counts)

        # Log model architecture as text
        model_str = str(model)
        self.task.upload_artifact(
            name="model_architecture",
            artifact_object=model_str,
        )

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.log_hyperparameters({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        })

    def create_or_update_dataset(
        self,
        dataset_name: str,
        items: List[Dict[str, Any]],
        dataset_project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create (or update) a ClearML Dataset version from in-memory items.

        The items will be serialized to a temporary directory as JSONL files and
        published as a new dataset version.

        Args:
            dataset_name: Name of the dataset in ClearML.
            items: List of data records, each a JSON-serializable dict.
            dataset_project: Optional ClearML project for the dataset. Defaults to the current project.
            tags: Optional tags to assign to the dataset version.
            metadata: Optional metadata dictionary to attach to the dataset.

        Returns:
            The ClearML dataset ID of the created/published dataset.
        """
        target_project: str = dataset_project or f"{self.project_name}/datasets"

        # Materialize items into a temporary directory as a single JSONL file
        temp_dir = Path(tempfile.mkdtemp(prefix="clearml_ds_"))
        jsonl_path = temp_dir / "data.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in items:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Create and publish dataset
        ds = Dataset.create(dataset_name=dataset_name, dataset_project=target_project)
        if tags:
            for tag in tags:
                ds.add_tags(tag)
        if metadata:
            ds.set_metadata(metadata)

        ds.add_files(path=str(jsonl_path))
        # Ensure files are uploaded to the ClearML files server or configured storage
        ds.upload()
        ds.finalize()
        ds.publish()

        # Upload the materialized JSONL as an artifact for traceability
        self.task.upload_artifact(
            name=f"dataset_{dataset_name}_jsonl",
            artifact_object=str(jsonl_path),
        )

        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            # Non-fatal cleanup failure
            pass

        return ds.id

    def get_dataset_local_copy(
        self,
        dataset_name: Optional[str] = None,
        dataset_project: Optional[str] = None,
        dataset_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """Retrieve a local copy of a ClearML Dataset version.

        Args:
            dataset_name: Dataset name.
            dataset_project: Dataset project. Defaults to the current project datasets namespace.
            dataset_id: Specific dataset ID to retrieve.
            tags: Optional list of tags to filter by (e.g., version tag).

        Returns:
            Local filesystem path containing the dataset files.

        Raises:
            ValueError: If neither dataset_id nor dataset_name is provided.
        """
        if not (dataset_id or dataset_name):
            raise ValueError("Provide either dataset_id or dataset_name to retrieve a dataset")

        target_project: Optional[str] = dataset_project or f"{self.project_name}/datasets"

        ds: Dataset
        if dataset_id:
            ds = Dataset.get(dataset_id=dataset_id)
        else:
            ds = Dataset.get(dataset_name=dataset_name, dataset_project=target_project, dataset_tags=tags)

        local_path = Path(ds.get_local_copy())
        return local_path

    def log_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        title: str = "Attention Heatmap",
        iteration: int = 0,
        head_idx: Optional[int] = None,
    ) -> None:
        """Log attention weights as heatmap visualization.

        Args:
            attention_weights: Attention weights tensor.
            title: Plot title.
            iteration: Iteration number.
            head_idx: Specific attention head to visualize (if multi-head).
        """
        # Convert to numpy and handle different tensor shapes
        if attention_weights.dim() == 4:  # (batch, heads, seq, seq)
            attn = attention_weights[0]  # Take first batch
            if head_idx is not None:
                attn = attn[head_idx]  # Take specific head
            else:
                attn = attn.mean(dim=0)  # Average across heads
        elif attention_weights.dim() == 3:  # (batch, seq, seq) or (heads, seq, seq)
            attn = attention_weights[0] if attention_weights.shape[0] > 1 else attention_weights
        else:
            attn = attention_weights

        attn_np = attn.detach().cpu().numpy()

        # Create heatmap using plotly
        fig = go.Figure(
            data=go.Heatmap(
                z=attn_np,
                colorscale="Viridis",
                colorbar=dict(title="Attention Weight"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Key Position",
            yaxis_title="Query Position",
        )

        # Log to ClearML
        self.logger.report_plotly(
            title="Attention",
            series=title,
            figure=fig,
            iteration=iteration,
        )

    def log_training_curves(
        self,
        train_losses: list[float],
        val_losses: Optional[list[float]] = None,
        train_accuracies: Optional[list[float]] = None,
        val_accuracies: Optional[list[float]] = None,
        title: str = "Training Curves",
    ) -> None:
        """Log training curves visualization.

        Args:
            train_losses: Training loss values.
            val_losses: Validation loss values.
            train_accuracies: Training accuracy values.
            val_accuracies: Validation accuracy values.
            title: Plot title.
        """
        fig = go.Figure()

        epochs = list(range(1, len(train_losses) + 1))

        # Add loss curves
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses,
                mode="lines",
                name="Train Loss",
                line=dict(color="blue"),
            )
        )

        if val_losses:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=val_losses,
                    mode="lines",
                    name="Val Loss",
                    line=dict(color="red"),
                )
            )

        # Add accuracy curves if provided
        if train_accuracies:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_accuracies,
                    mode="lines",
                    name="Train Accuracy",
                    line=dict(color="green"),
                    yaxis="y2",
                )
            )

        if val_accuracies:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=val_accuracies,
                    mode="lines",
                    name="Val Accuracy",
                    line=dict(color="orange"),
                    yaxis="y2",
                )
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="Accuracy",
                overlaying="y",
                side="right",
            ),
        )

        self.logger.report_plotly(
            title="Training",
            series=title,
            figure=fig,
            iteration=0,
        )

    def export_torchscript(
        self,
        model: nn.Module,
        example_inputs: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        export_path: Union[str, Path],
    ) -> Path:
        """Export the provided model as a TorchScript file and log it.

        Args:
            model: The PyTorch model to export.
            example_inputs: Tuple of (src_ids, decoder_input, tgt_mask) used for tracing.
            export_path: Path where the TorchScript file will be written.

        Returns:
            Path to the exported TorchScript file.
        """
        model.eval()
        src_ids, decoder_input, tgt_mask = example_inputs
        scripted = torch.jit.trace(model, (src_ids, decoder_input, tgt_mask))
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(export_path))

        # Upload as artifact
        self.task.upload_artifact(name="model_torchscript", artifact_object=str(export_path))
        return export_path

    def register_model_artifact(
        self,
        model_path: Union[str, Path],
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
    ) -> Model:
        """Register a model file in the ClearML model registry and publish it.

        Args:
            model_path: Path to the model file (e.g., checkpoint, TorchScript, ONNX).
            name: Optional display name in the registry.
            tags: Optional tags to attach (e.g., ["serving-ready"]).
            comment: Optional free-text comment/description.

        Returns:
            The published ClearML Model instance.
        """
        model_path = Path(model_path)
        registry_name = name or model_path.name
        mdl = Model(task=self.task, name=registry_name)
        if tags:
            for tag in tags:
                mdl.add_tags(tag)
        if comment:
            mdl.set_comment(comment)
        mdl.update_weights(weights_path=str(model_path))
        mdl.publish()
        return mdl

    def log_gradient_norms(
        self,
        model: nn.Module,
        iteration: int,
        log_histogram: bool = True,
    ) -> None:
        """Log gradient norms for monitoring training stability.

        Args:
            model: PyTorch model.
            iteration: Iteration number.
            log_histogram: Whether to log gradient histogram.
        """
        total_norm = 0.0
        param_count = 0

        gradient_norms = {}
        all_gradients = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

                gradient_norms[f"grad_norm/{name}"] = param_norm.item()
                all_gradients.extend(param.grad.data.flatten().cpu().numpy())

        total_norm = total_norm ** (1.0 / 2)

        # Log total gradient norm
        self.log_metric("Gradients", "total_norm", total_norm, iteration)

        # Log individual parameter gradient norms
        for name, norm in gradient_norms.items():
            title, series = name.split("/", 1)
            self.log_metric(title, series, norm, iteration)

        # Log gradient histogram
        if log_histogram and all_gradients:
            self.logger.report_histogram(
                title="Gradients",
                series="gradient_distribution",
                values=all_gradients,
                iteration=iteration,
            )

    def log_learning_rate(self, lr: float, iteration: int) -> None:
        """Log current learning rate.

        Args:
            lr: Learning rate value.
            iteration: Iteration number.
        """
        self.log_metric("Training", "learning_rate", lr, iteration)

    def save_model_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        checkpoint_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint and register with ClearML.

        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch.
            loss: Current loss.
            checkpoint_path: Path to save checkpoint.
            metadata: Additional metadata.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Create checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "metadata": metadata or {},
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Register model with ClearML
        model_obj = Model(task=self.task)
        model_obj.update_weights(weights_path=str(checkpoint_path))
        model_obj.publish()

        # Log checkpoint info
        self.task.upload_artifact(
            name=f"checkpoint_epoch_{epoch}",
            artifact_object=str(checkpoint_path),
        )

    def log_text_generation_sample(
        self,
        input_text: str,
        generated_text: str,
        title: str = "Text Generation",
        iteration: int = 0,
    ) -> None:
        """Log text generation samples.

        Args:
            input_text: Input text.
            generated_text: Generated text.
            title: Log title.
            iteration: Iteration number.
        """
        sample_text = f"Input: {input_text}\nGenerated: {generated_text}"

        # Use a conservative signature compatible across ClearML versions
        msg = f"[{title}] {sample_text}"
        self.logger.report_text(msg)

    def log_confusion_matrix(
        self,
        y_true: list[int],
        y_pred: list[int],
        class_names: Optional[list[str]] = None,
        title: str = "Confusion Matrix",
        iteration: int = 0,
    ) -> None:
        """Log confusion matrix visualization.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            class_names: Class names for labels.
            title: Plot title.
            iteration: Iteration number.
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        fig = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            color_continuous_scale="Blues",
            title=title,
        )

        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )

        self.logger.report_plotly(
            title="Evaluation",
            series=title,
            figure=fig,
            iteration=iteration,
        )

    def finish_experiment(self) -> None:
        """Mark experiment as completed."""
        self.task.mark_completed()

    def get_experiment_url(self) -> str:
        """Get the ClearML experiment URL.

        Returns:
            URL to the ClearML experiment.
        """
        return self.task.get_output_log_web_page()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.finish_experiment()
        else:
            # Log the exception with a version-tolerant signature
            self.logger.report_text(f"Exception occurred: {exc_type.__name__}: {exc_val}")
