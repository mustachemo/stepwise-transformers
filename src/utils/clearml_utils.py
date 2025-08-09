"""
ClearML utilities for experiment tracking and model management.

This module provides utilities for integrating transformer training and
evaluation with ClearML for experiment tracking, visualization, and
model registry.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from clearml import Task, Logger, Model
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("Warning: ClearML not available. Install with: pip install clearml")


class ClearMLTracker:
    """
    ClearML experiment tracker for transformer training.
    
    Handles task initialization, configuration logging, metrics tracking,
    and artifact management.
    """
    
    def __init__(
        self,
        project_name: str = "stepwise-transformers",
        task_name: str = "transformer_experiment",
        tags: Optional[List[str]] = None
    ):
        """
        Initialize ClearML tracker.
        
        Args:
            project_name: ClearML project name
            task_name: Task name for this experiment
            tags: Optional list of tags for the experiment
        """
        if not CLEARML_AVAILABLE:
            raise ImportError("ClearML is required but not installed")
            
        self.task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags or ["transformer", "pytorch"]
        )
        
        self.logger = Logger.current_logger()
        self.step = 0
        
    def connect_config(self, config: Dict[str, Any]) -> None:
        """
        Connect configuration dictionary to ClearML.
        
        Args:
            config: Configuration dictionary to track
        """
        self.task.connect(config)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log training/validation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (uses internal counter if not provided)
        """
        if step is None:
            step = self.step
            self.step += 1
            
        for metric_name, value in metrics.items():
            # Determine series name based on metric name
            if 'train' in metric_name.lower():
                series = 'training'
            elif 'val' in metric_name.lower() or 'valid' in metric_name.lower():
                series = 'validation'
            elif 'test' in metric_name.lower():
                series = 'test'
            else:
                series = 'metrics'
                
            # Clean metric name
            clean_name = metric_name.replace('train_', '').replace('val_', '').replace('test_', '')
            
            self.logger.report_scalar(
                title=series,
                series=clean_name,
                value=value,
                iteration=step
            )
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameter names and values
        """
        for name, value in hyperparams.items():
            self.logger.report_single_value(name=name, value=value)
    
    def log_model_architecture(self, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> None:
        """
        Log model architecture information.
        
        Args:
            model: PyTorch model
            input_shape: Shape of model input for parameter counting
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log parameter counts
        self.logger.report_single_value("total_parameters", total_params)
        self.logger.report_single_value("trainable_parameters", trainable_params)
        
        # Log model summary as text
        model_summary = str(model)
        self.logger.report_text(log_text=model_summary, print_console=False)
    
    def log_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_name: str = "attention",
        head_idx: int = 0,
        step: Optional[int] = None
    ) -> None:
        """
        Log attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights tensor of shape (batch, heads, seq_len, seq_len)
            tokens: Optional list of token strings for labeling
            layer_name: Name of the attention layer
            head_idx: Index of attention head to visualize
            step: Optional step number
        """
        if step is None:
            step = self.step
            
        # Extract attention weights for specified head
        # Take first batch item and specified head
        attn = attention_weights[0, head_idx].detach().cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        if tokens is not None:
            sns.heatmap(
                attn,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                cbar=True
            )
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        else:
            sns.heatmap(attn, cmap='Blues', cbar=True)
            
        plt.title(f'{layer_name} - Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        
        # Log to ClearML
        self.logger.report_matplotlib_figure(
            title="attention_heatmaps",
            series=f"{layer_name}_head_{head_idx}",
            figure=plt.gcf(),
            iteration=step
        )
        
        plt.close()
    
    def log_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        Log training curves as plots.
        
        Args:
            train_losses: List of training losses
            val_losses: Optional list of validation losses
            train_metrics: Optional dictionary of training metrics
            val_metrics: Optional dictionary of validation metrics
        """
        # Loss curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Metrics curves
        if train_metrics or val_metrics:
            plt.subplot(1, 2, 2)
            
            if train_metrics:
                for metric_name, values in train_metrics.items():
                    plt.plot(values, label=f'Train {metric_name}', linestyle='-')
                    
            if val_metrics:
                for metric_name, values in val_metrics.items():
                    plt.plot(values, label=f'Val {metric_name}', linestyle='--')
                    
            plt.title('Metrics Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Log to ClearML
        self.logger.report_matplotlib_figure(
            title="training_progress",
            series="curves",
            figure=plt.gcf(),
            iteration=0
        )
        
        plt.close()
    
    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        checkpoint_dir: str = "checkpoints"
    ) -> str:
        """
        Save model checkpoint and register with ClearML.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            epoch: Current epoch number
            loss: Current loss value
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Path to saved checkpoint file
        """
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save checkpoint
        filename = f"checkpoint_epoch_{epoch}.pth"
        filepath = checkpoint_path / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        torch.save(checkpoint, filepath)
        
        # Register with ClearML
        model_obj = Model(task=self.task, config_dict=checkpoint)
        model_obj.update_weights(weights_path=str(filepath))
        model_obj.publish()
        
        return str(filepath)
    
    def log_gradient_norms(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None
    ) -> None:
        """
        Log gradient norms for model parameters.
        
        Args:
            model: PyTorch model
            step: Optional step number
        """
        if step is None:
            step = self.step
            
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log individual parameter gradients
                self.logger.report_scalar(
                    title="gradients",
                    series=name,
                    value=param_norm.item(),
                    iteration=step
                )
        
        total_norm = total_norm ** (1. / 2)
        
        # Log total gradient norm
        self.logger.report_scalar(
            title="gradients",
            series="total_norm",
            value=total_norm,
            iteration=step
        )
    
    def finish(self) -> None:
        """Finish the ClearML task."""
        self.task.close()
