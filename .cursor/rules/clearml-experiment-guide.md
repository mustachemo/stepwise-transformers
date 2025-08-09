# ClearML Experiment Guide

This rule provides guidance for conducting transformer experiments using ClearML for experiment tracking and management.

## ClearML Integration Standards

### Experiment Setup
- Use `Task.init()` to initialize ClearML tasks with descriptive names
- Set project names to organize related experiments: `project_name="stepwise-transformers"`
- Tag experiments with relevant categories: `tags=["transformer", "attention", "training"]`
- Connect configuration dictionaries using `task.connect(config)`

### Logging Best Practices
- Log hyperparameters at the beginning of experiments
- Use `Logger.current_logger()` for metrics and scalars
- Report training metrics every epoch: loss, accuracy, learning rate
- Log model artifacts and checkpoints for reproducibility
- Save attention visualizations as image artifacts

### Model Tracking
- Version model architectures using ClearML's model registry
- Track model performance across different configurations
- Log gradient norms and weight distributions for debugging
- Save intermediate checkpoints for long training runs

### Visualization Guidelines
- Use ClearML's built-in plotting for training curves
- Create attention heatmaps and save as artifacts
- Log model architecture diagrams for documentation
- Track resource utilization (GPU, memory) during training

### Experiment Organization
- Use consistent naming conventions: `{model_type}_{dataset}_{timestamp}`
- Group related experiments using project hierarchies
- Document experiment goals and hypotheses in task descriptions
- Compare experiments using ClearML's comparison tools

## Code Standards for ClearML Integration

### Task Initialization Template
```python
from clearml import Task, Logger

task = Task.init(
    project_name="stepwise-transformers",
    task_name="transformer_attention_experiment",
    tags=["transformer", "attention", "pytorch"]
)

# Connect configuration
config = {
    "model": {"d_model": 512, "n_heads": 8, "n_layers": 6},
    "training": {"batch_size": 32, "learning_rate": 1e-4, "epochs": 100}
}
task.connect(config)
```

### Logging Pattern
```python
logger = Logger.current_logger()

# Log metrics during training
logger.report_scalar("training", "loss", value=loss, iteration=step)
logger.report_scalar("validation", "accuracy", value=acc, iteration=epoch)

# Log artifacts
logger.report_image("attention", "heatmap", image=attention_plot, iteration=epoch)
```

### Model Registry Usage
```python
from clearml import Model

# Register trained model
model = Model(task=task, config_dict=config)
model.update_weights(weights_path="model_checkpoint.pth")
model.publish()
```

This guide ensures consistent and effective use of ClearML for transformer experiments and model development.
