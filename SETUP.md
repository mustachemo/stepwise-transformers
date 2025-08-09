# Quick Setup Guide

## Prerequisites

1. **Python 3.12+** installed
2. **UV package manager** installed ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
3. **ClearML account** (free at [app.clear.ml](https://app.clear.ml))

## Installation

1. **Clone and setup environment:**
   ```bash
   git clone <your-repo-url>
   cd stepwise-transformers
   uv sync
   ```

2. **Configure ClearML:**
   ```bash
   uv run clearml-init
   ```
   Follow the prompts to connect to your ClearML account.

## Running Experiments

### Basic Usage
```bash
# Run with default configuration
uv run python run_experiment.py

# Use small model for faster training
uv run python run_experiment.py --config-name small_transformer

# Override specific parameters
uv run python run_experiment.py model.d_model=256 training.batch_size=16

# Use development environment settings
uv run python run_experiment.py --config-name development
```

### Advanced Configuration
```bash
# Train with gated feed-forward networks
uv run python run_experiment.py feed_forward_type=gated glu_variant=swiglu

# Custom experiment name and tags
uv run python run_experiment.py logging.experiment_name="my_experiment" logging.tags=["custom","test"]

# Adjust training parameters
uv run python run_experiment.py training.learning_rate=1e-3 training.max_epochs=50
```

## Monitoring Experiments

1. **ClearML Web UI:** Your experiment URL will be printed when you start training
2. **Local logs:** Check the `outputs/` directory for Hydra logs
3. **Checkpoints:** Saved in the `checkpoints/` directory

## Project Structure

```
stepwise-transformers/
├── run_experiment.py          # 🚀 Main entry point
├── config/                    # ⚙️ Hydra configurations
├── stepwise_transformers/     # 📦 Main package
│   ├── attention/            # 🎯 Attention mechanisms
│   ├── positional_encoding/  # 📍 Position encodings
│   ├── feed_forward/         # 🔄 Feed-forward networks
│   ├── layers/              # 🏗️ Transformer layers
│   ├── models/              # 🤖 Complete models
│   └── utils/               # 🛠️ Utilities & ClearML integration
└── pyproject.toml           # 📋 Project configuration
```

## Features Included

✅ **Complete Transformer Architecture**
- Encoder/decoder stacks with PyTorch optimization
- Multiple attention mechanisms and positional encodings
- Standard and gated feed-forward networks

✅ **ClearML Integration**
- Comprehensive experiment tracking
- Attention heatmap visualization
- Training curve monitoring
- Model checkpointing and versioning

✅ **Educational Features**
- Detailed logging and analysis
- Attention weight visualization
- Gradient monitoring
- Layer-wise statistics

✅ **Production Ready**
- Hydra configuration management
- UV dependency management
- Rich console interface
- Proper error handling

## Next Steps

1. **Run your first experiment:** `uv run python run_experiment.py`
2. **Explore configurations:** Check `config/` directory
3. **Monitor in ClearML:** View training progress and visualizations
4. **Experiment with parameters:** Try different model architectures and training settings
