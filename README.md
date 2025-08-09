# Stepwise Transformers

A comprehensive implementation of transformer architecture with ClearML integration for experiment tracking and visualization.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- ClearML account (free at [app.clear.ml](https://app.clear.ml))

### Installation & Setup
```bash
# 1. Clone and setup
git clone <your-repo-url>
cd stepwise-transformers
uv sync

# 2. Configure ClearML
uv run clearml-init

# 3. Run your first experiment!
uv run python run_experiment.py
```

### Basic Usage
```bash
# Default configuration
uv run python run_experiment.py

# Small model for faster training
uv run python run_experiment.py --config-name small_transformer

# Custom parameters
uv run python run_experiment.py model.d_model=256 training.batch_size=16
```

## 📋 Overview

This project provides an educational yet production-ready implementation of transformers, featuring comprehensive ClearML integration for experiment tracking, model management, and visualization. Perfect for learning transformer architecture while maintaining professional-grade code quality.

## ✨ Features

- ✅ **Complete Transformer Implementation**
  - Encoder/decoder stacks with PyTorch optimization
  - Multiple attention mechanisms and positional encodings
  - Standard and gated feed-forward networks

- ✅ **ClearML Integration**
  - Comprehensive experiment tracking
  - Attention heatmap visualization
  - Training curve monitoring
  - Model checkpointing and versioning

- ✅ **Educational Features**
  - Detailed logging and analysis
  - Attention weight visualization
  - Gradient monitoring
  - Layer-wise statistics

- ✅ **Production Ready**
  - Hydra configuration management
  - UV dependency management
  - Rich console interface
  - Proper error handling

## 🏗️ Architecture

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

## 🛠️ Technology Stack

- **PyTorch**: Deep learning framework with optimized implementations
- **ClearML**: Comprehensive experiment tracking and model management
- **Hydra**: Flexible configuration management with overrides
- **UV**: Lightning-fast Python package management
- **Seaborn/Plotly**: Interactive attention visualization
- **Rich**: Beautiful CLI interfaces and progress tracking

## 🎯 Key Features

### ClearML Integration
- **Experiment Tracking**: Automatic logging of hyperparameters, metrics, and artifacts
- **Model Management**: Version control and checkpoint management
- **Visualization**: Attention heatmaps, training curves, gradient monitoring
- **Collaboration**: Share experiments and results with team members

### Educational Value
- **Step-by-step Learning**: Each component can be studied independently
- **Comprehensive Analysis**: Detailed statistics and visualizations
- **Both Implementations**: PyTorch native vs. custom educational implementations
- **Real-time Monitoring**: Watch your model learn with live visualizations

### Production Quality
- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Comprehensive validation and error messages
- **Performance**: Leverages PyTorch's optimized implementations
- **Scalability**: Easy to extend and modify for different use cases

## 📚 Learning Path

1. **🌱 Start Simple**: Run `run_experiment.py` and explore the outputs
2. **🔧 Understand Config**: Modify `config/default_config.yaml`
3. **👁️ Visualize**: Check ClearML dashboard for attention patterns
4. **⚡ Experiment**: Try different model architectures
5. **🚀 Advanced**: Explore custom implementations vs PyTorch native

## 📖 Documentation

- **[SETUP.md](SETUP.md)**: Detailed setup and usage guide
- **[Code Documentation](stepwise_transformers/)**: Comprehensive docstrings
- **[Configuration Guide](config/)**: Hydra configuration examples

## 🎨 Model Architecture

The implemented transformer follows the original "Attention Is All You Need" architecture:

### Core Components
- **Multi-Head Attention**: Configurable number of attention heads
- **Positional Encoding**: Both sinusoidal and learned variants
- **Feed-Forward Networks**: Standard and gated linear unit variants
- **Layer Normalization**: Pre-norm and post-norm configurations
- **Residual Connections**: Proper gradient flow optimization

### Educational Features
- **Attention Visualization**: Real-time attention pattern analysis
- **Gradient Monitoring**: Track vanishing/exploding gradient issues
- **Layer Analysis**: Individual layer performance metrics
- **Training Dynamics**: Learning rate scheduling and optimization insights

## 🤝 Contributing

This project follows modern Python development practices:
- **UV** for dependency management
- **Hydra** for configuration
- **Type hints** throughout
- **Rich** for beautiful CLI
- **ClearML** for experiment tracking

## 📄 License

MIT License - feel free to use for learning and research!

---

**Ready to start?** Run `uv run python run_experiment.py` and watch your transformer train with full ClearML integration! 🎉
