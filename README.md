# Stepwise Transformers Learning with ClearML

A comprehensive implementation of transformer architecture with standard attention mechanisms, integrated with ClearML for advanced experiment tracking, visualization, and model management.

## 🎯 Project Overview

This project provides a clean, educational implementation of the transformer architecture from "Attention Is All You Need" with comprehensive ClearML integration. Learn transformer concepts through hands-on implementation while leveraging professional-grade experiment tracking, attention visualization, and model registry capabilities.

## 🏗️ Architecture

### Core Components

- **Transformer Architecture**: Complete implementation with encoder/decoder stacks
- **Standard Attention**: Scaled dot-product attention with multi-head mechanism
- **Positional Encoding**: Sinusoidal and learned positional encoding options
- **Feed-Forward Networks**: Position-wise feed-forward with configurable activations
- **ClearML Integration**: Comprehensive experiment tracking and visualization
- **Model Registry**: Automated model versioning and checkpoint management

### Technology Stack

- **Experiment Tracking**: [ClearML](https://clear.ml/) for professional experiment management
- **Deep Learning**: PyTorch for transformer implementations
- **Visualization**: Matplotlib and Seaborn for attention heatmaps and training curves
- **Data Processing**: NumPy for mathematical operations and data handling
- **Logging**: Structured logging with gradient tracking and metric visualization
- **Configuration**: JSON-based configuration management

## 🏃 Quick Start

### Installation and Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stepwise-transformers.git
cd stepwise-transformers
   ```

2. **Create virtual environment**
   ```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
pip install -r requirements.txt
```

4. **Set up ClearML** (optional - for experiment tracking)
   ```bash
   clearml-init
   ```

### Training Your First Transformer

**Basic training with default configuration:**
```bash
python -m src.main train
```

**Training with custom configuration:**
```bash
python -m src.main train --config config/default_config.json
```

**Training with custom ClearML project:**
```bash
python -m src.main train --project-name "my-transformers" --task-name "experiment-1"
```

## 📊 Features

### Transformer Implementation
- ✅ **Complete Architecture**: Full encoder-decoder transformer
- ✅ **Standard Attention**: Scaled dot-product with multi-head support
- ✅ **Positional Encoding**: Both sinusoidal and learned variants
- ✅ **Layer Components**: LayerNorm, residual connections, feed-forward networks
- ✅ **Configurable**: Flexible model sizing and hyperparameters

### ClearML Integration
- ✅ **Experiment Tracking**: Automatic logging of hyperparameters and metrics
- ✅ **Attention Visualization**: Real-time attention heatmap generation
- ✅ **Model Registry**: Automatic model versioning and checkpoint management
- ✅ **Training Curves**: Loss and accuracy visualization over time
- ✅ **Gradient Monitoring**: Gradient norm tracking for debugging
- ✅ **Resource Tracking**: GPU and memory utilization monitoring

## 📂 Project Structure

```
stepwise-transformers/
├── src/                           # Source code
│   ├── components/               # Transformer components
│   │   ├── attention.py         # Multi-head attention implementation
│   │   ├── positional_encoding.py # Positional encoding variants
│   │   └── feed_forward.py      # Feed-forward networks
│   ├── models/                  # Complete model implementations
│   │   ├── transformer_layers.py # Encoder/decoder layer implementations
│   │   └── transformer.py       # Full transformer model
│   ├── utils/                   # Utilities and helpers
│   │   ├── clearml_utils.py     # ClearML integration utilities
│   │   └── data_utils.py        # Data processing and tokenization
│   ├── train.py                 # Training script with ClearML integration
│   └── main.py                  # CLI entry point
├── config/                      # Configuration files
│   └── default_config.json     # Default training configuration
├── .cursor/rules/              # Development rules
│   └── clearml-experiment-guide.md # ClearML usage guidelines
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

### Key Commands

- `python -m src.main train` - Start transformer training
- `python -m src.main train --config config/custom.json` - Train with custom config
- `clearml-agent init` - Set up ClearML agent for remote execution
- `clearml-data upload` - Upload datasets to ClearML data management

## 🎨 Model Architecture

The implemented transformer follows the original "Attention Is All You Need" architecture:

### Encoder Stack
- **Multi-Head Self-Attention**: 8 attention heads with scaled dot-product attention
- **Position-wise Feed-Forward**: Two linear transformations with ReLU activation
- **Residual Connections**: Around each sub-layer with layer normalization
- **Positional Encoding**: Sinusoidal encoding added to input embeddings

### Decoder Stack
- **Masked Multi-Head Self-Attention**: Prevents attending to future positions
- **Encoder-Decoder Attention**: Cross-attention between decoder and encoder outputs
- **Position-wise Feed-Forward**: Same as encoder with residual connections
- **Output Projection**: Linear layer to vocabulary size with softmax

### Key Features
- **Configurable Model Size**: Adjustable d_model, heads, layers, and feed-forward dimensions
- **Dropout Regularization**: Applied throughout the model for better generalization
- **Label Smoothing**: Configurable smoothing factor for more robust training
- **Gradient Clipping**: Prevents exploding gradients during training

## 🧪 Development Standards

This project follows strict Python coding standards:

- **Type Hints**: All functions include comprehensive type annotations
- **Documentation**: Google-style docstrings with Args, Returns, Raises sections
- **Error Handling**: Specific exception handling with proper logging
- **Logging**: Structured logging with Loguru for debugging
- **Code Style**: PEP 8 compliance with 4-space indentation
- **Testing**: Comprehensive unit tests for all components
- **Experiment Tracking**: ClearML for professional experiment management and versioning

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📚 Learning Resources

- **Papers**: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - The original transformer paper
- **Books**: "Deep Learning" by Goodfellow, Bengio, Courville
- **Online Courses**: CS224N (Stanford), CS182 (Berkeley)
- **Experiment Tracking**: [ClearML Documentation](https://clear.ml/docs/) and tutorials
- **Implementations**: PyTorch tutorials, Hugging Face transformers library

## 🎯 What You'll Learn

- ✅ **Transformer Architecture**: Complete understanding of encoder-decoder structure
- ✅ **Attention Mechanisms**: How scaled dot-product attention works in practice
- ✅ **Training Dynamics**: Proper initialization, optimization, and regularization
- ✅ **Experiment Tracking**: Professional ML experiment management with ClearML
- ✅ **Model Development**: From implementation to deployment best practices
- ✅ **Visualization**: Understanding model behavior through attention heatmaps

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ClearML](https://clear.ml/) for the comprehensive ML operations platform
- [PyTorch](https://pytorch.org/) for the excellent deep learning framework
- Vaswani et al. for the groundbreaking "Attention Is All You Need" paper
- The transformer research community for continuous innovation

---

**Ready to master transformers? Start building and training with professional experiment tracking! 🚀**
