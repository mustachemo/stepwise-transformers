# Stepwise Transformers Learning with MLflow

An interactive learning environment for transformer architecture, attention mechanisms, and neural network concepts through hands-on training and real-time MLflow monitoring.

## 🎯 Project Overview

This project provides a comprehensive environment for learning transformer architecture through interactive training sessions with MLflow tracking. Users can explore transformer components, visualize attention mechanisms, build models, and train them in real-time—all with professional monitoring and logging.

## 🏗️ Architecture

### Core Components

- **Interactive Training**: Step-by-step guided learning with MLflow tracking
- **Component Library**: Live transformer component exploration and testing with parameter logging
- **Model Builder**: Interactive interface for assembling transformer models with configuration tracking
- **Training Interface**: Real-time training visualization with MLflow metrics
- **Visualization Panels**: Attention heatmaps, architecture diagrams, and progress charts with artifact logging
- **Training Management**: Save, load, and compare training runs using MLflow

### Technology Stack

- **Training Tracking**: [MLflow](https://mlflow.org/) for training run management and monitoring
- **Styling**: Rich library for text formatting and styling
- **Logging**: Loguru for robust, structured logging
- **Neural Networks**: PyTorch for transformer implementations
- **Data Processing**: NumPy for mathematical operations
- **Configuration**: Dataclasses for structured configuration management

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/stepwise-transformers.git
cd stepwise-transformers

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if using uv)
uv sync
```

### 2. Run Training

```bash
# Run complete training pipeline
python train.py
```

This will:
- Train attention mechanisms (single-head, multi-head, scaled dot-product)
- Train positional encoding (sinusoidal, learned, relative)
- Train transformer blocks (encoder, decoder)
- Log all parameters, metrics, and artifacts to MLflow

### 3. Monitor Training

```bash
# Start MLflow UI for monitoring
mlflow ui
```

Then open http://localhost:5000 in your browser to:
- View real-time training metrics
- Compare different training runs
- Download artifacts and visualizations
- Track parameter changes

## 📊 Training Components

### Attention Mechanisms
```python
from transformer_manager import TransformerManager
from components.attention import AttentionModule

# Initialize
manager = TransformerManager()
attention = AttentionModule(manager)

# Train attention mechanisms
query = torch.randn(4, 10, 64)
key = torch.randn(4, 10, 64)
value = torch.randn(4, 10, 64)

# Single-head attention
single_output = attention.single_head_attention(query, key, value)

# Multi-head attention
multi_output = attention.multi_head_attention(8, 64, query, key, value)

# Scaled dot-product attention
scaled_output = attention.scaled_dot_product_attention(query, key, value)
```

### Positional Encoding
```python
from components.positional_encoding import PositionalEncodingModule

pos_encoding = PositionalEncodingModule(manager)

# Sinusoidal encoding
pe_sinusoidal = pos_encoding.sinusoidal_encoding(20, 128)

# Learned encoding
pe_learned = pos_encoding.learned_encoding(20, 128)

# Relative encoding
pe_relative = pos_encoding.relative_encoding(20, 128, 32)
```

### Transformer Blocks
```python
from components.transformer_block import TransformerBlockModule

block = TransformerBlockModule(manager)

# Encoder block
input_tensor = torch.randn(2, 10, 64)
encoder_output = block.encoder_block(input_tensor, 64, 8, 256, 0.1)

# Decoder block
decoder_output = block.decoder_block(input_tensor, encoder_output, 64, 8, 256, 0.1)
```

## 📁 Project Structure

```
stepwise-transformers/
├── components/              # Transformer component implementations
│   ├── attention.py        # Attention mechanism components
│   ├── positional_encoding.py  # Positional encoding components
│   ├── transformer_block.py    # Transformer block components
│   └── base.py             # Base component class
├── transformer_manager.py   # Main training manager with MLflow
├── train.py                # Complete training pipeline
├── main.py                 # Application entry point
├── models/                 # Saved transformer models
├── checkpoints/            # Training checkpoints
├── artifacts/              # MLflow artifacts (visualizations, etc.)
├── logs/                   # Application logs
├── data/                   # Training datasets
└── mlruns/                 # MLflow tracking data
```

## 🎨 Features

### Interactive Learning
- **Step-by-step training** with MLflow tracking
- **Real-time visualization** of attention mechanisms with artifact logging
- **Interactive model building** with training run tracking
- **Live training monitoring** with MLflow metrics

### Visualization Capabilities
- **Attention weight heatmaps** for understanding attention patterns
- **Model architecture diagrams** showing component relationships
- **Training progress charts** with loss and accuracy curves
- **Gradient flow visualization** for debugging

### Training Management
- **Save and load training runs** for reproducible research
- **Compare different model configurations** side by side
- **Export results** for further analysis
- **Collaborative features** for sharing training runs

## 📋 Training Pipeline

### Phase 1: Attention Training
- Single-head attention computation
- Multi-head attention with parameter tracking
- Scaled dot-product attention with masking
- Real-time metric logging

### Phase 2: Positional Encoding Training
- Sinusoidal positional encoding
- Learned positional encoding
- Relative positional encoding
- Encoding quality metrics

### Phase 3: Transformer Block Training
- Encoder block with residual connections
- Decoder block with cross-attention
- Layer normalization and feed-forward networks
- Block performance metrics

## 🧪 Development Standards

This project follows strict Python coding standards:

- **Type Hints**: All functions include comprehensive type annotations
- **Documentation**: Google-style docstrings with Args, Returns, Raises sections
- **Error Handling**: Specific exception handling with proper logging
- **Logging**: Structured logging with Loguru for debugging
- **Code Style**: PEP 8 compliance with 4-space indentation
- **Testing**: Comprehensive unit tests for all components
- **Training Tracking**: MLflow for training run management and versioning

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📚 Learning Resources

- **Papers**: "Attention Is All You Need", "BERT", "GPT" papers
- **Books**: "Deep Learning" by Goodfellow, Bengio, Courville
- **Online Courses**: CS224N (Stanford), CS182 (Berkeley)
- **Training Tracking**: MLflow documentation and tutorials
- **Implementations**: PyTorch tutorials, Hugging Face transformers

## 🎯 Success Metrics

- [ ] Can run interactive training for transformer learning
- [ ] Can implement transformer components with training tracking
- [ ] Can create educational training sessions with MLflow
- [ ] Can provide real-time model training and evaluation tracking
- [ ] Can enable hands-on exploration of transformer concepts through training
- [ ] Can develop custom training components for transformer visualization
- [ ] Can create engaging educational content through MLflow training

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MLflow](https://mlflow.org/) for the amazing training tracking framework
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- [PyTorch](https://pytorch.org/) for neural network capabilities
- The transformer research community for foundational work

---

**Ready to dive into transformers? Start your learning journey with our interactive training pipeline! 🚀**
