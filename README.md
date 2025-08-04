# zkLoRA: Zero-Knowledge Proofs for LoRA Fine-tuning

A CUDA-accelerated implementation of zero-knowledge proofs for Low-Rank Adaptation (LoRA) fine-tuning of large language models. This project enables verifiable and privacy-preserving fine-tuning of Transformer models while maintaining computational efficiency.

## 🚀 Overview

zkLoRA is the first framework to provide end-to-end zero-knowledge verifiability for LoRA fine-tuning of large-scale language models. It ensures that all computational steps in the fine-tuning process are provably correct without revealing sensitive information such as model parameters or training data.

### Key Features

- **Zero-Knowledge Verifiable Fine-tuning**: Prove correctness of LoRA fine-tuning without revealing model weights or training data
- **GPU-Accelerated**: CUDA-optimized implementation for efficient proof generation
- **Transformer Support**: Handles complex operations including self-attention, feed-forward networks, and normalization layers
- **Cryptographic Security**: Built on BLS12-381 elliptic curves with polynomial commitments
- **Scalable**: Designed to work with large language models like LLaMA

## 🏗️ Architecture

The framework consists of several key components:

### Core Cryptographic Modules
- **BLS12-381 Operations** (`bls12-381.cu`): Elliptic curve arithmetic and pairing operations
- **Polynomial Commitments** (`polynomial.cu`, `commitment.cu`): Cryptographic commitments for model parameters
- **Lookup Arguments** (`tlookup.cu`): Efficient verification of non-arithmetic operations

### Neural Network Components
- **Self-Attention** (`self-attn.cu`): Zero-knowledge proofs for attention mechanisms
- **Feed-Forward Networks** (`ffn.cu`): Verifiable computation of MLP layers
- **Activation Functions** (`zkrelu.cu`, `zksoftmax.cu`): ZK-friendly implementations of ReLU and Softmax
- **Normalization** (`rmsnorm.cu`): RMSNorm layer verification

### Tensor Operations
- **Field Element Tensors** (`fr-tensor.cu`): Operations on finite field elements
- **Group Element Tensors** (`g1-tensor.cu`): Elliptic curve point operations

## 📋 Requirements

### Hardware
- NVIDIA GPU with CUDA capability 7.0 or higher
- GPU Memory: 80GB

### Software
- CUDA Toolkit 11.0+
- GCC 7.3+ or compatible C++ compiler
- Python 3.8+


## 📖 Documentation

### Core Concepts

- **LoRA Fine-tuning**: Adapts pre-trained models by learning low-rank matrices
- **Zero-Knowledge Proofs**: Proves correctness without revealing private information
- **Polynomial Commitments**: Cryptographic primitive for committing to polynomials
- **Lookup Arguments**: Efficient method for verifying table lookups in ZK


## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
make test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](License) file for details.


## 🙏 Acknowledgments

- Built on top of existing zero-knowledge proof libraries
- Inspired by advances in verifiable machine learning
- Special thanks to the open-source cryptography community

## 🔗 Related Work

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [zkLLM: Zero Knowledge Proofs for Large Language Models](https://arxiv.org/abs/2404.16109)
- [ZKML: Zero-Knowledge Machine Learning](https://github.com/zkml-community)

---

**Note**: This is an experimental research project. Use in production environments is not recommended without thorough security auditing.
