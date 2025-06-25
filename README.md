# Vision Transformer (ViT) Implementation

A PyTorch implementation of Vision Transformer (ViT) from scratch. This repository contains a clean, educational implementation of the attention mechanism and transformer architecture for computer vision tasks.

## Features

- Custom multi-head attention implementation
- Efficient attention computation with 3x speedup
- Educational code with clear variable names and comments
- Modular design for easy understanding

## Architecture Overview

This implementation includes all the core components of a Vision Transformer:

- **Patch Embedding**: Converts input images into sequences of patch embeddings
- **Positional Encoding**: Adds learnable positional information to patches
- **Classification Token**: Special token for aggregating global image information
- **Multi-Head Attention**: Both conceptual and optimized implementations
- **Feed-Forward Networks**: MLP blocks with GELU activation
- **Layer Normalization**: Custom implementation for better understanding
- **Classification Head**: Final layer for class predictions

## Model Components

### Core Classes

- `ViT`: Main Vision Transformer model
- `VisionEncoder`: Transformer encoder with multiple layers
- `EfficientMultiHeadedAttention`: Optimized attention mechanism
- `MultiHeadedAttention`: Conceptual multi-head attention for learning
- `Attention`: Single attention head implementation
- `MLP`: Feed-forward network with layer normalization
- `LayerNormalization`: Custom layer normalization
- `ClassificationHead`: Final classification layer
- `Solver`: Implements training and validation logic

### Key Features

**Efficient Attention**: The `EfficientMultiHeadedAttention` class provides approximately 3x speedup over the conceptual implementation by computing all attention heads in parallel rather than sequentially.

**Educational Design**: The code includes both conceptual (`MultiHeadedAttention`) and efficient implementations to help understand the attention mechanism evolution.

**Patch Processing**: Sophisticated patch embedding that reshapes input images into sequences while preserving spatial relationships.

## Usage

### Basic Usage

- Modify main.py file

### Model Parameters
>> These are passed in `Solver' class in `main.py`.

- `imageNet_path` 
- `image_size=224`
- `patch_size=16` 
- `embed_dim=768` 
- `mlp_dim=3072` 
- `num_classes=1000` 
- `num_heads=12`
- `epochs=300` 
- `dropout=0.1`
- `num_steps=10000`
- `weight_decay=0.1`
- `warmup_steps=500`
- `learning_rate=3e-2`
- `decay_type="cosine"`

## Architecture Details

### Patch Embedding Process

1. Input image `[B, C, H, W]` is divided into patches
2. Each patch is flattened and projected to embedding dimension
3. Patches are arranged as a sequence `[B, num_patches, embed_dim]`

### Attention Mechanism

The implementation provides two attention variants:

**Conceptual Multi-Head Attention**: Creates separate attention modules for each head, helpful for understanding the mechanism.

**Efficient Multi-Head Attention**: Computes all heads in parallel using matrix operations, providing significant speedup.

### Position Encoding

Learnable positional embeddings are added to patch embeddings to provide spatial information to the transformer.

## Performance

The efficient attention implementation provides approximately **3x speedup** compared to the conceptual version while maintaining identical functionality.

## Installation

```bash
git clone https://github.com/abdullahejazjanjua/vision_transformer
cd vision-transformer
pip install -r requirements.txt
```

## Training on ImageNet

- Modify parameters passed to the Solver class to implement any transformer

```bash
python main.py
```

## Educational Value

This implementation is designed for learning and includes:

- Clear variable names and extensive comments
- Step-by-step tensor shape transformations
- Both conceptual and optimized implementations
- Modular design for easy experimentation

## Paper Reference

Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.

```bibtex
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```