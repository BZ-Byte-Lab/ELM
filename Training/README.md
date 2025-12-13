# ELM Training Pipeline

Training pipeline for the ELM (Embedding Language Model) MLP Adapter.

## Overview

This module trains a lightweight MLP adapter (~21M parameters) to map Qwen3-Embedding-4B embeddings (2560-dim) into the token embedding space of Qwen3-4B-Instruct. The LLM remains completely frozen during training.

## Architecture

- **E_0**: Frozen token embeddings from Qwen3-4B-Instruct
- **E_A**: Trainable MLP adapter (2560 → 4096 → 2560 with residual)
- **M_0**: Frozen transformer layers from Qwen3-4B-Instruct

## Setup

```bash
cd Training
conda env create -f environment.yml
conda activate elm-training
```

## Training

```bash
python scripts/train.py \
    --batch-size 16 \
    --grad-accum 2 \
    --epochs 3 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --hidden-dim 4096
```

## Hardware Requirements

- 40GB VRAM GPU (e.g., A100)
- Memory usage: ~16-20 GB with batch_size=16

## Data Requirements

- `../data/embeddings/train_embeddings.safetensors`
- `../data/embeddings/val_embeddings.safetensors`
- `../data/synthesis/train_synthesis.jsonl`
- `../data/synthesis/val_synthesis.jsonl`

## Output

Checkpoints are saved to `../data/checkpoints/`:
- `adapter_step_*.safetensors` - Adapter weights
- `checkpoint_step_*.pt` - Full training state (optimizer, scheduler)
- `adapter_best.safetensors` - Best model by validation loss
