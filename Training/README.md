# ELM Training Pipeline

Training pipeline for the ELM (Embedding Language Model) MLP Adapter with specialized summary-only training and Bayesian optimization.

## Overview

This module trains a lightweight MLP adapter (~21M parameters) to map Qwen3-Embedding-4B embeddings (2560-dim) into the token embedding space of Qwen3-4B-Instruct. The LLM remains completely frozen during training.

## 🆕 Summary-Only Training Features

**NEW**: Specialized summary-only training pipeline with:
- **📊 Data Filtering**: Filter summary tasks from multi-task datasets
- **🔍 Bayesian Optimization**: Hyperparameter tuning with Optuna
- **🎯 Text Drift Loss**: Cosine similarity-based semantic fidelity
- **📈 BERTScore Evaluation**: Semantic-aware evaluation metrics
- **⚡ 2-Epoch Training**: Fast optimization cycles
- **📊 Enhanced wandb**: Comprehensive experiment tracking

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

## Quick Start

### Basic Training

```bash
python scripts/train.py \
    --batch-size 16 \
    --grad-accum 2 \
    --epochs 3 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --hidden-dim 4096
```

### Training with W&B Logging

```bash
python scripts/train.py \
    --batch-size 8 \
    --grad-accum 2 \
    --epochs 3 \
    --lr 1e-4 \
    --warmup-steps 1000 \
    --hidden-dim 4096 \
    --wandb \
    --wandb-project elm-training \
    --wandb-run-name my-experiment
```

### Resume from Checkpoint

```bash
python scripts/train.py \
    --resume ../data/checkpoints/checkpoint_step_1000.pt \
    --batch-size 8 \
    --grad-accum 2
```

## 🚀 Summary-Only Training

### 1. Filter Summary Data
```bash
python scripts/filter_summary_data.py \
    --input-dir data/synthesis \
    --input-embeddings-dir data/embeddings \
    --output-dir data/summary_filtered/synthesis \
    --output-embeddings-dir data/summary_filtered/embeddings
```

### 2. Summary Training with BERTScore
```bash
python scripts/train_summary.py \
    --data-dir data/summary_filtered \
    --epochs 2 \
    --batch-size 8 \
    --learning-rate 2e-4 \
    --use-drift-loss \
    --drift-weight 0.03 \
    --wandb \
    --wandb-project elm-summary
```

### 3. Bayesian Optimization
```bash
python scripts/run_bayesian_optimization.py \
    --study-name elm-summary-optimization \
    --trials 50 \
    --timeout 24 \
    --project elm-summary-optimization
```

### Expected Results
- **BERTScore Composite**: 0.85 - 0.92
- **Training Time**: 2-3 hours per trial (2 epochs)
- **Memory Usage**: ~16GB VRAM (batch_size=8)

📖 **Full Documentation**: [Summary Training Guide](docs/summary_training_guide.md) | [wandb Guide](docs/wandb_guide.md)

## Command Line Arguments

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--llm-model` | str | `Qwen/Qwen3-4B-Instruct-2507` | Base LLM model name |
| `--hidden-dim` | int | 4096 | Adapter intermediate hidden dimension |

### Training Hyperparameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | 1e-4 | Learning rate for AdamW optimizer |
| `--warmup-steps` | int | 1000 | Number of warmup steps for learning rate |
| `--weight-decay` | float | 0.01 | Weight decay (L2 regularization) |
| `--max-grad-norm` | float | 1.0 | Maximum gradient norm for clipping |

### Batch Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | 16 | Batch size per device |
| `--grad-accum` | int | 2 | Gradient accumulation steps |

**Effective batch size** = `batch-size` × `grad-accum`

### Training Schedule

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 3 | Number of training epochs |
| `--max-steps` | int | None | Max training steps (overrides epochs) |
| `--eval-steps` | int | 500 | Evaluate every N steps |
| `--save-steps` | int | 1000 | Save checkpoint every N steps |
| `--logging-steps` | int | 50 | Log metrics every N steps |

### Data Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-seq-length` | int | 2048 | Maximum sequence length |
| `--num-workers` | int | 4 | Number of dataloader workers |

### Memory Optimization

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no-bf16` | flag | False | Disable bfloat16 mixed precision |
| `--no-grad-checkpoint` | flag | False | Disable gradient checkpointing |

### Checkpointing & Logging

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--resume` | str | None | Path to checkpoint to resume from |
| `--base-dir` | str | None | Base directory (defaults to project root) |
| `--log-file` | str | None | Path to log file |
| `--seed` | int | 42 | Random seed for reproducibility |

### Weights & Biases

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--wandb` | flag | False | Enable Weights & Biases logging |
| `--wandb-project` | str | `elm-training` | W&B project name |
| `--wandb-run-name` | str | None | W&B run name |

## Optimizer Configuration

The training uses **AdamW optimizer** with the following configuration:

- **Optimizer**: AdamW (decoupled weight decay)
- **Parameters optimized**: Only adapter parameters (~21M)
- **Beta coefficients**: (0.9, 0.999)
- **Epsilon**: 1e-8
- **Weight decay**: 0.01 (configurable via `--weight-decay`)

### Learning Rate Schedule

**Linear warmup + linear decay**:
1. **Warmup phase**: Learning rate increases linearly from 0 to `--lr` over `--warmup-steps`
2. **Decay phase**: Learning rate decreases linearly from `--lr` to 0 over remaining steps

## Hardware Requirements

### Recommended Configuration

- **GPU**: 40GB VRAM (e.g., A100, A6000)
- **Memory usage**: ~16-20GB with `batch-size=16`
- **For 24GB GPUs**: Use `--batch-size 8` with `--grad-accum 2`

### Memory Optimization Features

- **Mixed precision training**: bfloat16 (enabled by default)
- **Gradient checkpointing**: Enabled by default
- **Frozen LLM**: Only adapter parameters require gradients

## Data Requirements

The training script expects the following data files:

```
../data/
├── embeddings/
│   ├── train_embeddings.safetensors
│   └── val_embeddings.safetensors
└── synthesis/
    ├── train_synthesis.jsonl
    └── val_synthesis.jsonl
```

### Data Format

- **Embeddings**: SafeTensors format, shape `[N, 2560]`
- **Synthesis**: JSONL format with fields:
  - `task_type`: Type of task (see supported tasks below)
  - `embedding_file_idx`: Index into embeddings file
  - `query`: Input query text
  - `response`: Expected response text

### Supported Task Types

The training pipeline supports 13 single-text task types:

**Factual Tasks:**
- `keywords` - Extract key concepts
- `category` - Classify content by academic field
- `questions` - Generate questions answerable from content

**Descriptive Tasks:**
- `summary` - Concise summarization
- `describe` - Detailed description
- `explain_beginner` - Beginner-friendly explanation
- `explain_expert` - Expert-level technical explanation
- `related_topics` - Suggest related topics

**Creative Tasks:**
- `characteristics_pos` - List strengths/interesting aspects
- `characteristics_neg` - List limitations/criticisms
- `style_academic` - Rewrite in academic tone
- `style_casual` - Rewrite in casual tone
- `counterfactual` - Imagine alternative applications

**Note:** Pair-based tasks (`compare`, `hypothetical`) have been removed from the training pipeline.

## Output

### Checkpoint Directory Structure

Checkpoints are saved to `../data/checkpoints/`:

```
checkpoints/
├── checkpoint_step_500.pt          # Full training state
├── adapter_step_500.safetensors    # Adapter weights only
├── checkpoint_step_1000.pt
├── adapter_step_1000.safetensors
├── adapter_best.safetensors        # Best model by validation loss
└── checkpoint_final.pt             # Final checkpoint
```

### Checkpoint Contents

- **`adapter_step_*.safetensors`**: Adapter weights only (~21M parameters)
- **`checkpoint_step_*.pt`**: Full training state including:
  - Adapter weights
  - Optimizer state
  - Learning rate scheduler state
  - Training step and epoch
  - Best validation loss
- **`adapter_best.safetensors`**: Best performing model on validation set

## Monitoring Training

### Console Output

The training script provides real-time progress bars with:
- Current loss
- Learning rate
- Global step
- GPU memory usage

### Weights & Biases

Enable W&B logging to track:
- Training loss
- Validation loss
- Learning rate curves
- Model checkpoints
- System metrics (GPU, memory)

### Log Files

Specify a log file to save all training logs:

```bash
python scripts/train.py --log-file training.log
```

## Training Tips

1. **Effective batch size**: Aim for 16-32 for stable training
2. **Learning rate**: 1e-4 works well for most cases; try 5e-5 to 2e-4 if needed
3. **Warmup steps**: Use 5-10% of total training steps
4. **Validation**: Monitor validation loss; stop if it plateaus or increases
5. **Gradient clipping**: Keep at 1.0 to prevent gradient explosion
6. **Mixed precision**: Keep bfloat16 enabled for faster training and lower memory

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch-size` (try 8 or 4)
- Increase `--grad-accum` to maintain effective batch size
- Ensure `--no-bf16` is NOT set (mixed precision saves memory)
- Ensure `--no-grad-checkpoint` is NOT set

### Training Instability

- Reduce learning rate (`--lr 5e-5`)
- Increase warmup steps (`--warmup-steps 2000`)
- Check gradient clipping (`--max-grad-norm 1.0`)

### Slow Training

- Increase `--batch-size` if memory allows
- Reduce `--num-workers` if CPU bottleneck
- Enable W&B async logging for minimal overhead
