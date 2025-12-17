# ELM Summary-Only Training Guide

This guide provides comprehensive instructions for training the ELM model specifically on summary tasks using the new summary-only pipeline with Bayesian optimization and text drift loss.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Bayesian Optimization](#bayesian-optimization)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

## Overview

The ELM Summary-Only Training Pipeline is specifically designed for:

1. **Summary-Only Training**: Filters and trains exclusively on summary tasks
2. **Bayesian Optimization**: Hyperparameter tuning with Optuna
3. **Text Drift Loss**: Cosine similarity-based semantic fidelity
4. **BERTScore Evaluation**: Semantic-aware evaluation replacing ROUGE
5. **2-Epoch Constraint**: Fast optimization cycles
6. **Enhanced wandb Tracking**: Comprehensive experiment tracking

### Key Features
- ✅ Summary-only data filtering and processing
- ✅ Text drift loss for semantic fidelity
- ✅ BERTScore-based evaluation
- ✅ Bayesian hyperparameter optimization
- ✅ 2-epoch fast training cycles
- ✅ Comprehensive wandb integration

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (40GB recommended)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free disk space
- **Python**: 3.9+ with CUDA support

### Software Dependencies
```bash
# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install safetensors>=0.3.0

# Optimization dependencies
pip install optuna>=3.0.0
pip install evaluate>=0.4.0
pip install bert-score>=0.3.13

# Tracking and evaluation
pip install wandb>=0.16.0
pip install tqdm>=4.65.0
pip install numpy>=1.24.0
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd elm

# Install in development mode
pip install -e .

# Install summary-specific dependencies
pip install -e ".[summary]"
```

## Data Preparation

### 1. Filter Summary Data
Extract summary-only data from existing multi-task datasets:

```bash
# Filter summary data from existing datasets
python scripts/filter_summary_data.py \
    --input-dir data/synthesis \
    --input-embeddings-dir data/embeddings \
    --output-dir data/summary_filtered/synthesis \
    --output-embeddings-dir data/summary_filtered/embeddings
```

This will:
- Filter for `task_type == "summary"` entries
- Create new embedding index mappings
- Generate filtered SafeTensors embeddings
- Provide train/val/test statistics

### 2. Expected Data Structure
```
data/summary_filtered/
├── synthesis/
│   ├── train_synthesis.jsonl
│   ├── val_synthesis.jsonl
│   └── test_synthesis.jsonl
└── embeddings/
    ├── train_embeddings.safetensors
    ├── val_embeddings.safetensors
    └── test_embeddings.safetensors
```

### 3. Data Format
Each JSONL line should contain:
```json
{
    "task_type": "summary",
    "input_prompt_template": "Summarize the following text:\n\n{text}\n\nSummary:",
    "embedding_index": 0,
    "target_text": "Generated summary text...",
    "variation": 0,
    "temperature": 0.3,
    "top_p": 0.85,
    "token_count": 128
}
```

## Configuration

### Basic Configuration
The summary training uses `SummaryTrainingConfig` with key settings:

```python
from summary_training_pipeline.config import SummaryTrainingConfig

config = SummaryTrainingConfig(
    # Model settings
    llm_model_name="Qwen/Qwen3-4B-Instruct",
    embedding_dim=2560,
    hidden_dim=4096,

    # Summary-specific
    summary_only=True,
    summary_data_path="data/summary_filtered",
    max_epochs=2,  # Fixed for optimization

    # Text drift loss
    use_text_drift_loss=True,
    text_drift_weight=0.03,
    text_drift_target_similarity=0.75,

    # BERTScore evaluation
    bertscore_model="microsoft/deberta-xlarge-mnli",
    bertscore_batch_size=16,
)
```

### Key Configuration Parameters

#### Model Architecture
- `embedding_dim`: 2560 (Qwen3-Embedding-4B)
- `hidden_dim`: [2048, 4096, 6144] (tuned by BO)
- `dropout_rate`: 0.0 - 0.3 (tuned by BO)
- `residual_scale`: 0.05 - 0.5 (tuned by BO)

#### Training Hyperparameters
- `learning_rate`: 1e-5 to 5e-3 (log scale, tuned by BO)
- `batch_size`: [4, 8, 16] (tuned by BO)
- `gradient_accumulation_steps`: [2, 4, 8] (tuned by BO)
- `max_epochs`: 2 (fixed for optimization)

#### Loss Configuration
- `use_contrastive_loss`: True (enabled)
- `contrastive_weight`: 0.0 - 0.1 (tuned by BO)
- `use_text_drift_loss`: True (enabled for summaries)
- `text_drift_weight`: 0.01 - 0.1 (tuned by BO)
- `text_drift_target_similarity`: 0.7 - 0.9 (tuned by BO)

## Training

### 1. Basic Summary Training
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

### 2. Advanced Training with Custom Parameters
```bash
python scripts/train_summary.py \
    --data-dir data/summary_filtered \
    --llm-model Qwen/Qwen3-4B-Instruct \
    --hidden-dim 4096 \
    --epochs 2 \
    --learning-rate 1e-4 \
    --batch-size 8 \
    --contrastive-weight 0.01 \
    --use-drift-loss \
    --drift-weight 0.05 \
    --drift-target-similarity 0.8 \
    --residual-scale 0.1 \
    --dropout 0.1 \
    --bertscore-model microsoft/deberta-xlarge-mnli \
    --early-stopping-patience 2 \
    --eval-interval 250 \
    --wandb \
    --wandb-project elm-summary \
    --wandb-run-name "summary-experiment-001"
```

### 3. Training with Checkpoint Resumption
```bash
python scripts/train_summary.py \
    --data-dir data/summary_filtered \
    --epochs 2 \
    --resume-from data/checkpoints/adapter_step_1000.safetensors \
    --wandb \
    --wandb-project elm-summary
```

### Training Monitoring

#### Key Metrics to Watch
- **Training Loss**: Should decrease steadily
- **BERTScore Composite**: Target > 0.85
- **Drift Similarity**: Target 0.7-0.9
- **Validation BERTScore**: Should improve over epochs

#### Early Stopping
Training will stop automatically if:
- No improvement in BERTScore for `early_stopping_patience` epochs
- Training becomes unstable (loss spikes)
- Maximum epochs reached (2 for optimization)

## Bayesian Optimization

### 1. Run Full Optimization
```bash
python scripts/run_bayesian_optimization.py \
    --study-name elm-summary-optimization \
    --project elm-summary-optimization \
    --trials 50 \
    --timeout 24 \
    --data-dir data/summary_filtered \
    --checkpoint-dir data/optuna_checkpoints \
    --results-dir data/optimization_results \
    --eval-samples 100
```

### 2. Resume Existing Optimization
```bash
python scripts/run_bayesian_optimization.py \
    --study-name elm-summary-optimization \
    --storage sqlite:///optimization.db \
    --resume \
    --trials 100 \
    --timeout 48
```

### 3. Parallel Optimization
```bash
# Run multiple parallel optimizers
python scripts/run_bayesian_optimization.py \
    --study-name elm-summary-optimization \
    --storage sqlite:///optimization.db \
    --jobs 4 \
    --trials 200
```

### Optimization Parameters

#### Search Space
- **Learning Rate**: 1e-5 to 5e-3 (log scale)
- **Hidden Dimension**: [2048, 4096, 6144]
- **Dropout Rate**: 0.0 to 0.3
- **Residual Scale**: 0.05 to 0.5
- **Contrastive Weight**: 0.0 to 0.1
- **Text Drift Weight**: 0.01 to 0.1
- **Text Drift Target Similarity**: 0.7 to 0.9
- **Batch Size**: [4, 8, 16]

#### Optimization Strategy
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Pruning**: Median pruning with 2-epoch warmup
- **Objective**: Maximize BERTScore composite
- **Constraint**: Fixed 2 epochs per trial

### Expected Optimization Results

#### Performance Targets
- **BERTScore Composite**: 0.85 - 0.92
- **BERTScore F1**: 0.88 - 0.94
- **Training Time**: 2-3 hours per trial (2 epochs)
- **Best Parameters**: Usually found in 20-30 trials

#### Typical Best Configuration
```json
{
    "learning_rate": 1.5e-4,
    "hidden_dim": 4096,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "dropout_rate": 0.12,
    "residual_scale": 0.15,
    "contrastive_weight": 0.008,
    "text_drift_weight": 0.045,
    "text_drift_target_similarity": 0.78
}
```

## Evaluation

### 1. BERTScore Evaluation
```python
from optimization.bertscore_metrics import create_evaluator

evaluator = create_evaluator(config)
metrics = evaluator.evaluate_batch(predictions, references)

print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
print(f"BERTScore Composite: {metrics['bertscore_composite']:.4f}")
```

### 2. Manual Evaluation
```bash
# Generate predictions
python scripts/generate_summaries.py \
    --checkpoint data/checkpoints/best_summary_model.safetensors \
    --input data/summary_filtered/test_synthesis.jsonl \
    --output results/predictions.jsonl

# Evaluate with BERTScore
python scripts/evaluate_bertscore.py \
    --predictions results/predictions.jsonl \
    --references data/summary_filtered/test_synthesis.jsonl
```

### 3. Quality Analysis
```python
# Analyze generation quality
from optimization.bertscore_metrics import BERTScoreEvaluator

evaluator = BERTScoreEvaluator()
analysis = evaluator.evaluate_batch(predictions, references, return_detailed=True)

print(f"Average length ratio: {analysis['length_ratio']:.2f}")
print(f"Empty prediction rate: {analysis['empty_prediction_rate']:.3f}")
print(f"Unique token ratio: {analysis['avg_unique_token_ratio']:.3f}")
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train_summary.py --batch-size 4

# Enable gradient checkpointing
python scripts/train_summary.py --gradient-checkpointing

# Use mixed precision
python scripts/train_summary.py --use-bf16
```

#### 2. Poor BERTScore Performance
```bash
# Increase text drift weight
python scripts/train_summary.py --drift-weight 0.08

# Adjust learning rate
python scripts/train_summary.py --learning-rate 1e-4

# Enable contrastive loss
python scripts/train_summary.py --use-contrastive --contrastive-weight 0.02
```

#### 3. Slow Training
```bash
# Increase batch size if memory allows
python scripts/train_summary.py --batch-size 16

# Reduce evaluation frequency
python scripts/train_summary.py --eval-interval 500

# Use fewer evaluation samples
python scripts/run_bayesian_optimization.py --eval-samples 50
```

#### 4. wandb Sync Issues
```bash
# Check wandb status
wandb status

# Re-authenticate
wandb logout && wandb login

# Use offline mode
WANDB_MODE=offline python scripts/train_summary.py
```

### Performance Optimization

#### Memory Optimization
```python
config = SummaryTrainingConfig(
    batch_size=4,                    # Reduce batch size
    gradient_accumulation_steps=8,   # Increase accumulation
    use_bf16=True,                  # Enable mixed precision
    use_gradient_checkpointing=True, # Enable gradient checkpointing
    max_seq_length=1024,            # Reduce sequence length
)
```

#### Speed Optimization
```python
config = SummaryTrainingConfig(
    num_workers=8,                  # Increase data loading workers
    pin_memory=True,                # Enable pin memory
    persistent_workers=True,        # Keep workers alive
    prefetch_factor=4,              # Prefetch more batches
)
```

### Debug Mode
```bash
# Enable debug logging
python scripts/train_summary.py --log-level DEBUG

# Run on small dataset
python scripts/train_summary.py --data-dir data/debug_summary --epochs 1

# Disable wandb for faster iteration
python scripts/train_summary.py --no-wandb
```

## Best Practices

### 1. Hyperparameter Selection
- Start with recommended default values
- Use Bayesian optimization for final tuning
- Monitor both BERTScore and training stability
- Consider compute budget vs. performance gains

### 2. Evaluation Strategy
- Use BERTScore composite as primary metric
- Monitor individual precision/recall/f1 scores
- Check for generation quality issues (repetition, length)
- Validate on held-out test set

### 3. Production Deployment
- Use best configuration from optimization
- Validate on diverse test sets
- Monitor inference latency and memory usage
- Implement A/B testing for model updates

This guide should help you successfully train and optimize ELM models for summary generation tasks!