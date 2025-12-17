# Weights & Biases (wandb) Usage Guide for ELM Summary Training

This guide explains how to use Weights & Biases for tracking ELM summary training experiments and Bayesian optimization progress.

## Table of Contents
- [Account Setup](#account-setup)
- [API Key Configuration](#api-key-configuration)
- [Monitoring Training Progress](#monitoring-training-progress)
- [Tracking Bayesian Optimization](#tracking-bayesian-optimization)
- [Analyzing Hyperparameter Importance](#analyzing-hyperparameter-importance)
- [Comparing Trials and Results](#comparing-trials-and-results)
- [Exporting Results](#exporting-results)

## Account Setup

### 1. Create a wandb Account
1. Go to [https://wandb.ai](https://wandb.ai)
2. Sign up for a free account
3. Verify your email address

### 2. Install wandb
```bash
pip install wandb>=0.16.0
```

### 3. Login to wandb
```bash
wandb login
```
Follow the prompts to authenticate with your account.

## API Key Configuration

### Method 1: Environment Variable (Recommended)
```bash
export WANDB_API_KEY="your-api-key-here"
```

### Method 2: Configuration File
Create `~/.netrc` file:
```
machine api.wandb.ai
    login user
    password your-api-key-here
```

### Method 3: Interactive Login
```bash
wandb login
```
This will prompt you to enter your API key.

## Monitoring Training Progress

### Single Training Run
```bash
python scripts/train_summary.py \
    --data-dir data/summary_filtered \
    --epochs 2 \
    --wandb \
    --wandb-project elm-summary \
    --wandb-run-name "experiment-001"
```

### Key Metrics to Monitor

#### Training Metrics
- `train/lm_loss`: Language modeling loss
- `train/drift_loss`: Text drift loss (if enabled)
- `train/drift_similarity`: Cosine similarity between embeddings
- `train/total_loss`: Combined training loss

#### Validation Metrics
- `eval/bertscore_precision`: BERTScore precision
- `eval/bertscore_recall`: BERTScore recall
- `eval/bertscore_f1`: BERTScore F1 score
- `eval/bertscore_composite`: Composite BERTScore metric

#### Hyperparameters
- `params/learning_rate`: Learning rate
- `params/text_drift_weight`: Drift loss weight
- `params/batch_size`: Batch size
- `params/hidden_dim`: Hidden dimension

### wandb Dashboard Features

1. **Charts Tab**: View metric trends over time
2. **Panels**: Customize your dashboard with:
   - Line plots for loss curves
   - Scatter plots for parameter relationships
   - Histograms for metric distributions
   - Parallel coordinates for multi-parameter analysis

3. **System Metrics**: Monitor GPU usage, memory, and training speed

## Tracking Bayesian Optimization

### Start Optimization
```bash
python scripts/run_bayesian_optimization.py \
    --project elm-summary-optimization \
    --trials 50 \
    --timeout 24 \
    --wandb-project elm-summary-optimization
```

### Optimization Progress Tracking

#### Trial Status
- `trial_status`: "started", "completed", "pruned", or "failed"
- `active_trials`: Number of currently running trials
- `completed_trials`: Total completed trials
- `pruned_trials`: Total pruned trials

#### Trial Metrics
For each trial, wandb tracks:
- `trial_N/metrics/objective`: Primary optimization objective
- `trial_N/metrics/bertscore_composite`: BERTScore composite
- `trial_N/time/training_minutes`: Training time
- `trial_N/params/*`: Trial hyperparameters

#### Optimization Progress
- `optimization/hours_spent`: Total optimization time
- `optimization/total_trials`: Total trials attempted
- `optimization/completed_trials`: Successfully completed trials
- `optimization/pruning_rate`: Percentage of trials pruned

## Analyzing Hyperparameter Importance

### Automatic Importance Analysis
wandb automatically generates hyperparameter importance charts showing:

1. **Bar Chart**: Visual ranking of parameter importance
2. **Correlation Matrix**: Parameter correlations
3. **Parallel Coordinates**: Multi-dimensional parameter relationships

### Key Parameters for Summary Training
The most important parameters typically include:
- `text_drift_weight`: Drift loss regularization strength
- `learning_rate`: Learning rate for adapter training
- `contrastive_weight`: Contrastive loss weight
- `hidden_dim`: Adapter hidden dimension
- `dropout_rate`: Dropout regularization

### Example Insights
- High `text_drift_weight` (0.05-0.1) improves factual consistency
- Learning rates between 1e-4 and 3e-4 work best
- Smaller batch sizes (4-8) with gradient accumulation perform well
- Hidden dimensions of 4096 provide good balance of performance/efficiency

## Comparing Trials and Results

### Side-by-Side Comparison
1. Go to your project dashboard
2. Select multiple runs using checkboxes
3. Click "Compare" to see:
   - Parameter differences
   - Metric comparisons
   - Performance correlations

### Table View
- Click "Table" tab to see all runs in a spreadsheet format
- Add/remove columns to focus on key metrics
- Sort by performance metrics
- Filter by parameter ranges

### Best Run Identification
Look for runs with:
- Highest `bertscore_composite` (>0.85)
- Stable training curves (no sudden drops)
- Reasonable training times (<3 hours per trial)
- Good parameter balance (not extreme values)

## Exporting Results

### Export to CSV
```python
import wandb
api = wandb.Api()

# Get project runs
runs = api.runs("your-username/elm-summary-optimization")

# Export to DataFrame
import pandas as pd
data = []
for run in runs:
    data.append({
        'run_id': run.id,
        'bertscore_composite': run.summary.get('summary/best_value', 0),
        'learning_rate': run.config.get('learning_rate', 0),
        'text_drift_weight': run.config.get('text_drift_weight', 0),
        # Add other parameters as needed
    })

df = pd.DataFrame(data)
df.to_csv('optimization_results.csv', index=False)
```

### Export Best Configuration
```bash
# Download best config artifact from wandb
wandb artifact get best_config:latest
```

### Generate Reports
```python
# Create optimization report
import wandb
run = wandb.init(project="elm-summary-report")

# Log analysis plots
wandb.log({
    'optimization_summary': create_summary_plot(optimization_data),
    'parameter_importance': create_importance_plot(importance_data),
    'best_configuration': best_config
})

wandb.finish()
```

## Advanced Features

### Custom Metrics
Add custom metrics to your training:

```python
# In your training loop
wandb.log({
    'custom/factual_consistency': calculate_factual_score(predictions, references),
    'custom/fluency_score': calculate_fluency_score(predictions),
    'custom/compression_ratio': calculate_compression_ratio(predictions, references)
})
```

### Alerting
Set up alerts for:
- New best scores
- Training failures
- Optimization completion

### Team Collaboration
- Share project links with team members
- Add notes and comments to runs
- Create team-wide dashboards

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   # Re-login
   wandb logout
   wandb login
   ```

2. **Sync Issues**
   ```bash
   # Check wandb status
   wandb status

   # Force sync
   wandb sync your-run-directory
   ```

3. **Memory Issues**
   - Reduce logging frequency with `--log-interval`
   - Disable system metrics tracking

### Performance Tips
1. Use offline mode for faster training:
   ```bash
   WANDB_MODE=offline python train_summary.py --wandb
   ```

2. Sync later:
   ```bash
   wandb sync wandb/offline-run-*
   ```

3. Limit data tracked:
   - Reduce image logging
   - Filter out system metrics
   - Use sampling for large datasets

## Integration with CI/CD

### GitHub Actions
```yaml
- name: Login to wandb
  uses: wandb/actions/wandb-login@v0
  with:
    wandb_api_key: ${{ secrets.WANDB_API_KEY }}

- name: Train model
  run: |
    python scripts/train_summary.py --wandb
```

### Environment Variables
```bash
export WANDB_PROJECT="elm-summary"
export WANDB_ENTITY="your-team"
export WANDB_TAGS="production,summary-training"
```

This comprehensive guide should help you effectively track and analyze your ELM summary training experiments using wandb!