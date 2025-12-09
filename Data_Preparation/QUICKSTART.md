# Quick Start Guide

Get up and running with the ELM data preparation pipeline in minutes!

## Step 1: Set Up Environment

```bash
# Navigate to project directory
cd /home/benz/coding_project/elm

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate elm
```

## Step 2: Run the Pipeline

```bash
# Option 1: From project root
python Data_Preparation/scripts/run_pipeline.py

# Option 2: From Data_Preparation directory
cd Data_Preparation
python scripts/run_pipeline.py
```

This will:
- Download WikiText-2 from HuggingFace (~1-2 min)
- Preprocess and filter paragraphs (~3-5 min)
- Split into train/val/test (80/10/10)
- Generate embeddings with Qwen3-Embedding-4B (~5-10 min)

## Step 3: Verify the Output

```bash
# Check that files were created
ls -lh data/wikitext2_processed/
ls -lh data/embeddings/
```

Expected output:
```
data/wikitext2_processed/
  train.parquet
  val.parquet
  test.parquet

data/embeddings/
  train_embeddings.safetensors
  val_embeddings.safetensors
  test_embeddings.safetensors
```

## Step 4: Use the Dataset

```bash
# Run example usage script (from project root)
python Data_Preparation/scripts/example_usage.py

# Or from Data_Preparation directory
cd Data_Preparation
python scripts/example_usage.py
```

Or use in your own code:

```python
from data_pipeline import ELMDataset, Config

# Load training data
config = Config()
train_dataset = ELMDataset(
    data_dir=config.data_dir,
    split="train",
    load_embeddings=True
)

# Get a sample
sample = train_dataset[0]
print(f"Text: {sample['text'][:100]}...")
print(f"Embedding shape: {sample['embedding'].shape}")  # (2560,)

# Create DataLoader
dataloader = train_dataset.get_dataloader(batch_size=32, shuffle=True)

# Use in training loop
for batch in dataloader:
    texts = batch['text']
    embeddings = batch['embedding']  # (batch_size, 2560)
    # Your training code here...
```

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
python Data_Preparation/scripts/run_pipeline.py --batch-size 4
```

### Flash Attention Not Available

Disable flash attention:
```bash
python Data_Preparation/scripts/run_pipeline.py --no-flash-attention
```

### Want to Regenerate Embeddings

```bash
# Use different batch size or model
python Data_Preparation/scripts/run_embedding_only.py --batch-size 4
```

**Note:** All examples assume you're in the project root. If you're in Data_Preparation/, omit the `Data_Preparation/` prefix.

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Check [data_pipeline/config.py](data_pipeline/config.py) to customize settings
- See [scripts/example_usage.py](scripts/example_usage.py) for more examples

## Common Use Cases

### Custom Token Range

```bash
python Data_Preparation/scripts/run_pipeline.py --min-tokens 50 --max-tokens 1500
```

### Only Preprocess (No Embeddings)

```bash
python Data_Preparation/scripts/run_pipeline.py --skip-embeddings
```

### Use Existing Processed Data

```bash
python Data_Preparation/scripts/run_pipeline.py --skip-download
python Data_Preparation/scripts/run_embedding_only.py
```

## Support

If you encounter issues:
1. Check logs in `logs/pipeline.log`
2. Verify GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check VRAM usage: `nvidia-smi`

Happy embedding!
