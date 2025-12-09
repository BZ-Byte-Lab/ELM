# ELM Data Preparation Pipeline

A complete data preparation pipeline for Embedding Language Models (ELM) using WikiText-2 and Qwen3-Embedding-4B.

## Features

- Download and preprocess WikiText-2 dataset from HuggingFace
- Extract clean paragraphs with configurable token filtering (100-2000 tokens)
- Generate high-quality embeddings using Qwen3-Embedding-4B (2560 dimensions)
- Efficient data storage using Parquet and SafeTensors formats
- Unified dataset class with embedding interpolation support
- Comprehensive logging and error handling

## Project Structure

```
elm/
├── Data_Preparation/           # Data preparation pipeline (this module)
│   ├── data_pipeline/          # Core pipeline modules
│   │   ├── config.py          # Configuration settings
│   │   ├── download.py        # Dataset loading
│   │   ├── preprocess.py      # Text preprocessing
│   │   ├── embeddings.py      # Embedding generation
│   │   ├── dataset.py         # PyTorch dataset class
│   │   └── utils.py           # Utility functions
│   ├── scripts/               # Executable scripts
│   │   ├── run_pipeline.py    # Main pipeline script
│   │   ├── run_embedding_only.py  # Regenerate embeddings
│   │   └── example_usage.py   # Usage examples
│   ├── elm_colab.ipynb        # Jupyter notebook
│   ├── README.md              # This file
│   ├── QUICKSTART.md          # Quick start guide
│   └── implementation.md      # Implementation details
├── Data_Synthesis/            # (Reserved for future use)
├── data/                      # Output directory (created at runtime)
│   ├── wikitext2_processed/
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── embeddings/
│       ├── train_embeddings.safetensors
│       ├── val_embeddings.safetensors
│       └── test_embeddings.safetensors
├── logs/                      # Pipeline execution logs
├── environment.yml            # Conda environment specification
└── pyproject.toml             # Package configuration
```

## Installation

### 1. Create Conda Environment

```bash
# Navigate to project root
cd /path/to/elm

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate elm
```

### 2. Install Package (Optional)

```bash
# From project root
pip install -e .
```

## Usage

### Running the Complete Pipeline

To run the full pipeline (download, preprocess, and generate embeddings):

```bash
# Option 1: From project root
python Data_Preparation/scripts/run_pipeline.py

# Option 2: From Data_Preparation directory
cd Data_Preparation
python scripts/run_pipeline.py
```

This will:
1. Download WikiText-2 from HuggingFace
2. Extract and clean paragraphs
3. Filter by token count (100-2000 tokens)
4. Split into train/val/test (80/10/10)
5. Generate embeddings using Qwen3-Embedding-4B
6. Save processed data and embeddings

### Command-Line Options

```bash
# Skip download if you already have processed data
python Data_Preparation/scripts/run_pipeline.py --skip-download

# Skip embedding generation (only preprocess text)
python Data_Preparation/scripts/run_pipeline.py --skip-embeddings

# Override batch size (default: 8 for 16GB VRAM)
python Data_Preparation/scripts/run_pipeline.py --batch-size 4

# Override token filtering
python Data_Preparation/scripts/run_pipeline.py --min-tokens 50 --max-tokens 1500

# Disable flash attention if not available
python Data_Preparation/scripts/run_pipeline.py --no-flash-attention

# Custom log file
python Data_Preparation/scripts/run_pipeline.py --log-file my_run.log

# Note: All examples above assume you're in the project root.
# If you're in Data_Preparation/, use: python scripts/run_pipeline.py [options]
```

### Regenerating Embeddings Only

If you want to regenerate embeddings with different parameters:

```bash
# Regenerate all splits (from project root)
python Data_Preparation/scripts/run_embedding_only.py

# Only specific splits
python Data_Preparation/scripts/run_embedding_only.py --splits train val

# Use different model
python Data_Preparation/scripts/run_embedding_only.py --model-name "Qwen/Qwen2-Embedding-7B"

# Adjust batch size for your GPU
python Data_Preparation/scripts/run_embedding_only.py --batch-size 4

# Disable optimizations
python Data_Preparation/scripts/run_embedding_only.py --no-flash-attention --no-fp16
```

## Using the Dataset Class

```python
from pathlib import Path
from data_pipeline import ELMDataset, Config

# Initialize configuration
config = Config()

# Load training data with embeddings
train_dataset = ELMDataset(
    data_dir=config.data_dir,
    split="train",
    load_embeddings=True
)

# Get a single sample
sample = train_dataset[0]
print(f"Text: {sample['text'][:100]}...")
print(f"Embedding shape: {sample['embedding'].shape}")
print(f"Metadata: {sample['metadata']}")

# Get dataset statistics
stats = train_dataset.get_statistics()
print(f"Dataset stats: {stats}")

# Create a PyTorch DataLoader
dataloader = train_dataset.get_dataloader(batch_size=32, shuffle=True)

for batch in dataloader:
    texts = batch['text']          # List of strings
    embeddings = batch['embedding']  # numpy array (batch_size, 2560)
    metadata = batch['metadata']    # List of dicts
    break

# Interpolate between embeddings
interpolated = train_dataset.interpolate_embeddings(
    idx1=0,
    idx2=100,
    alpha=0.5  # 50% mix
)
print(f"Interpolated embedding shape: {interpolated.shape}")
```

## Configuration

Edit [data_pipeline/config.py](data_pipeline/config.py) to customize:

```python
# Text filtering
min_tokens = 100      # Minimum tokens per paragraph
max_tokens = 2000     # Maximum tokens per paragraph

# Dataset split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Embedding model
model_name = "Qwen/Qwen3-Embedding-4B"
embedding_dim = 2560
max_length = 8192
batch_size = 8        # Adjust based on your GPU VRAM

# Optimizations
use_flash_attention = True
use_fp16 = True
```

## Hardware Requirements

- **GPU**: Recommended for embedding generation
  - 16GB VRAM: batch_size=8 (default)
  - 8GB VRAM: batch_size=2-4
  - 24GB+ VRAM: batch_size=16
- **CPU**: Works but very slow for embeddings
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for processed data and embeddings

## Output Format

### Processed Text (Parquet)

Each split contains:
- `text_id`: Unique identifier
- `text`: Cleaned paragraph text
- `token_count`: Number of tokens
- `char_count`: Number of characters

### Embeddings (SafeTensors)

Each split contains:
- `embeddings`: numpy array of shape (n_samples, 2560)
- Metadata: split name, model, number of texts

## Text Preprocessing

The pipeline applies the following cleaning steps:

1. Remove Wikipedia section headers (` = = Section = = `)
2. Remove formatting artifacts (`@-@`, `@.@`, `@,@`)
3. Remove `<unk>` tokens
4. Normalize whitespace
5. Filter low-quality text (too many special chars/digits)
6. Filter by token count (100-2000 tokens)

## Embedding Generation

- **Model**: Qwen3-Embedding-4B
- **Dimension**: 2560
- **Method**: Last token pooling (as per Qwen3 specification)
- **Normalization**: L2 normalization applied
- **No instructions**: Documents encoded directly (per official recommendation)
- **Optimizations**: Flash Attention 2, FP16

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python Data_Preparation/scripts/run_pipeline.py --batch-size 2

# Disable FP16
python Data_Preparation/scripts/run_embedding_only.py --no-fp16
```

### Flash Attention Not Available

```bash
# Disable flash attention
python Data_Preparation/scripts/run_pipeline.py --no-flash-attention
```

### Missing Dependencies

```bash
# Reinstall conda environment
conda env remove -n elm
conda env create -f environment.yml
conda activate elm

# Or install specific package
pip install flash-attn --no-build-isolation
```

## Pipeline Performance

Approximate times (with 16GB GPU):
- Download WikiText-2: ~1-2 minutes
- Preprocessing: ~3-5 minutes
- Embedding generation: ~5-10 minutes (varies by dataset size)

## Citation

If you use this pipeline, please cite:

**Qwen3-Embedding-4B:**
```
@misc{qwen3-embedding,
  title={Qwen3-Embedding},
  author={Qwen Team},
  year={2024},
  url={https://huggingface.co/Qwen/Qwen3-Embedding-4B}
}
```

**WikiText-2:**
```
@misc{merity2016pointer,
  title={Pointer Sentinel Mixture Models},
  author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
  year={2016},
  eprint={1609.07843},
  archivePrefix={arXiv}
}
```

## License

This project is provided as-is for research and educational purposes.
