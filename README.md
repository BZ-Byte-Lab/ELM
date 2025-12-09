# ELM Project

Embedding Language Model (ELM) data preparation and synthesis pipeline.

## Project Structure

This project is organized into two main components:

```
elm/
├── Data_Preparation/    # WikiText-2 preprocessing and embedding generation
├── Data_Synthesis/      # (Reserved for future synthetic data generation)
├── data/                # Output directory for all processed data
├── logs/                # Pipeline execution logs
├── environment.yml      # Conda environment specification
└── pyproject.toml       # Package configuration
```

## Components

### Data_Preparation

Complete data preparation pipeline for ELM using WikiText-2 and Qwen3-Embedding-4B.

**Features:**
- Download and preprocess WikiText-2 dataset from HuggingFace
- Extract clean paragraphs with configurable token filtering (100-2000 tokens)
- Generate high-quality embeddings using Qwen3-Embedding-4B (2560 dimensions)
- Efficient data storage using Parquet and SafeTensors formats
- Unified dataset class with embedding interpolation support

**Quick Start:**
```bash
# From project root
python Data_Preparation/scripts/run_pipeline.py
```

**Documentation:**
- See [Data_Preparation/README.md](Data_Preparation/README.md) for full documentation
- See [Data_Preparation/QUICKSTART.md](Data_Preparation/QUICKSTART.md) for quick start guide

### Data_Synthesis

Reserved for future development of synthetic data generation capabilities.

## Installation

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate elm

# Install package in editable mode (optional)
pip install -e .
```

## Usage

All scripts can be run from the project root:

```bash
# Run full data preparation pipeline
python Data_Preparation/scripts/run_pipeline.py

# Run embedding generation only
python Data_Preparation/scripts/run_embedding_only.py

# Run usage examples
python Data_Preparation/scripts/example_usage.py
```

Or navigate to the specific component directory:

```bash
cd Data_Preparation
python scripts/run_pipeline.py
```

## Output

All processed data is stored in the `data/` directory at the project root:
- `data/wikitext2_processed/` - Preprocessed text files (Parquet format)
- `data/embeddings/` - Generated embeddings (SafeTensors format)

All logs are stored in the `logs/` directory at the project root.

## Hardware Requirements

- **GPU**: Recommended for embedding generation
  - 16GB VRAM: batch_size=8 (default)
  - 8GB VRAM: batch_size=2-4
  - 24GB+ VRAM: batch_size=16
- **CPU**: Works but very slow for embeddings
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for processed data and embeddings

## License

This project is provided as-is for research and educational purposes.
