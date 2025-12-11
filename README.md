# ELM Project

Embedding Language Model (ELM) data preparation and synthesis pipeline.

## Project Structure

This project is organized into two main components:

```
elm/
├── Data_Preparation/    # WikiText-2 preprocessing and embedding generation
├── Data_Synthesis/      # Synthetic data generation using LLM
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

Synthetic data generation pipeline using Qwen3-30B-A3B-Instruct via OpenRouter API.

**Features:**
- Generate diverse training targets across 16 task types (Factual, Descriptive, Creative, Pair-Based)
- FAISS k-NN index for efficient neighbor lookup in pair-based tasks
- OpenRouter API integration with rate limiting and retry logic
- Checkpointing with automatic resume capability
- Quality filtering (min tokens, repetition detection, instruction leakage)

**Quick Start:**
```bash
# Set API key
export OPENROUTER_API_KEY="your-key"

# From project root
python Data_Synthesis/scripts/run_synthesis.py
```

**Documentation:**
- See [Data_Synthesis/README.md](Data_Synthesis/README.md) for full documentation

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
- `data/synthesis/` - Synthetic training targets (JSONL format)

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
