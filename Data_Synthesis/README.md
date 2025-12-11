# ELM Data Synthesis Pipeline

Data synthesis pipeline for generating training targets using Qwen3-30B-A3B-Instruct via OpenRouter API.

## Overview

This pipeline generates synthetic training data from pre-computed embeddings by using an LLM to create diverse textual interpretations across 16 different task types. It includes k-NN pairing for comparative tasks, quality filtering, checkpointing, and comprehensive validation.

## Features

- **16 Task Types** across 4 categories (Factual, Descriptive, Creative, Pair-Based)
- **FAISS k-NN Index** for efficient neighbor lookup in pair-based tasks
- **OpenRouter API Integration** with rate limiting and retry logic
- **Checkpointing** every 20 generations with automatic resume capability
- **Quality Filtering** (min tokens, repetition detection, instruction leakage)
- **Validation Checklist** ensuring coverage and output quality
- **JSONL Output Format** for multi-task training

## Task Categories

### Category A - Factual (Temperature=0.3, top_p=0.85)
- `keywords`: Extract 5-7 key concepts (min 25 tokens)
- `category`: Identify academic field/domain (min 40 tokens)
- `questions`: Generate 3 answerable questions (min 60 tokens)

### Category B - Descriptive (Temperature=0.5, top_p=0.9)
- `summary`: Concise 2-3 sentence summary (min 50 tokens)
- `describe`: Detailed content description (min 120 tokens)
- `explain_beginner`: Beginner-friendly explanation (min 100 tokens)
- `explain_expert`: Expert-level explanation (min 100 tokens)
- `related_topics`: 5 related topics with connections (min 80 tokens)

### Category C - Creative (Temperature=0.7, top_p=0.92)
- `characteristics_pos`: 5 strengths/interesting aspects (min 80 tokens)
- `characteristics_neg`: 5 limitations/criticisms (min 80 tokens)
- `style_academic`: Formal academic description (min 100 tokens)
- `style_casual`: Conversational description (min 100 tokens)
- `counterfactual`: Application to random domain (min 100 tokens)

### Category D - Pair-Based (k-NN Required)
- `compare`: Compare two related texts using top-5 neighbors (T=0.5, min 150 tokens)
- `hypothetical`: Describe conceptual midpoint using top-3 neighbors with α∈[0.3,0.7] (T=0.7, min 120 tokens)

## Installation

### Prerequisites

```bash
# Ensure Data_Preparation phase is completed with embeddings generated
# Install new dependencies
cd /path/to/elm
pip install openai>=1.0.0 jsonlines>=3.0.0 faiss-cpu>=1.7.0
```

### Set API Key

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

## Usage

### Run Full Synthesis

```bash
# Generate for all splits (train, val, test)
python Data_Synthesis/scripts/run_synthesis.py

# Generate for specific splits
python Data_Synthesis/scripts/run_synthesis.py --splits train

# Custom rate limiting and checkpoint interval
python Data_Synthesis/scripts/run_synthesis.py \
    --requests-per-second 0.5 \
    --checkpoint-interval 50
```

### Validation Only

```bash
# Validate existing outputs
python Data_Synthesis/scripts/run_synthesis.py --validate-only
```

### Resume Interrupted Run

Checkpoints are saved automatically every 20 generations (configurable). Simply re-run the same command to resume:

```bash
python Data_Synthesis/scripts/run_synthesis.py --splits train
```

## Output Format

### JSONL Structure

Each line in `data/synthesis/{split}_synthesis.jsonl`:

```json
{
  "task_type": "summary",
  "input_prompt_template": "Write a concise summary...",
  "embedding_index": 42,
  "target_text": "This passage discusses...",
  "variation": 0,
  "temperature": 0.5,
  "top_p": 0.9,
  "token_count": 87
}
```

For pair-based tasks, additional fields:
- `neighbor_idx`: Index of k-NN neighbor used
- `alpha`: Interpolation weight (hypothetical task only)

## Architecture

```
Data_Synthesis/
├── synthesis_pipeline/
│   ├── config.py              # Configuration dataclass
│   ├── task_registry.py       # 16 task definitions with prompts
│   ├── knn_index.py           # FAISS k-NN index
│   ├── api_client.py          # OpenRouter client with rate limiting
│   ├── quality_filter.py      # Output filtering
│   ├── checkpoint.py          # Checkpointing & resume
│   ├── output_writer.py       # JSONL writer
│   ├── generator.py           # Main orchestrator
│   ├── validator.py           # Validation checklist
│   └── utils.py               # Logging & helpers
└── scripts/
    └── run_synthesis.py       # CLI entry point
```

## Configuration

Key parameters in [synthesis_pipeline/config.py](synthesis_pipeline/config.py):

- `requests_per_second`: API rate limit (default: 1.0)
- `checkpoint_interval`: Save every N generations (default: 20)
- `variations_per_task`: Number of variations (default: 2)
- `min_samples_per_embedding`: Coverage requirement (default: 15)
- `max_rejection_rate`: Maximum allowed rejection rate (default: 0.20)
- `knn_k`: Total k-NN neighbors to compute (default: 10)

## Quality Filtering

Each generated output is checked for:

1. **Minimum token count** (varies by task type)
2. **Instruction leakage** (contains prompt text)
3. **Repetitive n-grams** (threshold: 0.5)
4. **Nonsensical output** (<50% alphabetic characters)
5. **Too similar to original** (>80% word overlap)
6. **Duplicates** (MD5 hash check)

## Validation Checklist

Post-generation validation ensures:

- Every embedding has at least 15 samples across different tasks
- Compare tasks use k-NN pairs (verify neighbor_idx exists)
- Hypothetical tasks have alpha in [0.3, 0.7]
- No duplicate outputs (hash check)
- Quality rejection rate below 20% per task

## Estimated Output

- **6,017 total embeddings** (4,813 train, 601 val, 603 test)
- **~48 outputs per embedding** (14 single tasks × 2 vars + 2 pair tasks × ~5 neighbors × 2 vars)
- **Total: ~289,000 synthetic samples**

## Integration with Data_Preparation

The synthesis pipeline loads pre-computed embeddings from Phase 1:

```python
from data_pipeline import ELMDataset, Config as DataConfig

dataset = ELMDataset(
    data_dir=config.data_dir,
    split="train",
    load_embeddings=True,
    config=data_config
)
```

Embeddings are L2-normalized (unit vectors) from Qwen3-Embedding-4B, stored in SafeTensors format.

## Troubleshooting

### API Key Not Set

```
ValueError: OPENROUTER_API_KEY environment variable not set
```

**Solution**: Export the API key before running:
```bash
export OPENROUTER_API_KEY="your-key"
```

### High Rejection Rate

If a task exceeds 20% rejection rate, check:
- Prompt template clarity
- Min token threshold (may be too high)
- Quality filter settings in [quality_filter.py](synthesis_pipeline/quality_filter.py)

### Slow Generation

- Increase `requests_per_second` (if API allows)
- Use GPU-accelerated FAISS: set `use_gpu_index=True` in config
- Process fewer splits or reduce `variations_per_task`

## Citation

If you use this pipeline, please cite:

**Qwen3-30B-A3B-Instruct:**
```
@misc{qwen3-instruct,
  title={Qwen3},
  author={Qwen Team},
  year={2024},
  url={https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct}
}
```

**OpenRouter:**
```
@misc{openrouter,
  title={OpenRouter},
  url={https://openrouter.ai}
}
```

## License

Part of the ELM (Embedding Language Model) project.
