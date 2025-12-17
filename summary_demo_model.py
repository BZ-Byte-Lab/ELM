#!/usr/bin/env python3
"""Demo script for ELM summary model showing examples of trained summary adapter."""

import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import logging

# Add Training module to path
sys.path.append(str(Path(__file__).parent / "Training"))

from summary_training_pipeline.model import ELMModel
from summary_training_pipeline.config import SummaryTrainingConfig
from summary_training_pipeline.checkpoint import AdapterCheckpoint
from summary_training_pipeline.task_prompts import SINGLE_TEXT_PROMPTS
from safetensors.torch import load_file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_summary_model(checkpoint_path: str = "Training/data/checkpoints/adapter_best.safetensors"):
    """Load the trained ELM summary model for demonstration.

    Args:
        checkpoint_path: Path to adapter checkpoint file

    Returns:
        ELMModel with loaded adapter weights
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create config with summary training settings
    config = SummaryTrainingConfig(
        llm_model_name="Qwen/Qwen3-4B-Instruct-2507",
        hidden_dim=4096,
        embedding_dim=2560,
        use_residual=True,
        use_bf16=True,
        use_gradient_checkpointing=True,
        max_seq_length=2048,
    )

    # Create model and move to device
    model = ELMModel(config)
    model = model.to(device)
    model.eval()

    # Load adapter weights from checkpoint
    if Path(checkpoint_path).exists():
        logger.info(f"Loading adapter weights from {checkpoint_path}")
        adapter_weights = load_file(checkpoint_path)
        model.adapter.load_state_dict(adapter_weights)
        logger.info("Successfully loaded adapter weights")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return model, config


def load_sample_summary_embeddings(num_samples: int = 10):
    """Load sample summary embeddings for demonstration.

    Args:
        num_samples: Number of embeddings to load

    Returns:
        Tuple of (embeddings, texts)
    """
    # Load summary-specific embeddings and original text data
    embeddings_path = "data/summary_filtered/embeddings/test_embeddings.safetensors"
    text_data_path = "data/wikitext2_processed/test.parquet"

    try:
        import safetensors.torch as st
        import pandas as pd

        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings_dict = st.load_file(embeddings_path)

        # Get embeddings (usually stored under 'embeddings' or similar key)
        if 'embeddings' in embeddings_dict:
            embeddings = embeddings_dict['embeddings']
        else:
            # Use the first tensor we find
            key = list(embeddings_dict.keys())[0]
            embeddings = embeddings_dict[key]
            logger.info(f"Using key '{key}' for embeddings")

        # Load original text data from wikitext2
        logger.info(f"Loading original texts from {text_data_path}")
        df = pd.read_parquet(text_data_path)

        # Take first num_samples
        embeddings = embeddings[:num_samples]
        texts = df['text'].iloc[:num_samples].tolist()

        logger.info(f"Loaded {len(embeddings)} embeddings and {len(texts)} original texts")
        return embeddings, texts

    except Exception as e:
        logger.error(f"Error loading summary embeddings or text data: {e}")
        # Create dummy embeddings and texts for demonstration
        logger.info("Creating dummy embeddings and texts for demonstration")
        dummy_embeddings = torch.randn(num_samples, 2560)
        dummy_texts = [f"Sample text {i+1} for summary demonstration. This is a placeholder text since we couldn't load the actual data." for i in range(num_samples)]
        return dummy_embeddings, dummy_texts


def run_summary_demo(model: ELMModel, config: SummaryTrainingConfig, num_examples: int = 10):
    """Run demonstration of the trained summary model.

    Args:
        model: Trained ELMModel
        config: Training config
        num_examples: Number of examples to show
    """
    logger.info(f"Running summary demonstration with {num_examples} examples")

    # Load sample embeddings and texts
    embeddings, texts = load_sample_summary_embeddings(num_examples)

    # Move to device and match dtype (model uses bfloat16)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    embeddings = embeddings.to(device=device, dtype=model_dtype)

    # Summary task prompt template
    task_type = "summary"
    prompt_template = SINGLE_TEXT_PROMPTS[task_type]

    print("\n" + "="*80)
    print("ELM SUMMARY MODEL DEMONSTRATION")
    print("="*80)
    print(f"Model: {config.llm_model_name}")
    print(f"Adapter hidden dim: {config.hidden_dim}")
    print(f"Device: {device}")
    print(f"Task: {task_type}")
    print("="*80)

    for i in range(num_examples):
        print(f"\n--- Example {i+1} ---")

        # Show source text (truncated)
        if i < len(texts) and texts[i]:
            source_text = texts[i]
        else:
            source_text = f"Sample embedding {i+1} for summarization"

        print(f"\nSource Text: {source_text[:600]}..." if len(source_text) > 600 else f"\nSource Text: {source_text}")

        # Remove {text} placeholder (embedding carries the content)
        prompt = prompt_template.replace("{text}", "")

        print(f"\nTask Prompt: {prompt}")

        # Format EXACTLY as during training: <EMB> [prompt]
        full_text = f"{config.emb_token} {prompt}"

        # Tokenize
        inputs = model.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_length - 150,  # Leave room for generation
            padding=False,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Find <EMB> token position
        emb_token_id = model.emb_token_id
        emb_positions = (input_ids == emb_token_id).nonzero(as_tuple=True)[1]

        # Generate summary with appropriate parameters
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeddings=embeddings[i:i+1],
                embedding_positions=emb_positions,
                max_new_tokens=150,  # Summaries are typically shorter
                do_sample=True,
                temperature=0.5,    # Match summary training diversity
                top_p=0.9,          # Match summary training settings
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=model.tokenizer.eos_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )

        # Decode response
        # Remove input tokens from generated output
        response_ids = generated_ids[0][input_ids.shape[1]:]
        response = model.tokenizer.decode(response_ids, skip_special_tokens=True)

        print(f"\nGenerated Summary: {response}")
        print("-" * 60)

    print("\n" + "="*80)
    print("SUMMARY DEMONSTRATION COMPLETE")
    print("="*80)


def main():
    """Main function to run the summary demo."""
    import argparse

    parser = argparse.ArgumentParser(description="ELM Summary Model Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Training/data/checkpoints/adapter_best.safetensors",
        help="Path to adapter checkpoint file",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of examples to generate",
    )
    args = parser.parse_args()

    try:
        # Load model
        logger.info("Loading trained ELM summary model...")
        model, config = load_summary_model(args.checkpoint)

        # Run demo
        run_summary_demo(model, config, args.examples)

    except Exception as e:
        logger.error(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()