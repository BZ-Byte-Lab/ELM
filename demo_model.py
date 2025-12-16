#!/usr/bin/env python3
"""Demo script for ELM model showing 10 examples of trained adapter."""

import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import logging

# Add Training module to path
sys.path.append(str(Path(__file__).parent / "Training"))

from training_pipeline.model import ELMModel
from training_pipeline.config import TrainingConfig
from training_pipeline.checkpoint import AdapterCheckpoint
from training_pipeline.task_prompts import SINGLE_TEXT_PROMPTS
from safetensors.torch import load_file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_demo_model(checkpoint_path: str = "data/checkpoints/adapter_best.safetensors"):
    """Load the trained ELM model for demonstration.

    Args:
        checkpoint_path: Path to adapter checkpoint file

    Returns:
        ELMModel with loaded adapter weights
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create config with default settings (matching training)
    config = TrainingConfig(
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


def load_sample_embeddings(num_samples: int = 10):
    """Load sample embeddings for demonstration.

    Args:
        num_samples: Number of embeddings to load

    Returns:
        Tuple of (embeddings, texts)
    """
    # Load validation embeddings (you can change to train if needed)
    embeddings_path = "data/embeddings/val_embeddings.safetensors"
    data_path = "data/wikitext2_processed/val.parquet"

    try:
        import safetensors.torch as st
        import pandas as pd

        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings_dict = st.load_file(embeddings_path)
        # The key might be different, let's check what's available
        print("Available keys in embeddings file:", list(embeddings_dict.keys())[:5])

        # Get embeddings (usually stored under 'embeddings' or similar key)
        if 'embeddings' in embeddings_dict:
            embeddings = embeddings_dict['embeddings']
        else:
            # Use the first tensor we find
            key = list(embeddings_dict.keys())[0]
            embeddings = embeddings_dict[key]
            print(f"Using key '{key}' for embeddings")

        # Load corresponding texts
        logger.info(f"Loading texts from {data_path}")
        df = pd.read_parquet(data_path)

        # Take first num_samples
        embeddings = embeddings[:num_samples]
        texts = df['text'].iloc[:num_samples].tolist()

        logger.info(f"Loaded {len(embeddings)} embeddings and texts")
        return embeddings, texts

    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        # Create dummy embeddings for demonstration
        logger.info("Creating dummy embeddings for demonstration")
        dummy_embeddings = torch.randn(num_samples, 2560)
        dummy_texts = [f"Sample text {i+1}" for i in range(num_samples)]
        return dummy_embeddings, dummy_texts


def create_demo_prompt():
    """Create a demonstration prompt template."""
    return """<EMB> Based on the above content, please provide:"""


def run_demo(model: ELMModel, config: TrainingConfig, num_examples: int = 10):
    """Run demonstration of the trained model.

    Args:
        model: Trained ELMModel
        config: Training config
        num_examples: Number of examples to show
    """
    logger.info(f"Running demonstration with {num_examples} examples")

    # Load sample embeddings and texts
    embeddings, texts = load_sample_embeddings(num_examples)

    # Move to device and match dtype (model uses bfloat16)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    embeddings = embeddings.to(device=device, dtype=model_dtype)

    # Use ACTUAL training task types (cycle through them)
    # These match exactly what the model saw during training
    task_types = [
        "summary",              # Summarization
        "keywords",             # Key concept extraction
        "explain_beginner",     # Simple explanation
        "category",             # Classification
        "questions",            # Question generation
        "describe",             # Detailed description
        "related_topics",       # Related topics
        "explain_expert",       # Technical explanation
        "characteristics_pos",  # Strengths/interesting aspects
        "style_casual",         # Casual tone rewrite
    ]

    print("\n" + "="*80)
    print("ELM MODEL DEMONSTRATION")
    print("="*80)
    print(f"Model: {config.llm_model_name}")
    print(f"Adapter hidden dim: {config.hidden_dim}")
    print(f"Device: {device}")
    print("="*80)

    for i in range(num_examples):
        print(f"\n--- Example {i+1} ---")

        # Show source text (truncated)
        source_text = texts[i] if i < len(texts) else f"Sample embedding {i+1}"
        print(f"\nSource Text: {source_text[:600]}..." if len(source_text) > 600 else f"\nSource Text: {source_text}")

        # Select task type and get EXACT training prompt
        task_type = task_types[i % len(task_types)]
        prompt_template = SINGLE_TEXT_PROMPTS[task_type]

        # Remove {text} placeholder (embedding carries the content)
        # and {random_domain} if present (for counterfactual task)
        prompt = prompt_template.replace("{text}", "").replace("{random_domain}", "various domains")

        print(f"\nTask Type: {task_type}")
        print(f"Task Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Task Prompt: {prompt}")

        # Format EXACTLY as during training: <EMB> [prompt]
        # The prompt already contains the task structure and ends with a completion cue
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

        # Generate response with appropriate parameters
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeddings=embeddings[i:i+1],
                embedding_positions=emb_positions,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.5,    # Match training diversity
                top_p=0.9,          # Match training settings
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=model.tokenizer.eos_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )

        # Decode response
        # Remove input tokens from generated output
        response_ids = generated_ids[0][input_ids.shape[1]:]
        response = model.tokenizer.decode(response_ids, skip_special_tokens=True)

        print(f"\nGenerated Response: {response}")
        print("-" * 60)

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


def main():
    """Main function to run the demo."""
    import argparse

    parser = argparse.ArgumentParser(description="ELM Model Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/adapter_best.safetensors",
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
        logger.info("Loading trained ELM model...")
        model, config = load_demo_model(args.checkpoint)

        # Run demo
        run_demo(model, config, args.examples)

    except Exception as e:
        logger.error(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()