"""Dataset and collator for ELM training."""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from torch.utils.data import Dataset
from safetensors.numpy import load_file
from transformers import PreTrainedTokenizer
import logging

from .task_prompts import get_single_text_prompt, is_single_text_task

logger = logging.getLogger(__name__)


class ELMTrainingDataset(Dataset):
    """Dataset for ELM adapter training.

    Loads synthetic JSONL data and maps embedding_index to actual embeddings.
    Each sample contains:
        - task_type: Type of synthesis task
        - embedding_index: Index into embeddings SafeTensor
        - target_text: Generated target text
    """

    def __init__(
        self,
        synthesis_path: Path,
        embeddings_path: Path,
        max_samples: int = None,
    ):
        """Initialize training dataset.

        Args:
            synthesis_path: Path to JSONL file (e.g., train_synthesis.jsonl)
            embeddings_path: Path to SafeTensors embeddings
            max_samples: Maximum number of samples to load (None = all)
        """
        self.synthesis_path = synthesis_path
        self.embeddings_path = embeddings_path

        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}")
        tensors = load_file(str(embeddings_path))
        self.embeddings = tensors["embeddings"]  # (num_samples, 2560)
        logger.info(f"Loaded embeddings shape: {self.embeddings.shape}")

        # Load synthesis data
        logger.info(f"Loading synthesis data from {synthesis_path}")
        self.samples = self._load_jsonl(synthesis_path, max_samples)
        logger.info(f"Loaded {len(self.samples)} training samples")

        # Validate
        self._validate_data()

    def _load_jsonl(self, path: Path, max_samples: int = None) -> List[Dict[str, Any]]:
        """Load JSONL samples."""
        samples = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                samples.append(json.loads(line.strip()))
        return samples

    def _validate_data(self):
        """Validate that all embedding indices are within bounds."""
        max_idx = len(self.embeddings)
        invalid_count = 0

        for i, sample in enumerate(self.samples):
            idx = sample["embedding_index"]
            if idx < 0 or idx >= max_idx:
                logger.warning(
                    f"Sample {i} has invalid embedding_index {idx} "
                    f"(max: {max_idx - 1})"
                )
                invalid_count += 1

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} samples with invalid indices")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample.

        Returns dict with:
            - embedding: np.ndarray (2560,)
            - task_type: str
            - target_text: str
            - embedding_index: int
        """
        sample = self.samples[idx]
        embedding_idx = sample["embedding_index"]

        # Clamp index to valid range (defensive)
        embedding_idx = max(0, min(embedding_idx, len(self.embeddings) - 1))

        embedding = self.embeddings[embedding_idx]

        return {
            "embedding": embedding,
            "task_type": sample["task_type"],
            "target_text": sample["target_text"],
            "embedding_index": embedding_idx,
        }


class TrainingCollator:
    """Collator for batching ELM training samples.

    Creates input sequences with <EMB> token followed by task prompt,
    then target text as labels.
    """

    # Task type to prompt mapping (from Data_Synthesis task registry)
    TASK_PROMPTS = {
        "keywords": "Extract key concepts:",
        "category": "Classify this content:",
        "questions": "Generate questions:",
        "summary": "Summarize:",
        "describe": "Describe in detail:",
        "explain_beginner": "Explain simply:",
        "explain_expert": "Explain technically:",
        "related_topics": "List related topics:",
        "characteristics_pos": "List strengths:",
        "characteristics_neg": "List weaknesses:",
        "style_academic": "Write academically:",
        "style_casual": "Write casually:",
        "counterfactual": "Imagine differently:",
        "compare": "Compare these:",
        "hypothetical": "Describe the midpoint:",
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        emb_token: str = "<EMB>",
    ):
        """Initialize collator.

        Args:
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length
            emb_token: Special token for embedding placeholder
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.emb_token = emb_token
        self.emb_token_id = tokenizer.convert_tokens_to_ids(emb_token)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples.

        Input format: "<EMB> [Task Prompt]\n[Target Text]"

        Returns:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            embeddings: (batch, embedding_dim)
            embedding_positions: (batch,) - position of <EMB> token
            labels: (batch, seq_len) - -100 for prompt, token IDs for target
        """
        input_texts = []
        embeddings = []

        for sample in batch:
            task_type = sample["task_type"]
            target_text = sample["target_text"]

            # Phase I: Use full synthesis prompts for single-text tasks
            # Phase II will handle compare/hypothetical
            if task_type in ["compare", "hypothetical"]:
                # Keep short prompts for pair-based tasks (Phase II will handle)
                prompt = self.TASK_PROMPTS.get(task_type, "Process:")
            else:
                # Use full synthesis prompts for single-text tasks
                try:
                    prompt = get_single_text_prompt(task_type)
                    # Remove {text} placeholder (embedding carries content)
                    prompt = prompt.replace("{text}", "")
                    # Remove {random_domain} placeholder for counterfactual
                    # (it was filled during synthesis, now just part of the text)
                    prompt = prompt.replace("{random_domain}", "")
                except ValueError as e:
                    # Fallback to short prompt if task not found
                    logger.warning(f"Could not get full prompt for task '{task_type}': {e}")
                    prompt = self.TASK_PROMPTS.get(task_type, "Process:")

            # Format: <EMB> [prompt]\n[target]
            # The <EMB> will be replaced by adapted embedding during forward pass
            input_text = f"{self.emb_token} {prompt}\n{target_text}"
            input_texts.append(input_text)
            embeddings.append(sample["embedding"])

        # Tokenize
        encodings = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Find <EMB> positions (should be at position 0 or 1 for each sample)
        emb_positions = []
        for i in range(len(batch)):
            # Find first occurrence of EMB token in this sequence
            emb_mask = (input_ids[i] == self.emb_token_id)
            if emb_mask.any():
                pos = emb_mask.nonzero(as_tuple=True)[0][0].item()
            else:
                # Fallback to position 0 if not found (shouldn't happen)
                pos = 0
                logger.warning(f"Sample {i}: <EMB> token not found, using position 0")
            emb_positions.append(pos)

        embedding_positions = torch.tensor(emb_positions, dtype=torch.long)

        # Create labels for causal LM
        # Mask prompt tokens with -100, keep target tokens
        labels = input_ids.clone()

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Mask prompt portion (everything before "\n")
        # We need to find where the "\n" token is and mask everything before it
        newline_id = self.tokenizer.convert_tokens_to_ids("\n")

        for i, text in enumerate(input_texts):
            # Find prompt portion
            prompt_text = text.split("\n")[0] + "\n"  # Include the newline
            prompt_tokens = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
            )["input_ids"]

            # Mask prompt tokens
            prompt_len = len(prompt_tokens)
            if prompt_len < len(labels[i]):
                labels[i, :prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "embeddings": torch.tensor(np.stack(embeddings), dtype=torch.float32),
            "embedding_positions": embedding_positions,
            "labels": labels,
        }
