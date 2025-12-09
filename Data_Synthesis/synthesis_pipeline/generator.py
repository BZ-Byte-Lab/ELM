"""
Main generation orchestrator for synthesis pipeline.
"""

import asyncio
import random
import sys
from typing import List, Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Integration with Data_Preparation
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Data_Preparation"))
from data_pipeline import ELMDataset, Config as DataConfig

from .config import SynthesisConfig
from .task_registry import TaskRegistry, COUNTERFACTUAL_DOMAINS
from .knn_index import KNNIndex
from .api_client import OpenRouterClient
from .quality_filter import QualityFilter
from .checkpoint import CheckpointManager
from .output_writer import OutputWriter, SynthesisOutput
from .utils import get_logger

logger = get_logger("generator")


class SynthesisGenerator:
    """Main orchestrator for data synthesis."""

    def __init__(self, config: SynthesisConfig):
        """Initialize synthesis generator.

        Args:
            config: Synthesis configuration
        """
        self.config = config
        self.task_registry = TaskRegistry()
        self.api_client = OpenRouterClient(config)
        self.quality_filter = QualityFilter(config.repetition_threshold)
        self.checkpoint_manager = CheckpointManager(config)

        # Initialize k-NN index (built lazily)
        self.knn_index: Optional[KNNIndex] = None

        # Load Data_Preparation dataset
        self.data_config = DataConfig()
        self.data_config.base_dir = config.base_dir

        # Random state
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    def load_dataset(self, split: str) -> ELMDataset:
        """Load dataset from Data_Preparation module.

        Args:
            split: Data split name

        Returns:
            ELMDataset instance
        """
        logger.info(f"Loading {split} dataset...")
        dataset = ELMDataset(
            data_dir=self.config.data_dir,
            split=split,
            load_embeddings=True,
            config=self.data_config,
        )
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    def build_knn_index(self, dataset: ELMDataset) -> None:
        """Build k-NN index from dataset embeddings.

        Args:
            dataset: ELMDataset with embeddings
        """
        if dataset.embeddings is None:
            raise ValueError("Dataset must have embeddings loaded")

        self.knn_index = KNNIndex(self.config)
        self.knn_index.build_index(dataset.embeddings)

    def generate_for_split(self, split: str) -> Dict[str, Any]:
        """Generate synthetic data for a single split.

        Args:
            split: Data split name ('train', 'val', 'test')

        Returns:
            Dictionary of statistics
        """
        # Use async version if enabled
        if self.config.use_async:
            return asyncio.run(self.generate_for_split_async(split))

        # Sync version (for backward compatibility)
        logger.info(f"=" * 80)
        logger.info(f"Generating synthetic data for {split} split (sync mode)")
        logger.info(f"=" * 80)

        # Load dataset
        dataset = self.load_dataset(split)

        # Build k-NN index for pair-based tasks
        self.build_knn_index(dataset)

        # Initialize output writer
        writer = OutputWriter(self.config, split)

        # Track coverage: embedding_idx -> set of task types
        coverage: Dict[int, set] = {i: set() for i in range(len(dataset))}

        stats = {
            "total_generated": 0,
            "total_rejected": 0,
            "task_counts": {},
        }

        # Process single-text tasks first
        single_text_tasks = self.task_registry.get_single_text_tasks()
        for task_config in single_text_tasks:
            task_stats = self._process_single_text_task(
                dataset, task_config, writer, coverage, split
            )
            stats["task_counts"][task_config.name] = task_stats
            stats["total_generated"] += task_stats["generated"]
            stats["total_rejected"] += task_stats["rejected"]

        # Process pair-based tasks
        pair_tasks = self.task_registry.get_pair_tasks()
        for task_config in pair_tasks:
            task_stats = self._process_pair_task(
                dataset, task_config, writer, coverage, split
            )
            stats["task_counts"][task_config.name] = task_stats
            stats["total_generated"] += task_stats["generated"]
            stats["total_rejected"] += task_stats["rejected"]

        # Flush remaining outputs
        writer.close()

        # Check coverage requirement
        stats["coverage"] = self._calculate_coverage(coverage)

        # Check rejection rates
        stats["rejection_rates"] = self.quality_filter.get_rejection_rates()

        logger.info(f"\n{split} split completed:")
        logger.info(f"  Total generated: {stats['total_generated']}")
        logger.info(f"  Total rejected: {stats['total_rejected']}")
        logger.info(f"  Coverage: {stats['coverage']['mean']:.1f} tasks/embedding")

        return stats

    async def generate_for_split_async(self, split: str) -> Dict[str, Any]:
        """Generate synthetic data for a single split (async version).

        Args:
            split: Data split name ('train', 'val', 'test')

        Returns:
            Dictionary of statistics
        """
        logger.info(f"=" * 80)
        logger.info(f"Generating synthetic data for {split} split (async mode)")
        logger.info(f"=" * 80)

        # Load dataset
        dataset = self.load_dataset(split)

        # Build k-NN index for pair-based tasks
        self.build_knn_index(dataset)

        # Initialize output writer
        writer = OutputWriter(self.config, split)

        # Track coverage: embedding_idx -> set of task types
        coverage: Dict[int, set] = {i: set() for i in range(len(dataset))}

        stats = {
            "total_generated": 0,
            "total_rejected": 0,
            "task_counts": {},
        }

        # Process single-text tasks first
        single_text_tasks = self.task_registry.get_single_text_tasks()
        for task_config in single_text_tasks:
            task_stats = await self._process_single_text_task_async(
                dataset, task_config, writer, coverage, split
            )
            stats["task_counts"][task_config.name] = task_stats
            stats["total_generated"] += task_stats["generated"]
            stats["total_rejected"] += task_stats["rejected"]

        # Process pair-based tasks
        pair_tasks = self.task_registry.get_pair_tasks()
        for task_config in pair_tasks:
            task_stats = await self._process_pair_task_async(
                dataset, task_config, writer, coverage, split
            )
            stats["task_counts"][task_config.name] = task_stats
            stats["total_generated"] += task_stats["generated"]
            stats["total_rejected"] += task_stats["rejected"]

        # Flush remaining outputs
        writer.close()

        # Check coverage requirement
        stats["coverage"] = self._calculate_coverage(coverage)

        # Check rejection rates
        stats["rejection_rates"] = self.quality_filter.get_rejection_rates()

        logger.info(f"\n{split} split completed:")
        logger.info(f"  Total generated: {stats['total_generated']}")
        logger.info(f"  Total rejected: {stats['total_rejected']}")
        logger.info(f"  Coverage: {stats['coverage']['mean']:.1f} tasks/embedding")

        return stats

    def _process_single_text_task(
        self,
        dataset: ELMDataset,
        task_config,
        writer: OutputWriter,
        coverage: Dict[int, set],
        split: str,
    ) -> Dict[str, int]:
        """Process a single-text task across all embeddings.

        Args:
            dataset: ELMDataset instance
            task_config: TaskConfig for this task
            writer: OutputWriter instance
            coverage: Coverage tracking dictionary
            split: Data split name

        Returns:
            Dictionary with generated/rejected counts
        """
        logger.info(f"\nProcessing task: {task_config.name}")

        # Check for checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(split, task_config.name)
        completed_indices = checkpoint.completed_indices if checkpoint else set()

        generated = 0
        rejected = 0

        # Process each embedding
        indices_to_process = [i for i in range(len(dataset)) if i not in completed_indices]

        for idx in tqdm(indices_to_process, desc=f"{task_config.name}"):
            sample = dataset[idx]
            text = sample["text"]

            # Generate variations
            for var in range(task_config.variations):
                # Prepare prompt
                if task_config.name == "counterfactual":
                    random_domain = random.choice(COUNTERFACTUAL_DOMAINS)
                    prompt = task_config.prompt_template.format(
                        text=text, random_domain=random_domain
                    )
                else:
                    prompt = task_config.prompt_template.format(text=text)

                # Generate
                result = self.api_client.generate(prompt, task_config)

                if not result.success:
                    rejected += 1
                    continue

                # Quality filter
                is_valid, reason = self.quality_filter.filter(
                    result.text, task_config, text
                )

                if not is_valid:
                    rejected += 1
                    logger.debug(f"Rejected: {reason}")
                    continue

                # Check for duplicate
                if self.checkpoint_manager.is_duplicate(result.text):
                    rejected += 1
                    continue

                # Write output
                output = SynthesisOutput(
                    task_type=task_config.name,
                    input_prompt_template=task_config.prompt_template,
                    embedding_index=idx,
                    target_text=result.text,
                    metadata={
                        "variation": var,
                        "temperature": task_config.temperature,
                        "top_p": task_config.top_p,
                        "token_count": result.token_count,
                    },
                )
                writer.write(output)
                generated += 1
                coverage[idx].add(task_config.name)

            # Mark index as completed
            completed_indices.add(idx)

            # Checkpoint
            if self.checkpoint_manager.should_checkpoint(generated):
                self.checkpoint_manager.save_checkpoint(
                    split, task_config.name, completed_indices,
                    generated, rejected
                )

        # Final checkpoint
        self.checkpoint_manager.save_checkpoint(
            split, task_config.name, completed_indices, generated, rejected
        )

        return {"generated": generated, "rejected": rejected}

    async def _process_single_text_task_async(
        self,
        dataset: ELMDataset,
        task_config,
        writer: OutputWriter,
        coverage: Dict[int, set],
        split: str,
    ) -> Dict[str, int]:
        """Process a single-text task across all embeddings (async with batching).

        Args:
            dataset: ELMDataset instance
            task_config: TaskConfig for this task
            writer: OutputWriter instance
            coverage: Coverage tracking dictionary
            split: Data split name

        Returns:
            Dictionary with generated/rejected counts
        """
        logger.info(f"\nProcessing task: {task_config.name} (async mode)")

        # Check for checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(split, task_config.name)
        completed_indices = checkpoint.completed_indices if checkpoint else set()

        generated = 0
        rejected = 0

        indices_to_process = [i for i in range(len(dataset)) if i not in completed_indices]
        batch_size = self.config.batch_size

        # Process in batches
        total_batches = (len(indices_to_process) + batch_size - 1) // batch_size

        with tqdm(total=len(indices_to_process), desc=f"{task_config.name}") as pbar:
            for batch_num in range(total_batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, len(indices_to_process))
                batch_indices = indices_to_process[batch_start:batch_end]

                # Prepare all prompts for this batch
                batch_prompts = []
                batch_metadata = []

                for idx in batch_indices:
                    sample = dataset[idx]
                    text = sample["text"]

                    for var in range(task_config.variations):
                        # Prepare prompt
                        if task_config.name == "counterfactual":
                            random_domain = random.choice(COUNTERFACTUAL_DOMAINS)
                            prompt = task_config.prompt_template.format(
                                text=text, random_domain=random_domain
                            )
                        else:
                            prompt = task_config.prompt_template.format(text=text)

                        batch_prompts.append(prompt)
                        batch_metadata.append((idx, var, text))

                # Generate all prompts in parallel (10 concurrent via semaphore)
                results = await self.api_client.generate_batch_async(
                    batch_prompts,
                    task_config,
                )

                # Process results
                outputs_to_write = []

                for result, (idx, var, original_text) in zip(results, batch_metadata):
                    if not result.success:
                        rejected += 1
                        continue

                    # Quality filter
                    is_valid, reason = self.quality_filter.filter(
                        result.text, task_config, original_text
                    )

                    if not is_valid:
                        rejected += 1
                        logger.debug(f"Rejected: {reason}")
                        continue

                    # Check for duplicate
                    if self.checkpoint_manager.is_duplicate(result.text):
                        rejected += 1
                        continue

                    # Prepare output
                    output = SynthesisOutput(
                        task_type=task_config.name,
                        input_prompt_template=task_config.prompt_template,
                        embedding_index=idx,
                        target_text=result.text,
                        metadata={
                            "variation": var,
                            "temperature": task_config.temperature,
                            "top_p": task_config.top_p,
                            "token_count": result.token_count,
                        },
                    )
                    outputs_to_write.append(output)
                    generated += 1
                    coverage[idx].add(task_config.name)

                # Batch write outputs
                writer.write_batch(outputs_to_write)

                # Update completed indices
                for idx in batch_indices:
                    completed_indices.add(idx)

                # Checkpoint after each batch
                if self.checkpoint_manager.should_checkpoint(generated):
                    self.checkpoint_manager.save_checkpoint(
                        split, task_config.name, completed_indices,
                        generated, rejected
                    )

                # Update progress
                pbar.update(len(batch_indices))
                logger.debug(f"Batch {batch_num + 1}/{total_batches}: "
                           f"{generated} generated, {rejected} rejected")

        # Final checkpoint
        self.checkpoint_manager.save_checkpoint(
            split, task_config.name, completed_indices, generated, rejected
        )

        return {"generated": generated, "rejected": rejected}

    def _process_pair_task(
        self,
        dataset: ELMDataset,
        task_config,
        writer: OutputWriter,
        coverage: Dict[int, set],
        split: str,
    ) -> Dict[str, int]:
        """Process a pair-based task using k-NN neighbors.

        Args:
            dataset: ELMDataset instance
            task_config: TaskConfig for this task
            writer: OutputWriter instance
            coverage: Coverage tracking dictionary
            split: Data split name

        Returns:
            Dictionary with generated/rejected counts
        """
        logger.info(f"\nProcessing pair task: {task_config.name}")

        # Check for checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(split, task_config.name)
        completed_indices = checkpoint.completed_indices if checkpoint else set()

        generated = 0
        rejected = 0

        indices_to_process = [i for i in range(len(dataset)) if i not in completed_indices]

        for idx in tqdm(indices_to_process, desc=f"{task_config.name}"):
            sample = dataset[idx]
            text1 = sample["text"]

            # Get k-NN neighbors
            neighbor_indices, similarities = self.knn_index.search(idx, task_config.knn_k)

            # Generate with each neighbor
            for neighbor_idx in neighbor_indices[:task_config.knn_k]:
                neighbor_sample = dataset[int(neighbor_idx)]
                text2 = neighbor_sample["text"]

                for var in range(task_config.variations):
                    # Prepare prompt based on task type
                    if task_config.name == "hypothetical":
                        # Random alpha in range
                        alpha = random.uniform(*task_config.alpha_range)
                        prompt = task_config.prompt_template.format(
                            text1=text1, text2=text2,
                            alpha1=1-alpha, alpha2=alpha
                        )
                        metadata_extra = {"alpha": alpha, "neighbor_idx": int(neighbor_idx)}
                    else:  # compare
                        prompt = task_config.prompt_template.format(
                            text1=text1, text2=text2
                        )
                        metadata_extra = {"neighbor_idx": int(neighbor_idx)}

                    # Generate
                    result = self.api_client.generate(prompt, task_config)

                    if not result.success:
                        rejected += 1
                        continue

                    # Quality filter
                    is_valid, reason = self.quality_filter.filter(
                        result.text, task_config, text1 + " " + text2
                    )

                    if not is_valid:
                        rejected += 1
                        continue

                    # Check for duplicate
                    if self.checkpoint_manager.is_duplicate(result.text):
                        rejected += 1
                        continue

                    # Write output
                    output = SynthesisOutput(
                        task_type=task_config.name,
                        input_prompt_template=task_config.prompt_template,
                        embedding_index=idx,
                        target_text=result.text,
                        metadata={
                            "variation": var,
                            "temperature": task_config.temperature,
                            "top_p": task_config.top_p,
                            "token_count": result.token_count,
                            **metadata_extra,
                        },
                    )
                    writer.write(output)
                    generated += 1
                    coverage[idx].add(task_config.name)

            completed_indices.add(idx)

            if self.checkpoint_manager.should_checkpoint(generated):
                self.checkpoint_manager.save_checkpoint(
                    split, task_config.name, completed_indices,
                    generated, rejected
                )

        self.checkpoint_manager.save_checkpoint(
            split, task_config.name, completed_indices, generated, rejected
        )

        return {"generated": generated, "rejected": rejected}

    async def _process_pair_task_async(
        self,
        dataset: ELMDataset,
        task_config,
        writer: OutputWriter,
        coverage: Dict[int, set],
        split: str,
    ) -> Dict[str, int]:
        """Process a pair-based task using k-NN neighbors (async with batching).

        Args:
            dataset: ELMDataset instance
            task_config: TaskConfig for this task
            writer: OutputWriter instance
            coverage: Coverage tracking dictionary
            split: Data split name

        Returns:
            Dictionary with generated/rejected counts
        """
        logger.info(f"\nProcessing pair task: {task_config.name} (async mode)")

        # Check for checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(split, task_config.name)
        completed_indices = checkpoint.completed_indices if checkpoint else set()

        generated = 0
        rejected = 0

        indices_to_process = [i for i in range(len(dataset)) if i not in completed_indices]

        # Adjust batch size for pair tasks (smaller because more prompts per embedding)
        batch_size = max(1, self.config.batch_size // (task_config.knn_k * task_config.variations))
        total_batches = (len(indices_to_process) + batch_size - 1) // batch_size

        with tqdm(total=len(indices_to_process), desc=f"{task_config.name}") as pbar:
            for batch_num in range(total_batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, len(indices_to_process))
                batch_indices = indices_to_process[batch_start:batch_end]

                # Prepare all prompts for this batch
                batch_prompts = []
                batch_metadata = []

                for idx in batch_indices:
                    sample = dataset[idx]
                    text1 = sample["text"]

                    # Get k-NN neighbors
                    neighbor_indices, similarities = self.knn_index.search(idx, task_config.knn_k)

                    # Generate with each neighbor
                    for neighbor_idx in neighbor_indices[:task_config.knn_k]:
                        neighbor_sample = dataset[int(neighbor_idx)]
                        text2 = neighbor_sample["text"]

                        for var in range(task_config.variations):
                            # Prepare prompt based on task type
                            if task_config.name == "hypothetical":
                                alpha = random.uniform(*task_config.alpha_range)
                                prompt = task_config.prompt_template.format(
                                    text1=text1, text2=text2,
                                    alpha1=1-alpha, alpha2=alpha
                                )
                                metadata_extra = {"alpha": alpha, "neighbor_idx": int(neighbor_idx)}
                            else:  # compare
                                prompt = task_config.prompt_template.format(
                                    text1=text1, text2=text2
                                )
                                metadata_extra = {"neighbor_idx": int(neighbor_idx)}

                            batch_prompts.append(prompt)
                            batch_metadata.append((idx, var, text1 + " " + text2, metadata_extra))

                # Generate all prompts in parallel (10 concurrent via semaphore)
                results = await self.api_client.generate_batch_async(
                    batch_prompts,
                    task_config,
                )

                # Process results
                outputs_to_write = []

                for result, (idx, var, combined_text, metadata_extra) in zip(results, batch_metadata):
                    if not result.success:
                        rejected += 1
                        continue

                    # Quality filter
                    is_valid, reason = self.quality_filter.filter(
                        result.text, task_config, combined_text
                    )

                    if not is_valid:
                        rejected += 1
                        continue

                    # Check for duplicate
                    if self.checkpoint_manager.is_duplicate(result.text):
                        rejected += 1
                        continue

                    # Write output
                    output = SynthesisOutput(
                        task_type=task_config.name,
                        input_prompt_template=task_config.prompt_template,
                        embedding_index=idx,
                        target_text=result.text,
                        metadata={
                            "variation": var,
                            "temperature": task_config.temperature,
                            "top_p": task_config.top_p,
                            "token_count": result.token_count,
                            **metadata_extra,
                        },
                    )
                    outputs_to_write.append(output)
                    generated += 1
                    coverage[idx].add(task_config.name)

                # Batch write outputs
                writer.write_batch(outputs_to_write)

                # Update completed indices
                for idx in batch_indices:
                    completed_indices.add(idx)

                # Checkpoint after each batch
                if self.checkpoint_manager.should_checkpoint(generated):
                    self.checkpoint_manager.save_checkpoint(
                        split, task_config.name, completed_indices,
                        generated, rejected
                    )

                # Update progress
                pbar.update(len(batch_indices))
                logger.debug(f"Batch {batch_num + 1}/{total_batches}: "
                           f"{generated} generated, {rejected} rejected")

        # Final checkpoint
        self.checkpoint_manager.save_checkpoint(
            split, task_config.name, completed_indices, generated, rejected
        )

        return {"generated": generated, "rejected": rejected}

    def _calculate_coverage(self, coverage: Dict[int, set]) -> Dict[str, Any]:
        """Calculate coverage statistics.

        Args:
            coverage: Dictionary mapping embedding index to set of task types

        Returns:
            Coverage statistics dictionary
        """
        task_counts = [len(tasks) for tasks in coverage.values()]

        return {
            "min": min(task_counts) if task_counts else 0,
            "max": max(task_counts) if task_counts else 0,
            "mean": sum(task_counts) / len(task_counts) if task_counts else 0,
            "below_threshold": sum(
                1 for c in task_counts if c < self.config.min_samples_per_embedding
            ),
        }

    def run(self, splits: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run synthesis for specified splits.

        Args:
            splits: List of splits to process (default: all)

        Returns:
            Combined statistics
        """
        if splits is None:
            splits = ["train", "val", "test"]

        self.config.create_directories()

        all_stats = {}
        for split in splits:
            all_stats[split] = self.generate_for_split(split)

        return all_stats
