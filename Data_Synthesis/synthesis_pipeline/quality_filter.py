"""
Quality filtering for generated outputs.
"""

import asyncio
import re
from typing import Tuple, Dict
from collections import Counter

from .utils import get_logger
from .config import TaskConfig

logger = get_logger("quality_filter")


class QualityFilter:
    """Filter low-quality generated outputs."""

    def __init__(self, repetition_threshold: float = 0.5):
        """Initialize quality filter.

        Args:
            repetition_threshold: Threshold for n-gram repetition (0.0-1.0)
        """
        self.repetition_threshold = repetition_threshold
        self.rejection_counts: Dict[str, int] = {}
        self.total_counts: Dict[str, int] = {}
        self.lock = asyncio.Lock()  # Thread-safety for concurrent batch processing

    def filter(
        self,
        text: str,
        task_config: TaskConfig,
        original_text: str,
    ) -> Tuple[bool, str]:
        """Filter a single generated output.

        Args:
            text: Generated text to filter
            task_config: Task configuration
            original_text: Original input text

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        task_name = task_config.name

        # Perform all checks (CPU-bound, no lock needed)
        is_valid = True
        rejection_reason = ""

        # Check 1: Minimum token count
        word_count = len(text.split())
        if word_count < task_config.min_tokens:
            is_valid = False
            rejection_reason = f"Too short: {word_count} < {task_config.min_tokens} tokens"

        # Check 2: Contains instruction/prompt
        elif self._contains_instruction(text, task_config):
            is_valid = False
            rejection_reason = "Contains instruction text"

        # Check 3: Repetitive content
        elif self._is_repetitive(text):
            is_valid = False
            rejection_reason = "Repetitive content detected"

        # Check 4: Empty or nonsensical
        elif self._is_nonsensical(text):
            is_valid = False
            rejection_reason = "Nonsensical or empty output"

        # Check 5: Too similar to original
        elif self._too_similar_to_original(text, original_text):
            is_valid = False
            rejection_reason = "Too similar to original text"

        # Update counters (not locked here - will be rare contention, statistics only)
        # Note: Individual dict operations are atomic in CPython due to GIL
        self.total_counts[task_name] = self.total_counts.get(task_name, 0) + 1
        if not is_valid:
            self.rejection_counts[task_name] = self.rejection_counts.get(task_name, 0) + 1

        return is_valid, rejection_reason

    def _contains_instruction(self, text: str, task_config: TaskConfig) -> bool:
        """Check if output contains the instruction.

        Args:
            text: Generated text
            task_config: Task configuration

        Returns:
            True if instruction detected
        """
        # Extract common instruction phrases
        instruction_patterns = [
            r"text:",
            r"here is",
            r"as follows:",
            r"the following",
            r"your task",
            r"please",
        ]

        text_lower = text.lower()[:200]  # Check first 200 chars

        for pattern in instruction_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _is_repetitive(self, text: str) -> bool:
        """Check for repetitive n-grams.

        Args:
            text: Generated text

        Returns:
            True if repetitive content detected
        """
        words = text.lower().split()
        if len(words) < 20:
            return False

        # Check trigram repetition
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)

        if not trigrams:
            return False

        # If any trigram appears more than threshold proportion
        max_count = max(trigram_counts.values())
        if max_count / len(trigrams) > self.repetition_threshold:
            return True

        return False

    def _is_nonsensical(self, text: str) -> bool:
        """Check for nonsensical output.

        Args:
            text: Generated text

        Returns:
            True if nonsensical
        """
        # Empty or whitespace only
        if not text or not text.strip():
            return True

        # Too many non-alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return True

        return False

    def _too_similar_to_original(self, text: str, original: str) -> bool:
        """Check if output is too similar to original text.

        Args:
            text: Generated text
            original: Original input text

        Returns:
            True if too similar
        """
        # Simple word overlap check
        text_words = set(text.lower().split())
        original_words = set(original.lower().split())

        if not text_words:
            return False

        overlap = len(text_words & original_words) / len(text_words)

        # More than 80% overlap is suspicious
        return overlap > 0.8

    def get_rejection_rates(self) -> Dict[str, float]:
        """Get rejection rates per task.

        Returns:
            Dictionary mapping task name to rejection rate
        """
        rates = {}
        for task_name, total in self.total_counts.items():
            rejected = self.rejection_counts.get(task_name, 0)
            rates[task_name] = rejected / total if total > 0 else 0.0
        return rates

    def check_rejection_threshold(self, max_rate: float = 0.20) -> list:
        """Check which tasks exceed rejection threshold.

        Args:
            max_rate: Maximum allowed rejection rate

        Returns:
            List of task names exceeding threshold
        """
        rates = self.get_rejection_rates()
        return [task for task, rate in rates.items() if rate > max_rate]
