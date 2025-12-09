"""
Validation checklist implementation for synthesis outputs.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict
import jsonlines

from .config import SynthesisConfig
from .task_registry import TaskRegistry
from .utils import get_logger

logger = get_logger("validator")


class SynthesisValidator:
    """Validates synthesis outputs against requirements."""

    def __init__(self, config: SynthesisConfig):
        """Initialize synthesis validator.

        Args:
            config: Synthesis configuration
        """
        self.config = config
        self.task_registry = TaskRegistry()

    def validate_all(self, splits: List[str] = None) -> Dict[str, Any]:
        """Run all validation checks.

        Args:
            splits: Splits to validate (default: all)

        Returns:
            Validation results dictionary
        """
        if splits is None:
            splits = ["train", "val", "test"]

        results = {
            "coverage": {},
            "pair_tasks": {},
            "duplicates": {},
            "overall": {"passed": True, "issues": []},
        }

        for split in splits:
            output_path = self.config.get_synthesis_path(split)
            if not output_path.exists():
                logger.warning(f"Output file not found: {output_path}")
                continue

            # Load outputs
            outputs = self._load_outputs(output_path)

            # Validation 1: Coverage check
            coverage_result = self._check_coverage(outputs, split)
            results["coverage"][split] = coverage_result
            if not coverage_result["passed"]:
                results["overall"]["passed"] = False
                results["overall"]["issues"].append(
                    f"{split}: Coverage below threshold"
                )

            # Validation 2: Pair task verification
            pair_result = self._check_pair_tasks(outputs, split)
            results["pair_tasks"][split] = pair_result
            if not pair_result["passed"]:
                results["overall"]["passed"] = False
                results["overall"]["issues"].append(
                    f"{split}: Pair task validation failed"
                )

            # Validation 3: Duplicate check
            dup_result = self._check_duplicates(outputs, split)
            results["duplicates"][split] = dup_result
            if not dup_result["passed"]:
                results["overall"]["passed"] = False
                results["overall"]["issues"].append(
                    f"{split}: Duplicates found"
                )

        return results

    def _load_outputs(self, path: Path) -> List[Dict]:
        """Load outputs from JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of output dictionaries
        """
        outputs = []
        with jsonlines.open(path) as reader:
            for item in reader:
                outputs.append(item)
        return outputs

    def _check_coverage(self, outputs: List[Dict], split: str) -> Dict[str, Any]:
        """Check that every embedding has at least 15 samples.

        Args:
            outputs: List of output dictionaries
            split: Data split name

        Returns:
            Coverage check results
        """
        # Count tasks per embedding
        embedding_tasks: Dict[int, Set[str]] = defaultdict(set)

        for output in outputs:
            idx = output["embedding_index"]
            task = output["task_type"]
            embedding_tasks[idx].add(task)

        # Check threshold
        below_threshold = []
        for idx, tasks in embedding_tasks.items():
            if len(tasks) < self.config.min_samples_per_embedding:
                below_threshold.append((idx, len(tasks)))

        passed = len(below_threshold) == 0

        return {
            "passed": passed,
            "total_embeddings": len(embedding_tasks),
            "below_threshold_count": len(below_threshold),
            "below_threshold_examples": below_threshold[:10],
            "min_coverage": min(len(t) for t in embedding_tasks.values()) if embedding_tasks else 0,
            "max_coverage": max(len(t) for t in embedding_tasks.values()) if embedding_tasks else 0,
            "mean_coverage": sum(len(t) for t in embedding_tasks.values()) / len(embedding_tasks) if embedding_tasks else 0,
        }

    def _check_pair_tasks(self, outputs: List[Dict], split: str) -> Dict[str, Any]:
        """Verify pair-based tasks use valid k-NN neighbors.

        Args:
            outputs: List of output dictionaries
            split: Data split name

        Returns:
            Pair task validation results
        """
        pair_tasks = self.task_registry.get_pair_tasks()
        pair_task_names = {t.name for t in pair_tasks}

        issues = []

        for output in outputs:
            if output["task_type"] not in pair_task_names:
                continue

            # Check that neighbor_idx exists
            if "neighbor_idx" not in output:
                issues.append(f"Missing neighbor_idx for {output['task_type']}")
                continue

            # For hypothetical task, check alpha range
            if output["task_type"] == "hypothetical":
                if "alpha" not in output:
                    issues.append("Missing alpha for hypothetical task")
                elif not (0.3 <= output["alpha"] <= 0.7):
                    issues.append(f"Alpha {output['alpha']} outside [0.3, 0.7]")

        return {
            "passed": len(issues) == 0,
            "issues": issues[:20],  # Limit to first 20
            "total_pair_outputs": sum(
                1 for o in outputs if o["task_type"] in pair_task_names
            ),
        }

    def _check_duplicates(self, outputs: List[Dict], split: str) -> Dict[str, Any]:
        """Check for duplicate outputs using hash.

        Args:
            outputs: List of output dictionaries
            split: Data split name

        Returns:
            Duplicate check results
        """
        seen_hashes: Set[str] = set()
        duplicates = []

        for i, output in enumerate(outputs):
            text_hash = hashlib.md5(output["target_text"].encode()).hexdigest()

            if text_hash in seen_hashes:
                duplicates.append(i)
            else:
                seen_hashes.add(text_hash)

        return {
            "passed": len(duplicates) == 0,
            "duplicate_count": len(duplicates),
            "duplicate_indices": duplicates[:20],
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report.

        Args:
            results: Validation results dictionary

        Returns:
            Formatted report string
        """
        lines = ["=" * 80, "SYNTHESIS VALIDATION REPORT", "=" * 80, ""]

        # Overall status
        status = "PASSED" if results["overall"]["passed"] else "FAILED"
        lines.append(f"Overall Status: {status}")
        lines.append("")

        if results["overall"]["issues"]:
            lines.append("Issues:")
            for issue in results["overall"]["issues"]:
                lines.append(f"  - {issue}")
            lines.append("")

        # Coverage details
        lines.append("-" * 40)
        lines.append("COVERAGE CHECK")
        lines.append("-" * 40)
        for split, cov in results["coverage"].items():
            lines.append(f"\n{split}:")
            lines.append(f"  Total embeddings: {cov['total_embeddings']}")
            lines.append(f"  Min coverage: {cov['min_coverage']} tasks")
            lines.append(f"  Max coverage: {cov['max_coverage']} tasks")
            lines.append(f"  Mean coverage: {cov['mean_coverage']:.1f} tasks")
            lines.append(f"  Below threshold: {cov['below_threshold_count']}")

        # Pair task details
        lines.append("\n" + "-" * 40)
        lines.append("PAIR TASK VALIDATION")
        lines.append("-" * 40)
        for split, pair in results["pair_tasks"].items():
            lines.append(f"\n{split}:")
            lines.append(f"  Total pair outputs: {pair['total_pair_outputs']}")
            lines.append(f"  Issues: {len(pair['issues'])}")

        # Duplicate check
        lines.append("\n" + "-" * 40)
        lines.append("DUPLICATE CHECK")
        lines.append("-" * 40)
        for split, dup in results["duplicates"].items():
            lines.append(f"\n{split}:")
            lines.append(f"  Duplicates found: {dup['duplicate_count']}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)
