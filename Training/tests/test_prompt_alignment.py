"""Tests to validate prompt alignment between synthesis and training.

This test ensures that the 13 single-text task prompts in the Training module
match exactly with the prompts used during data synthesis.
"""

import sys
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Data_Synthesis"))

from Training.training_pipeline.task_prompts import SINGLE_TEXT_PROMPTS
from Data_Synthesis.synthesis_pipeline.task_registry import TaskRegistry


def test_prompt_alignment():
    """Test that training prompts exactly match synthesis prompts for single-text tasks."""
    registry = TaskRegistry()

    errors = []
    matched = []

    # Get all single-text tasks
    single_text_tasks = [
        task_name for task_name, task_config in registry.tasks.items()
        if not task_config.requires_pair
    ]

    print(f"\nTesting {len(single_text_tasks)} single-text tasks...")
    print("=" * 70)

    for task_name in single_text_tasks:
        task_config = registry.get_task(task_name)
        synthesis_prompt = task_config.prompt_template

        # Get training prompt
        training_prompt = SINGLE_TEXT_PROMPTS.get(task_name)

        if training_prompt is None:
            errors.append(f"Task '{task_name}' missing from training prompts")
            continue

        # Compare prompts
        if synthesis_prompt != training_prompt:
            errors.append(
                f"Task '{task_name}' prompt mismatch:\n"
                f"  Synthesis length: {len(synthesis_prompt)} chars\n"
                f"  Training length:  {len(training_prompt)} chars\n"
                f"  First difference at char {_find_first_diff(synthesis_prompt, training_prompt)}"
            )
        else:
            matched.append(task_name)
            print(f"✓ {task_name:20s} - prompts match ({len(synthesis_prompt)} chars)")

    print("=" * 70)

    # Verify compare/hypothetical are NOT in training prompts
    if "compare" in SINGLE_TEXT_PROMPTS:
        errors.append("'compare' should NOT be in SINGLE_TEXT_PROMPTS")
    if "hypothetical" in SINGLE_TEXT_PROMPTS:
        errors.append("'hypothetical' should NOT be in SINGLE_TEXT_PROMPTS")

    # Report results
    print(f"\nResults:")
    print(f"  Matched:    {len(matched)}/{len(single_text_tasks)}")
    print(f"  Errors:     {len(errors)}")

    if errors:
        print("\n" + "=" * 70)
        print("ERRORS FOUND:")
        print("=" * 70)
        for error in errors:
            print(f"\n  ✗ {error}")
        print("\n" + "=" * 70)
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("SUCCESS: All 13 single-text task prompts aligned! ✓")
        print("=" * 70)
        sys.exit(0)


def _find_first_diff(str1: str, str2: str) -> int:
    """Find the index of the first character difference between two strings."""
    for i, (c1, c2) in enumerate(zip(str1, str2)):
        if c1 != c2:
            return i
    return min(len(str1), len(str2))


if __name__ == "__main__":
    test_prompt_alignment()
