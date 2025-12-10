#!/usr/bin/env python3
"""
Update checkpoint files for all tasks to match actual train_synthesis.jsonl data.

This script automatically discovers all task types in the train_synthesis.jsonl file
and creates/updates checkpoint files for each task type with accurate completed indices
and generation counts.
"""

import json
from pathlib import Path
from datetime import datetime

def update_checkpoint(jsonl_path: Path, checkpoint_path: Path, task_name: str):
    """
    Update checkpoint file based on actual JSONL data.

    Args:
        jsonl_path: Path to train_synthesis.jsonl
        checkpoint_path: Path to checkpoint file
        task_name: Task name to update (e.g., "summary")
    """
    # Read actual data from JSONL
    completed_indices = set()
    total_count = 0

    print(f"Reading {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if record['task_type'] == task_name:
                completed_indices.add(record['embedding_index'])
                total_count += 1

    # Sort indices for consistency
    completed_indices = sorted(completed_indices)

    print(f"\nTask: {task_name}")
    print(f"  Total records found: {total_count}")
    print(f"  Unique embedding indices: {len(completed_indices)}")
    if completed_indices:
        print(f"  Index range: {min(completed_indices)} - {max(completed_indices)}")

    # Read existing checkpoint to preserve split name
    split_name = "train"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            old_checkpoint = json.load(f)
            split_name = old_checkpoint.get('split', 'train')
            old_count = len(old_checkpoint.get('completed_indices', []))
            old_gen = old_checkpoint.get('total_generations', 0)
            print(f"\nOld checkpoint:")
            print(f"  Completed indices: {old_count}")
            print(f"  Total generations: {old_gen}")

    # Create updated checkpoint
    checkpoint_data = {
        "split": split_name,
        "task_name": task_name,
        "completed_indices": completed_indices,
        "total_generations": total_count,
        "last_update": datetime.now().isoformat(),
        "rejection_count": 0,  # Reset since we're recalculating from actual data
        "metadata": {}
    }

    # Write updated checkpoint
    print(f"\nWriting updated checkpoint to {checkpoint_path}...")
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    print(f"✅ Checkpoint updated successfully!")
    print(f"\nNew checkpoint:")
    print(f"  Completed indices: {len(completed_indices)}")
    print(f"  Total generations: {total_count}")


if __name__ == "__main__":
    # Paths
    jsonl_path = Path("data/synthesis/train_synthesis.jsonl")
    checkpoints_dir = Path("data/synthesis/checkpoints")

    # Check if files exist
    if not jsonl_path.exists():
        print(f"❌ Error: {jsonl_path} not found!")
        exit(1)

    # Create checkpoint directory if needed
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # First, discover all unique task types in the JSONL file
    print("=" * 60)
    print("Discovering task types...")
    print("=" * 60)

    task_types = set()
    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            task_types.add(record['task_type'])

    task_types = sorted(task_types)
    print(f"\nFound {len(task_types)} task types: {', '.join(task_types)}\n")

    # Update checkpoint for each task type
    for i, task_name in enumerate(task_types, 1):
        print("=" * 60)
        print(f"Processing task {i}/{len(task_types)}: {task_name}")
        print("=" * 60)

        checkpoint_path = checkpoints_dir / f"train_{task_name}_checkpoint.json"
        update_checkpoint(jsonl_path, checkpoint_path, task_name)
        print()

    print("=" * 60)
    print(f"✅ All {len(task_types)} checkpoints updated successfully!")
    print("=" * 60)
