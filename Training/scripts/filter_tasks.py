#!/usr/bin/env python3
"""Filter out compare and hypothetical tasks from synthesis JSONL files."""

import json
import shutil
from pathlib import Path

# Data files to filter
data_files = [
    "data/synthesis/train_synthesis.jsonl",
    "data/synthesis/val_synthesis.jsonl",
    "data/synthesis/test_synthesis.jsonl"
]

tasks_to_remove = ["compare", "hypothetical"]

def filter_jsonl_file(file_path):
    """Filter a JSONL file to remove specified task types."""
    file_path = Path(file_path)

    # Create backup
    backup_path = file_path.with_suffix(".jsonl.bak")
    shutil.copy(file_path, backup_path)
    print(f"Created backup: {backup_path}")

    # Filter the file
    filtered_lines = []
    removed_count = 0
    total_count = 0

    with open(file_path, 'r') as f:
        for line in f:
            total_count += 1
            data = json.loads(line.strip())

            if data.get("task_type") not in tasks_to_remove:
                filtered_lines.append(line)
            else:
                removed_count += 1

    # Write filtered data
    with open(file_path, 'w') as f:
        f.writelines(filtered_lines)

    remaining_count = total_count - removed_count
    print(f"{file_path.name}: {total_count} total -> {removed_count} removed -> {remaining_count} remaining")
    return total_count, removed_count, remaining_count

if __name__ == "__main__":
    print("Filtering out 'compare' and 'hypothetical' tasks from synthesis data files...\n")

    total_all = 0
    removed_all = 0

    for data_file in data_files:
        total, removed, remaining = filter_jsonl_file(data_file)
        total_all += total
        removed_all += removed

    print(f"\nTotal: {total_all} -> {removed_all} removed -> {total_all - removed_all} remaining")
    print("Filtering complete!")
