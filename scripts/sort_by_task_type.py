#!/usr/bin/env python3
"""Sort train_synthesis.jsonl by task_type."""

import json
from pathlib import Path


def sort_by_task_type(input_path: str, output_path: str | None = None) -> None:
    """
    Sort JSONL file by task_type field.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output file. If None, overwrites input file.
    """
    input_file = Path(input_path)
    output_file = Path(output_path) if output_path else input_file

    # Read all records
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    # Count by task_type before sorting
    task_counts = {}
    for record in records:
        task_type = record.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    print("\nTask type counts:")
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count}")

    # Sort by task_type
    records.sort(key=lambda x: x.get("task_type", ""))

    # Write sorted records
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSorted records written to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sort train_synthesis.jsonl by task_type")
    parser.add_argument(
        "--input", "-i",
        default="data/synthesis/train_synthesis.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: overwrite input)"
    )

    args = parser.parse_args()
    sort_by_task_type(args.input, args.output)
