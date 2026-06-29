#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a scenario_type JSON into smaller batch JSON files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input scenario_type JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store split batch JSON files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of scenes per batch file.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional output file prefix. Defaults to input file stem.",
    )
    parser.add_argument(
        "--group-by-town",
        action="store_true",
        help="Split batches separately per source_town so each batch stays within one map.",
    )
    return parser.parse_args()


def write_batches(entries, output_dir, prefix, batch_size, town_name=None):
    num_batches = 0
    for start_idx in range(0, len(entries), batch_size):
        batch_entries = entries[start_idx:start_idx + batch_size]
        batch_start_data_id = batch_entries[0]["data_id"]
        batch_end_data_id = batch_entries[-1]["data_id"]
        town_suffix = f"_{town_name}" if town_name else ""
        output_path = output_dir / (
            f"{prefix}{town_suffix}_batch_{num_batches:02d}_data_{batch_start_data_id:04d}_{batch_end_data_id:04d}.json"
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(batch_entries, f, indent=2)
        print(f"Wrote {len(batch_entries)} scenes -> {output_path}")
        num_batches += 1
    return num_batches


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    with args.input.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("Input JSON must be a list of scenario entries")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.input.stem

    num_batches = 0
    if args.group_by_town:
        grouped_entries = {}
        for entry in entries:
            parameters = entry.get("parameters", {})
            town_name = parameters.get("source_town", "unknown_town")
            grouped_entries.setdefault(town_name, []).append(entry)

        for town_name in sorted(grouped_entries.keys()):
            town_entries = sorted(grouped_entries[town_name], key=lambda item: item["data_id"])
            num_batches += write_batches(town_entries, args.output_dir, prefix, args.batch_size, town_name=town_name)
    else:
        entries = sorted(entries, key=lambda item: item["data_id"])
        num_batches += write_batches(entries, args.output_dir, prefix, args.batch_size)

    print(f"Created {num_batches} batch files from {args.input}")


if __name__ == "__main__":
    main()
