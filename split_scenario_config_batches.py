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
    parser.add_argument(
        "--group-by-split",
        action="store_true",
        help="Split batches separately per dataset split using parameters.split_name.",
    )
    parser.add_argument(
        "--split-subdirs",
        action="store_true",
        help="When grouping by split, write JSON and YAML files into split subdirectories.",
    )
    parser.add_argument(
        "--base-yaml",
        type=Path,
        default=None,
        help="Optional base scenario yaml used to auto-generate batch yaml files.",
    )
    parser.add_argument(
        "--yaml-output-dir",
        type=Path,
        default=None,
        help="Directory to store generated batch yaml files.",
    )
    parser.add_argument(
        "--yaml-prefix",
        type=str,
        default=None,
        help="Optional yaml filename prefix. Defaults to base yaml stem.",
    )
    return parser.parse_args()


def replace_scenario_type(text, scenario_type_value):
    replaced = False
    output_lines = []
    for line in text.splitlines():
        if line.startswith("scenario_type:"):
            output_lines.append(f"scenario_type: '{scenario_type_value}'")
            replaced = True
        else:
            output_lines.append(line)

    if not replaced:
        raise ValueError("Base yaml does not contain a scenario_type field")
    return "\n".join(output_lines) + "\n"


def write_yaml_for_batch(base_text, json_root_dir, json_path, yaml_output_dir, yaml_prefix):
    yaml_output_dir.mkdir(parents=True, exist_ok=True)
    relative_json_path = json_path.relative_to(json_root_dir)
    scenario_type_value = f"{json_root_dir.name}/{relative_json_path.as_posix()}"
    yaml_name = f"{yaml_prefix}_{json_path.stem}.yaml"
    yaml_path = yaml_output_dir / yaml_name
    yaml_text = replace_scenario_type(base_text, scenario_type_value)
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Wrote yaml -> {yaml_path}")


def write_batches(
    entries,
    output_dir,
    prefix,
    batch_size,
    json_root_dir,
    group_name=None,
    base_yaml_text=None,
    yaml_output_dir=None,
    yaml_prefix=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_batches = 0
    for start_idx in range(0, len(entries), batch_size):
        batch_entries = entries[start_idx:start_idx + batch_size]
        batch_start_data_id = batch_entries[0]["data_id"]
        batch_end_data_id = batch_entries[-1]["data_id"]
        group_suffix = f"_{group_name}" if group_name else ""
        output_path = output_dir / (
            f"{prefix}{group_suffix}_batch_{num_batches:02d}_data_{batch_start_data_id:04d}_{batch_end_data_id:04d}.json"
        )
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(batch_entries, output_file, indent=2)
        print(f"Wrote {len(batch_entries)} scenes -> {output_path}")

        if base_yaml_text is not None and yaml_output_dir is not None and yaml_prefix is not None:
            write_yaml_for_batch(base_yaml_text, json_root_dir, output_path, yaml_output_dir, yaml_prefix)
        num_batches += 1

    return num_batches


def build_grouped_entries(entries, group_by_town, group_by_split):
    if group_by_town and group_by_split:
        raise ValueError("--group-by-town and --group-by-split are mutually exclusive")

    if group_by_split:
        grouped_entries = {}
        for entry in entries:
            split_name = entry.get("parameters", {}).get("split_name", "unknown_split")
            grouped_entries.setdefault(split_name, []).append(entry)
        return grouped_entries

    if group_by_town:
        grouped_entries = {}
        for entry in entries:
            town_name = entry.get("parameters", {}).get("source_town", "unknown_town")
            grouped_entries.setdefault(town_name, []).append(entry)
        return grouped_entries

    return {None: entries}


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.split_subdirs and not args.group_by_split:
        raise ValueError("--split-subdirs requires --group-by-split")
    if (args.base_yaml is None) != (args.yaml_output_dir is None):
        raise ValueError("--base-yaml and --yaml-output-dir must be provided together")

    with args.input.open("r", encoding="utf-8") as input_file:
        entries = json.load(input_file)
    if not isinstance(entries, list):
        raise ValueError("Input JSON must be a list of scenario entries")

    entries = sorted(entries, key=lambda item: item["data_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.input.stem

    base_yaml_text = None
    yaml_output_dir = None
    yaml_prefix = None
    if args.base_yaml is not None and args.yaml_output_dir is not None:
        base_yaml_text = args.base_yaml.read_text(encoding="utf-8")
        yaml_output_dir = args.yaml_output_dir
        yaml_prefix = args.yaml_prefix or args.base_yaml.stem

    grouped_entries = build_grouped_entries(entries, args.group_by_town, args.group_by_split)

    num_batches = 0
    for group_name in sorted(grouped_entries.keys(), key=lambda item: "" if item is None else str(item)):
        group_entries = sorted(grouped_entries[group_name], key=lambda item: item["data_id"])
        json_output_dir = args.output_dir / group_name if (args.split_subdirs and group_name is not None) else args.output_dir
        group_yaml_output_dir = (
            yaml_output_dir / group_name
            if (yaml_output_dir is not None and args.split_subdirs and group_name is not None)
            else yaml_output_dir
        )
        num_batches += write_batches(
            group_entries,
            json_output_dir,
            prefix,
            args.batch_size,
            args.output_dir,
            group_name=group_name,
            base_yaml_text=base_yaml_text,
            yaml_output_dir=group_yaml_output_dir,
            yaml_prefix=yaml_prefix,
        )

    print(f"Created {num_batches} batch files from {args.input}")


if __name__ == "__main__":
    main()
