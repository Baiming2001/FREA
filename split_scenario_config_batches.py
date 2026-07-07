#!/usr/bin/env python
import argparse
import json
from collections import defaultdict
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
        help="Split batches separately per parameters.split_name into train/val/test groups.",
    )
    parser.add_argument(
        "--split-subdirs",
        action="store_true",
        help="When grouping by split, write each split into its own subdirectory under output-dir.",
    )
    parser.add_argument(
        "--base-yaml",
        type=Path,
        default=None,
        help="Optional base scenario yaml template used to emit YAMLs for every batch JSON.",
    )
    parser.add_argument(
        "--yaml-output-dir",
        type=Path,
        default=None,
        help="Optional directory to store generated batch YAML files.",
    )
    parser.add_argument(
        "--yaml-prefix",
        type=str,
        default=None,
        help="Optional output yaml filename prefix. Defaults to the batch JSON prefix.",
    )
    return parser.parse_args()


def replace_scenario_type(base_yaml_text, scenario_type_value):
    replaced = False
    output_lines = []
    for line in base_yaml_text.splitlines():
        if line.startswith("scenario_type:"):
            output_lines.append(f"scenario_type: '{scenario_type_value}'")
            replaced = True
        else:
            output_lines.append(line)

    if not replaced:
        raise ValueError("Base yaml does not contain a scenario_type field")

    return "\n".join(output_lines) + "\n"


def resolve_scenario_type_value(json_path, scenario_type_root):
    try:
        return json_path.resolve().relative_to(scenario_type_root.resolve()).as_posix()
    except ValueError:
        return json_path.name


def write_batch_yaml(batch_json_path, yaml_output_path, base_yaml_text, scenario_type_root):
    scenario_type_value = resolve_scenario_type_value(batch_json_path, scenario_type_root)
    yaml_output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_output_path.write_text(
        replace_scenario_type(base_yaml_text, scenario_type_value),
        encoding="utf-8",
    )
    print(f"Wrote batch yaml -> {yaml_output_path}")


def write_batches(
    entries,
    output_dir,
    prefix,
    batch_size,
    town_name=None,
    split_name=None,
    base_yaml_text=None,
    yaml_output_dir=None,
    yaml_prefix=None,
    scenario_type_root=None,
):
    num_batches = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for start_idx in range(0, len(entries), batch_size):
        batch_entries = entries[start_idx:start_idx + batch_size]
        batch_start_data_id = batch_entries[0]["data_id"]
        batch_end_data_id = batch_entries[-1]["data_id"]
        town_suffix = f"_{town_name}" if town_name else ""
        split_suffix = f"_{split_name}" if split_name else ""
        output_path = output_dir / (
            f"{prefix}{split_suffix}{town_suffix}_batch_{num_batches:02d}_data_{batch_start_data_id:04d}_{batch_end_data_id:04d}.json"
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(batch_entries, f, indent=2)
        print(f"Wrote {len(batch_entries)} scenes -> {output_path}")

        if base_yaml_text is not None and yaml_output_dir is not None and scenario_type_root is not None:
            batch_yaml_prefix = yaml_prefix or prefix
            yaml_output_path = yaml_output_dir / (
                f"{batch_yaml_prefix}{split_suffix}{town_suffix}_batch_{num_batches:02d}.yaml"
            )
            write_batch_yaml(output_path, yaml_output_path, base_yaml_text, scenario_type_root)

        num_batches += 1
    return num_batches


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.yaml_output_dir is not None and args.base_yaml is None:
        raise ValueError("--base-yaml is required when --yaml-output-dir is used")
    if args.split_subdirs and not args.group_by_split:
        raise ValueError("--split-subdirs requires --group-by-split")

    with args.input.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("Input JSON must be a list of scenario entries")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.input.stem
    base_yaml_text = args.base_yaml.read_text(encoding="utf-8") if args.base_yaml is not None else None
    scenario_type_root = args.input.parent

    num_batches = 0
    if args.group_by_split:
        grouped_entries = defaultdict(list)
        for entry in entries:
            parameters = entry.get("parameters", {})
            split_name = parameters.get("split_name", "unknown_split")
            grouped_entries[split_name].append(entry)

        for split_name in sorted(grouped_entries.keys()):
            split_entries = sorted(grouped_entries[split_name], key=lambda item: item["data_id"])
            split_output_dir = args.output_dir / split_name if args.split_subdirs else args.output_dir
            split_yaml_output_dir = None
            if args.yaml_output_dir is not None:
                split_yaml_output_dir = args.yaml_output_dir / split_name if args.split_subdirs else args.yaml_output_dir
            if args.group_by_town:
                town_groups = defaultdict(list)
                for entry in split_entries:
                    parameters = entry.get("parameters", {})
                    town_name = parameters.get("source_town", "unknown_town")
                    town_groups[town_name].append(entry)
                for town_name in sorted(town_groups.keys()):
                    town_entries = sorted(town_groups[town_name], key=lambda item: item["data_id"])
                    num_batches += write_batches(
                        town_entries,
                        split_output_dir,
                        prefix,
                        args.batch_size,
                        town_name=town_name,
                        split_name=split_name,
                        base_yaml_text=base_yaml_text,
                        yaml_output_dir=split_yaml_output_dir,
                        yaml_prefix=args.yaml_prefix,
                        scenario_type_root=scenario_type_root,
                    )
            else:
                num_batches += write_batches(
                    split_entries,
                    split_output_dir,
                    prefix,
                    args.batch_size,
                    split_name=split_name,
                    base_yaml_text=base_yaml_text,
                    yaml_output_dir=split_yaml_output_dir,
                    yaml_prefix=args.yaml_prefix,
                    scenario_type_root=scenario_type_root,
                )
    elif args.group_by_town:
        grouped_entries = defaultdict(list)
        for entry in entries:
            parameters = entry.get("parameters", {})
            town_name = parameters.get("source_town", "unknown_town")
            grouped_entries[town_name].append(entry)

        for town_name in sorted(grouped_entries.keys()):
            town_entries = sorted(grouped_entries[town_name], key=lambda item: item["data_id"])
            num_batches += write_batches(
                town_entries,
                args.output_dir,
                prefix,
                args.batch_size,
                town_name=town_name,
                base_yaml_text=base_yaml_text,
                yaml_output_dir=args.yaml_output_dir,
                yaml_prefix=args.yaml_prefix,
                scenario_type_root=scenario_type_root,
            )
    else:
        entries = sorted(entries, key=lambda item: item["data_id"])
        num_batches += write_batches(
            entries,
            args.output_dir,
            prefix,
            args.batch_size,
            base_yaml_text=base_yaml_text,
            yaml_output_dir=args.yaml_output_dir,
            yaml_prefix=args.yaml_prefix,
            scenario_type_root=scenario_type_root,
        )

    print(f"Created {num_batches} batch files from {args.input}")


if __name__ == "__main__":
    main()
