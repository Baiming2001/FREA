#!/usr/bin/env python
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate batch scenario yaml files from a directory of batch JSON files."
    )
    parser.add_argument(
        "--base-yaml",
        type=Path,
        required=True,
        help="Base scenario yaml path.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="Directory containing batch JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store generated batch yaml files.",
    )
    parser.add_argument(
        "--yaml-prefix",
        type=str,
        default=None,
        help="Optional output yaml filename prefix. Defaults to base yaml stem.",
    )
    parser.add_argument(
        "--json-prefix",
        type=str,
        default=None,
        help="Optional filter for batch JSON filenames.",
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


def main():
    args = parse_args()
    base_text = args.base_yaml.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.json_prefix:
        json_files = sorted(args.json_dir.glob(f"{args.json_prefix}*.json"))
    else:
        json_files = sorted(args.json_dir.glob("*.json"))

    if not json_files:
        raise ValueError(f"No batch JSON files found in {args.json_dir}")

    yaml_prefix = args.yaml_prefix or args.base_yaml.stem

    for json_file in json_files:
        scenario_type_value = f"{args.json_dir.name}/{json_file.name}"
        yaml_name = f"{yaml_prefix}_{json_file.stem}.yaml"
        output_path = args.output_dir / yaml_name
        output_text = replace_scenario_type(base_text, scenario_type_value)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
