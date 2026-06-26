#!/usr/bin/env python
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a scenario yaml for a specific batch JSON file."
    )
    parser.add_argument(
        "--base-yaml",
        type=Path,
        required=True,
        help="Base scenario yaml path.",
    )
    parser.add_argument(
        "--scenario-type",
        type=str,
        required=True,
        help="Scenario type JSON filename to write into yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output yaml path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    text = args.base_yaml.read_text(encoding="utf-8")
    replaced = False
    output_lines = []
    for line in text.splitlines():
        if line.startswith("scenario_type:"):
            output_lines.append(f"scenario_type: '{args.scenario_type}'")
            replaced = True
        else:
            output_lines.append(line)

    if not replaced:
        raise ValueError("Base yaml does not contain a scenario_type field")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Wrote batch yaml -> {args.output}")


if __name__ == "__main__":
    main()
