"""Command line interface for cryoEMdoc."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .atlas import analyze_atlas, analyze_atlas_images
from .atlas_prerecognition import atlas_prerecognize
from .classifier import classify_image, classify_images
from .io import write_json, write_records_csv
from .pipeline import analyze_image, analyze_images
from .square import analyze_square, analyze_square_images


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return str(value)


def _emit(data: Any, output: str | Path | None = None) -> None:
    if output:
        if str(output).lower().endswith(".csv"):
            write_records_csv(data, output)
            return
        write_json(data, output)
    else:
        print(json.dumps(data, indent=2, default=_json_default))


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (dict, list)):
                    flat[f"{key}.{sub_key}"] = json.dumps(sub_value, default=_json_default)
                else:
                    flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, list):
            flat[key] = "; ".join(str(item) for item in value)
        else:
            flat[key] = value
    return flat


def _write_csv(records: list[dict[str, Any]], output: str | Path) -> None:
    rows = [_flatten_record(record) for record in records]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_folder(path: str | Path) -> bool:
    return Path(path).is_dir()


def cmd_classify(args: argparse.Namespace) -> None:
    if _is_folder(args.input):
        print(classify_images(args.input, output=args.output, recursive=args.recursive, device=args.device, model_dir=args.model_dir))
    else:
        print(classify_image(args.input, output=args.output, device=args.device, model_dir=args.model_dir))


def cmd_square(args: argparse.Namespace) -> None:
    if _is_folder(args.input):
        print(analyze_square_images(args.input, output=args.output, recursive=args.recursive, device=args.device, model_dir=args.model_dir))
    else:
        print(analyze_square(args.input, output=args.output, device=args.device, model_dir=args.model_dir))


def cmd_atlas(args: argparse.Namespace) -> None:
    if _is_folder(args.input):
        print(
            analyze_atlas_images(
                args.input,
                output=args.output,
                atlas_features=args.atlas_features,
                use_atlas_csv=args.use_atlas_csv,
                recursive=args.recursive,
                device=args.device,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
            )
        )
    else:
        print(
            analyze_atlas(
                args.input,
                output=args.output,
                atlas_features=args.atlas_features,
                use_atlas_csv=args.use_atlas_csv,
                device=args.device,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
            )
        )


def cmd_analyze(args: argparse.Namespace) -> None:
    if _is_folder(args.input):
        print(
            analyze_images(
                args.input,
                output=args.output,
                atlas_features=args.atlas_features,
                use_atlas_csv=args.use_atlas_csv,
                recursive=args.recursive,
                device=args.device,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
            )
        )
    else:
        print(
            analyze_image(
                args.input,
                output=args.output,
                atlas_features=args.atlas_features,
                use_atlas_csv=args.use_atlas_csv,
                device=args.device,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
            )
        )


def cmd_prerecognize(args: argparse.Namespace) -> None:
    result = atlas_prerecognize(
        args.input,
        output=args.output,
        recursive=args.recursive,
        save_annotated_images=not args.no_annotated,
        output_format=args.output_format,
        write_square_details=args.write_square_details,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, default=_json_default))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cryoemdoc", description="cryo-EM image triage and documentation")
    parser.add_argument("--version", action="version", version="cryoemdoc 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("input", help="Image path or folder.")
        subparser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        subparser.add_argument("--model-dir", default=None, help="External artifact root with cryoEMdoc subfolders.")
        subparser.add_argument("--output", default=None, help="Write standardized JSON output to this path.")
        subparser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)

    classify_parser = subparsers.add_parser("classify", help="Run image classification only.")
    add_common(classify_parser)
    classify_parser.set_defaults(func=cmd_classify)

    square_parser = subparsers.add_parser("square", help="Run square analyzer only.")
    add_common(square_parser)
    square_parser.set_defaults(func=cmd_square)

    atlas_parser = subparsers.add_parser("atlas", help="Run atlas analyzer only.")
    add_common(atlas_parser)
    atlas_parser.add_argument("--atlas-features", default="none", help="'none', 'generate', or path to existing CSV.")
    atlas_parser.add_argument("--use-atlas-csv", action="store_true", default=None, help="Generate atlas CSV features and use the CSV analyzer.")
    atlas_parser.add_argument("--output-dir", default=None, help="Directory for generated atlas CSV features.")
    atlas_parser.set_defaults(func=cmd_atlas)

    analyze_parser = subparsers.add_parser("analyze", help="Classify and route to the appropriate analyzer.")
    add_common(analyze_parser)
    analyze_parser.add_argument("--atlas-features", default="none", help="'none', 'generate', or path to existing CSV.")
    analyze_parser.add_argument("--use-atlas-csv", action="store_true", default=None, help="Generate atlas CSV features and use the CSV analyzer.")
    analyze_parser.add_argument("--output-dir", default=None, help="Directory for generated atlas CSV features.")
    analyze_parser.set_defaults(func=cmd_analyze)

    prerec_parser = subparsers.add_parser("atlas-prerecognize", help="Generate atlas summary CSV features.")
    prerec_parser.add_argument("input", help="Atlas image or folder.")
    prerec_parser.add_argument("--output", default="atlas_summary_scores.csv", help="CSV path or output directory.")
    prerec_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    prerec_parser.add_argument("--no-annotated", action="store_true", help="Skip annotated atlas overlay images.")
    prerec_parser.add_argument("--output-format", choices=["summary", "squares"], default="summary", help="Write one row per atlas or one row per detected square.")
    prerec_parser.add_argument("--write-square-details", action="store_true", help="Also write atlas_square_data.csv when output-format is summary.")
    prerec_parser.set_defaults(func=cmd_prerecognize)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
