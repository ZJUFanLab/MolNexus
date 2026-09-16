#!/usr/bin/env python3
"""Validate and delegate to the independent CGI prediction package."""

from __future__ import annotations

import argparse
import sys

from _runtime import (
    fail,
    local_path,
    project_root,
    require_dir,
    require_file,
    run_or_dry_run,
)


def add_optional(command: list[str], flag: str, value: str | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or execute MolNexus CGI prediction.")
    parser.add_argument("--project-root", help="Project root containing cgi/ and model/cgi/.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--compound-features", required=True, help="Real compound KPGT feature table.")
    parser.add_argument("--model-dir", default="model/cgi")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-format", choices=["triplet", "kpgt_features", "auto"], default="auto")
    parser.add_argument("--label-col")
    parser.add_argument("--compound-id-col")
    parser.add_argument("--gene-id-col")
    parser.add_argument("--compound-name-col")
    parser.add_argument("--gene-name-col")
    parser.add_argument("--compound-feature-prefix")
    parser.add_argument("--gene-feature-prefix")
    parser.add_argument("--compound-feature-cols")
    parser.add_argument("--gene-feature-cols")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--delimiter", default="auto")
    parser.add_argument("--decoder")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-save-fold-probs", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run inference. Without this flag only validate and print the command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root(args.project_root)
    require_file("cgi/predict.py", label="cgi_predict_module", root=root)
    input_csv = require_file(args.input_csv, label="input_csv", root=root)
    compound_features = require_file(
        args.compound_features, label="compound_features", root=root
    )
    model_dir = require_dir(args.model_dir, label="model_dir", root=root)
    output_dir = local_path(args.output_dir, root=root)
    checkpoints = sorted(model_dir.rglob("*.pt"))
    if not checkpoints:
        fail(
            "No CGI checkpoint files were found under model_dir.",
            code="missing_checkpoints",
            details={"model_dir": str(model_dir)},
        )

    command = [
        sys.executable,
        "-m",
        "cgi.predict",
        "--input_csv",
        str(input_csv),
        "--cpd_features_file",
        str(compound_features),
        "--model_dir",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--input_format",
        args.input_format,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--delimiter",
        args.delimiter,
        "--num_heads",
        str(args.num_heads),
        "--dropout",
        str(args.dropout),
    ]
    optional_values = {
        "--label_col": args.label_col,
        "--compound_id_col": args.compound_id_col,
        "--gene_id_col": args.gene_id_col,
        "--compound_name_col": args.compound_name_col,
        "--gene_name_col": args.gene_name_col,
        "--compound_feature_prefix": args.compound_feature_prefix,
        "--gene_feature_prefix": args.gene_feature_prefix,
        "--compound_feature_cols": args.compound_feature_cols,
        "--gene_feature_cols": args.gene_feature_cols,
        "--decoder": args.decoder,
    }
    for flag, value in optional_values.items():
        add_optional(command, flag, value)
    command.append("--no_save_fold_probs" if args.no_save_fold_probs else "--save_fold_probs")

    response = {
        "task": "cgi",
        "classes": ["CGI-N", "CGI-D", "CGI-U"],
        "checkpoint_count": len(checkpoints),
        "output_dir": str(output_dir),
        "result_files": {
            "predictions": str(output_dir / "predictions.csv"),
            "classification_report": str(output_dir / "classification_report.csv"),
            "confusion_matrix": str(output_dir / "confusion_matrix.csv"),
        },
        "warnings": [
            "The wrapper never enables random missing-feature fallback.",
            "Computational CGI predictions require independent validation.",
        ],
    }
    return run_or_dry_run(
        command=command,
        root=root,
        execute=args.execute,
        response=response,
        env_updates={"PYTHONHASHSEED": "42"},
    )


if __name__ == "__main__":
    raise SystemExit(main())
