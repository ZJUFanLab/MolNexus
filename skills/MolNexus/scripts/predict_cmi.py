#!/usr/bin/env python3
"""Validate and delegate to the independent CMI prediction package."""

from __future__ import annotations

import argparse
import sys

from _runtime import (
    fail,
    local_path,
    optional_file,
    project_root,
    require_dir,
    require_file,
    run_or_dry_run,
)


def add_optional(command: list[str], flag: str, value: str | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or execute MolNexus CMI prediction.")
    parser.add_argument("--project-root", help="Project root containing cmi/ and model/cmi/.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model-dir", default="model/cmi")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-format", choices=["triplet", "kpgt_features", "auto"], default="auto")
    parser.add_argument("--label-col")
    parser.add_argument("--compound-id-col")
    parser.add_argument("--metabolite-id-col")
    parser.add_argument("--compound-name-col")
    parser.add_argument("--metabolite-name-col")
    parser.add_argument("--compound-feature-prefix")
    parser.add_argument("--metabolite-feature-prefix")
    parser.add_argument("--compound-feature-cols")
    parser.add_argument("--metabolite-feature-cols")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--delimiter", default="auto")
    parser.add_argument("--model-glob", default="*.pt")
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--compound-features", help="Override the base compound feature table.")
    parser.add_argument("--external-compound-features", help="Real extra compound feature table.")
    parser.add_argument("--external-metabolite-features", help="Real extra metabolite feature table.")
    parser.add_argument("--external-compound-feature-id-col")
    parser.add_argument("--external-metabolite-feature-id-col")
    parser.add_argument("--external-feature-delimiter", default="auto")
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
    require_file("cmi/predict.py", label="cmi_predict_module", root=root)
    input_csv = require_file(args.input_csv, label="input_csv", root=root)
    model_dir = require_dir(args.model_dir, label="model_dir", root=root)
    output_dir = local_path(args.output_dir, root=root)
    compound_features = optional_file(
        args.compound_features, label="compound_features", root=root
    )
    external_compound = optional_file(
        args.external_compound_features,
        label="external_compound_features",
        root=root,
    )
    external_metabolite = optional_file(
        args.external_metabolite_features,
        label="external_metabolite_features",
        root=root,
    )

    checkpoints = sorted(model_dir.glob(args.model_glob))
    if not checkpoints:
        fail(
            "No CMI checkpoint files matched --model-glob.",
            code="missing_checkpoints",
            details={"model_dir": str(model_dir), "model_glob": args.model_glob},
        )
    if args.expected_folds > 0 and len(checkpoints) != args.expected_folds:
        fail(
            "CMI checkpoint count does not match --expected-folds.",
            code="unexpected_checkpoint_count",
            details={"expected": args.expected_folds, "found": len(checkpoints)},
        )

    command = [
        sys.executable,
        "-m",
        "cmi.predict",
        "--input_csv",
        str(input_csv),
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
        "--model_glob",
        args.model_glob,
        "--expected_folds",
        str(args.expected_folds),
        "--max_rows",
        str(args.max_rows),
        "--external_feature_delimiter",
        args.external_feature_delimiter,
    ]
    optional_values = {
        "--label_col": args.label_col,
        "--compound_id_col": args.compound_id_col,
        "--metabolite_id_col": args.metabolite_id_col,
        "--compound_name_col": args.compound_name_col,
        "--metabolite_name_col": args.metabolite_name_col,
        "--compound_feature_prefix": args.compound_feature_prefix,
        "--metabolite_feature_prefix": args.metabolite_feature_prefix,
        "--compound_feature_cols": args.compound_feature_cols,
        "--metabolite_feature_cols": args.metabolite_feature_cols,
        "--compound_features_file": str(compound_features) if compound_features else None,
        "--external_compound_features_file": str(external_compound) if external_compound else None,
        "--external_metabolite_features_file": str(external_metabolite) if external_metabolite else None,
        "--external_compound_feature_id_col": args.external_compound_feature_id_col,
        "--external_metabolite_feature_id_col": args.external_metabolite_feature_id_col,
    }
    for flag, value in optional_values.items():
        add_optional(command, flag, value)
    command.append("--no_save_fold_probs" if args.no_save_fold_probs else "--save_fold_probs")

    response = {
        "task": "cmi",
        "classes": ["CMI-N", "CMI-D", "CMI-U"],
        "checkpoint_count": len(checkpoints),
        "output_dir": str(output_dir),
        "result_files": {
            "predictions": str(output_dir / "predictions.csv"),
            "classification_report": str(output_dir / "classification_report.csv"),
            "confusion_matrix": str(output_dir / "confusion_matrix.csv"),
        },
        "warnings": [
            "The wrapper never enables random missing-feature fallback.",
            "Feature coverage is enforced by the CMI package during execution.",
            "Computational CMI predictions require independent validation.",
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
