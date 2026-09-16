#!/usr/bin/env python3
"""Validate and delegate to the independent CPI prediction package."""

from __future__ import annotations

import argparse
import sys

from _runtime import (
    fail,
    local_path,
    optional_file,
    project_root,
    require_file,
    run_or_dry_run,
)


SPLIT_ALIASES = {"warm": "Warm", "ccs": "CCS", "dcs": "DCS", "pcs": "PCS"}


def normalize_split(value: str) -> str:
    split = SPLIT_ALIASES.get(value.strip().lower())
    if not split:
        fail(
            "Unsupported CPI split.",
            code="invalid_split",
            details={"split": value, "supported": list(SPLIT_ALIASES.values())},
        )
    return split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or execute MolNexus CPI prediction.")
    parser.add_argument("--project-root", help="Project root containing cpi/ and model/cpi/.")
    parser.add_argument("--split", required=True, help="Warm, CCS, DCS, or PCS.")
    parser.add_argument(
        "--task-mode",
        choices=["target_prediction", "compound_screening"],
        default="target_prediction",
    )
    parser.add_argument("--compounds", required=True, help="Prepared compound list/triplet CSV or TSV.")
    parser.add_argument("--feat", required=True, help="Real external compound feature CSV or TSV.")
    parser.add_argument("--cgi", help="Optional external CGI file used to expand the inference graph.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run inference. Without this flag only validate and print the command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root(args.project_root)
    require_file("cpi/predict.py", label="cpi_predict_module", root=root)
    split = normalize_split(args.split)
    if args.ensemble_size < 1:
        fail("--ensemble-size must be at least 1.", code="invalid_ensemble_size")

    compounds = require_file(args.compounds, label="compounds", root=root)
    feat = require_file(args.feat, label="feat", root=root)
    cgi = optional_file(args.cgi, label="cgi", root=root)
    candidate_file = require_file(args.candidate_file, label="candidate_file", root=root)
    output_dir = local_path(args.out_dir, root=root)

    checkpoints = []
    for fold in range(1, args.ensemble_size + 1):
        checkpoints.append(
            require_file(
                f"model/cpi/{split}/ensemble_{fold}.pt",
                label=f"checkpoint_fold_{fold}",
                root=root,
            )
        )

    command = [
        sys.executable,
        "-m",
        "cpi.predict",
        "--task_mode",
        args.task_mode,
        "--split",
        split,
        "--ensemble_size",
        str(args.ensemble_size),
        "--compounds",
        str(compounds),
        "--feat",
        str(feat),
        "--out_dir",
        str(output_dir),
        "--candidate_file",
        str(candidate_file),
    ]
    if cgi:
        command.extend(["--cgi", str(cgi)])

    response = {
        "task": "cpi",
        "split": split,
        "task_mode": args.task_mode,
        "checkpoint_count": len(checkpoints),
        "output_dir": str(output_dir),
        "result_files": {
            "probability": str(output_dir / "ensemble_prob_matrix.csv"),
            "rank": str(output_dir / "ensemble_rank_matrix.csv"),
            "label": str(output_dir / "ensemble_label_matrix.csv"),
            "unanimous_vote": str(output_dir / "ensemble_label_vote5_matrix.csv"),
        },
        "warnings": ["Computational CPI predictions require independent validation."],
    }
    return run_or_dry_run(
        command=command,
        root=root,
        execute=args.execute,
        response=response,
        env_updates={"TRAIN_MODE": split, "PREDICT_SPLIT": split},
    )


if __name__ == "__main__":
    raise SystemExit(main())
