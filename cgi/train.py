from __future__ import annotations

import argparse
import logging
import os
import traceback
import pandas as pd
from .config import DEFAULT_PATHS

def build_parser():
    parser = argparse.ArgumentParser(description="Train the independent CGI package.")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--train-target-file", default=None)
    parser.add_argument("--val-target-file", default=None)
    parser.add_argument("--ppi-file", default=DEFAULT_PATHS["ppi_file"])
    parser.add_argument("--mpi-file", default=DEFAULT_PATHS["mpi_file"])
    parser.add_argument("--gpi-file", default=DEFAULT_PATHS["gpi_file"])
    parser.add_argument("--ggi-file", default=DEFAULT_PATHS["ggi_file"])
    parser.add_argument("--cpi-pos-file", default=DEFAULT_PATHS["cpi_pos_file"])
    parser.add_argument("--cpd-features-file", default=DEFAULT_PATHS["cpd_features_file"])
    parser.add_argument("--protein-features-file", default=DEFAULT_PATHS["protein_features_file"])
    parser.add_argument("--metabolite-features-file", default=DEFAULT_PATHS["metabolite_features_file"])
    parser.add_argument("--gene-features-file", default=DEFAULT_PATHS["gene_features_file"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "HGT_results"))
    parser.add_argument("--run-tag", default=os.environ.get("RUN_TAG", "CGI_finetune"))
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)

    fold = args.fold
    if fold is None:
        raw_fold = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("FOLD")
        if raw_fold is None:
            build_parser().error("--fold or SLURM_ARRAY_TASK_ID/FOLD is required")
        fold = int(raw_fold)
    train_template = DEFAULT_PATHS["train_template"]
    val_template = DEFAULT_PATHS["val_template"]
    train_file = args.train_target_file or train_template.format(fold=fold)
    val_file = args.val_target_file or val_template.format(fold=fold)
    if args.resume is not None:
        os.environ["RESUME"] = "1" if args.resume else "0"

    from .trainer import FinalTrainer, try_merge_hgt_cgi_summaries as merge_summary
    try:
        trainer = FinalTrainer(
            train_target_file=train_file,
            val_target_file=val_file,
            ppi_file=args.ppi_file,
            mpi_file=args.mpi_file,
            gpi_file=args.gpi_file,
            ggi_file=args.ggi_file,
            cpi_pos_file=args.cpi_pos_file,
            cpd_features_file=args.cpd_features_file,
            protein_features_file=args.protein_features_file,
            metabolite_features_file=args.metabolite_features_file,
            gene_features_file=args.gene_features_file,
        )
        resume = os.environ.get("RESUME", "0") == "1"
        results = trainer.train_and_test(fold, resume=resume)
        os.makedirs(args.results_dir, exist_ok=True)
        out = os.path.join(args.results_dir, f"HGT_CGI_CCS_5fold_fold{fold}_summary_{args.run_tag}.csv")
        pd.DataFrame([{"fold": fold, **results}]).to_csv(out, index=False)
        merged = merge_summary(folder=args.results_dir, folds=5, run_tag=args.run_tag)
        return 0
    except Exception:
        logging.error("Execution failed")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
