# train.py

import argparse
import logging
import os
from pathlib import Path
import traceback

from .config import (
    ABLATION_MODES,
    DEFAULT_LOG_FILE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_MODE,
    TRAIN_MODE_CONFIGS,
)

def _legacy_main():
    import pandas as pd
    from .trainer import FinalTrainer
    from .utils import set_seed, setup_logging
    set_seed(DEFAULT_SEED)

    train_mode = os.environ.get("TRAIN_MODE", DEFAULT_TRAIN_MODE)
    project_root = Path(__file__).resolve().parent.parent
    mode_cfg = TRAIN_MODE_CONFIGS[train_mode]

    log_file = mode_cfg.get("log_file", DEFAULT_LOG_FILE)
    setup_logging(log_file)

    fold = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"=== Starting fold {fold} ===")

    train_csv = project_root / TRAIN_MODE_CONFIGS[train_mode]["train_csv_template"].format(fold=fold)
    val_csv   = project_root / TRAIN_MODE_CONFIGS[train_mode]["val_csv_template"].format(fold=fold)

    try:
        trainer = FinalTrainer(
            train_target_file=train_csv,
            val_target_file=val_csv,
            ppi_file="demo/BioNexKG_demo/ppi.csv",
            mpi_file="demo/BioNexKG_demo/mpi.csv",
            gpi_file="demo/BioNexKG_demo/gpi.csv",
            ggi_file="demo/BioNexKG_demo/ggi.csv",
            cgi_file="demo/BioNexKG_demo/cgi.csv",
            cpd_features_file="demo/feat_demo_subset/demo_compounds_feat.csv",
            protein_features_file="demo/feat_demo_subset/demo_proteins_feat.csv",
            metabolite_features_file="demo/feat_demo_subset/demo_metabolites_feat.csv",
            gene_features_file="demo/feat_demo_subset/demo_genes_feat.csv",
        )

        print(f"\n=== Resuming fold {fold} training ===")
        results = trainer.train_and_test(fold, resume=True)

        summary_df = pd.DataFrame([{"fold": fold, **results}])
        out_name = f"{trainer.summary_prefix}_fold{fold}_summary.csv"
        summary_df.to_csv(out_name, index=False)
        print(f"CV5 summary saved to {out_name}")

    except Exception:
        logging.error("Execution failed")
        logging.error(traceback.format_exc())


def build_parser():
    parser = argparse.ArgumentParser(description="Train the independent CPI package.")
    parser.add_argument("--fold", type=int, default=None, help="Fold number; defaults to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--train-mode", choices=sorted(TRAIN_MODE_CONFIGS), default=None, help="Alias for TRAIN_MODE.")
    parser.add_argument(
        "--ablation-mode",
        choices=sorted(ABLATION_MODES),
        default=None,
        help="Background-graph ablation mode.",
    )
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fold is not None:
        os.environ["SLURM_ARRAY_TASK_ID"] = str(args.fold)
    if args.train_mode is not None:
        os.environ["TRAIN_MODE"] = args.train_mode
    if args.ablation_mode is not None:
        os.environ["ABLATION_MODE"] = args.ablation_mode
    return _legacy_main()


if __name__ == "__main__":
    main()
