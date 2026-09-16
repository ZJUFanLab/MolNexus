# predict.py

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

# Allow both `python -m cpi.predict` and direct script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "cpi"

from .config import (
    CGI_GLOBAL,
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PREDICT_SPLIT,
    DEFAULT_SEED,
    SUPPORTED_TRAIN_MODES,
    TRAIN_MODE_CONFIGS,
    normalize_ablation_mode,
)
from .inference_builder import load_inference_ckpt, resolve_cpi_threshold
#set_seed(328)
#set_seed(221)
#set_seed(215)
#set_seed(9524)
#set_seed(1212)

def collect_fold_records(summary_csv, ckpt_dir):
    """
    Returns:
      [
        {"fold": 1, "best_epoch": 123 or None},
        {"fold": 2, "best_epoch": None},
        ...
      ]
    """
    records = []

    if summary_csv is not None:
        df_sum = pd.read_csv(summary_csv)
        df_sum = df_sum[df_sum["fold"].apply(lambda x: str(x).isdigit())]

        for _, row in df_sum.iterrows():
            fold = int(row["fold"])
            best_epoch = None
            if "Best Epoch" in df_sum.columns and pd.notna(row["Best Epoch"]):
                best_epoch = int(row["Best Epoch"])
            records.append({"fold": fold, "best_epoch": best_epoch})
        return records

    # When summary_csv is not provided, automatically scan fold* under ckpt_dir
    fold_dirs = []
    for name in os.listdir(ckpt_dir):
        full_path = os.path.join(ckpt_dir, name)
        if os.path.isdir(full_path) and name.startswith("fold"):
            suffix = name[4:]
            if suffix.isdigit():
                fold_dirs.append(int(suffix))

    fold_dirs = sorted(fold_dirs)
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No fold directories found under ckpt_dir: {ckpt_dir}")

    for fold in fold_dirs:
        records.append({"fold": fold, "best_epoch": None})

    return records


def build_checkpoint_paths(project_root, split, ensemble_size):
    model_dir = project_root / DEFAULT_MODEL_ROOT / "cpi" / split
    return [model_dir / f"ensemble_{idx}.pt" for idx in range(1, ensemble_size + 1)]


def validate_checkpoint_paths(checkpoint_paths, project_root):
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            rel_path = checkpoint_path.relative_to(project_root)
            raise FileNotFoundError(f"Missing checkpoint: {rel_path}")

def main():
    #set_seed(328)
    #set_seed(221)
    #set_seed(215)
    #set_seed(9524)    
    #set_seed(1212)
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--task_mode",
    choices=["target_prediction", "compound_screening"],
    default="target_prediction",
    help=(
        "Control the orientation of the unlabeled aggregate matrix: "
        "target_prediction=compound columns and protein rows; "
        "compound_screening=protein columns and compound rows"
    ),
)

    parser.add_argument("--summary_csv", default=None, help="merged_summary.csv (optional)")
    parser.add_argument("--ckpt_dir", default=None, help="Checkpoint root directory (optional)")
    parser.add_argument("--split", choices=sorted(SUPPORTED_TRAIN_MODES), default=None, help="Prediction-weight split; defaults to the configuration or TRAIN_MODE")
    parser.add_argument("--ensemble_size", type=int, default=DEFAULT_ENSEMBLE_SIZE, help="Number of ensemble weights")
    parser.add_argument("--cgi", default=None, help="External CGI file")
    parser.add_argument("--feat", required=True, help="External compound feature file")
    parser.add_argument("--out_dir", default="itcm_results")
    parser.add_argument("--out_name", default="ITCM_preds_summary.csv")

    parser.add_argument(
        "--compounds",
        default=None,
        help="For --mode=unlabeled, provide a compound-list file (CSV/TSV) with a 'compound' column or the ('head', 'head_type') columns",
    )
    parser.add_argument(
        "--candidate_file",
        required=True,
        help="CSV/TSV with a 'tail' column (using tail_type == 'protein' when present) or a 'protein' column",
    )

    args = parser.parse_args()
    ablation_mode = "FULL"
    os.environ["ABLATION_MODE"] = ablation_mode
    global pd, torch, FinalTrainer
    import pandas as pd
    import torch
    from .trainer import FinalTrainer
    from .utils import set_seed
    set_seed(DEFAULT_SEED)
    set_seed(DEFAULT_SEED)

    project_root = Path(__file__).resolve().parent.parent
    split = args.split or os.environ.get("PREDICT_SPLIT") or os.environ.get("TRAIN_MODE") or DEFAULT_PREDICT_SPLIT
    if split not in TRAIN_MODE_CONFIGS:
        raise ValueError(f"Unknown split={split}, supported={list(TRAIN_MODE_CONFIGS.keys())}")
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")

    # FinalTrainer and HGT read TRAIN_MODE while constructing the model.
    # Synchronize it with --split so each checkpoint uses matching dimensions.
    os.environ["TRAIN_MODE"] = split

    checkpoint_paths = build_checkpoint_paths(project_root, split, args.ensemble_size)
    validate_checkpoint_paths(checkpoint_paths, project_root)
    fold_records = [
        {"fold": idx, "ckpt_path": checkpoint_path}
        for idx, checkpoint_path in enumerate(checkpoint_paths, start=1)
    ]

    if not args.compounds:
        parser.error("--compounds is required")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[PREDICT] split={split}")
    print(f"[PREDICT] ensemble_size={args.ensemble_size}")
    print("[PREDICT] checkpoint_paths:")
    for checkpoint_path in checkpoint_paths:
        print(f"  - {checkpoint_path.relative_to(project_root)}")
    print(f"[PREDICT] folds={ [x['fold'] for x in fold_records] }")

    metrics_list = []

    vote_pos = defaultdict(int)
    vote_seen = defaultdict(int)

    pair_sum = defaultdict(float)
    pair_cnt = defaultdict(int)
    all_compounds = set()
    all_proteins = set()
    fold_thrs = []
    missing_thresholds = []

    for rec in fold_records:
        fold = rec["fold"]
        ckpt_path = rec["ckpt_path"]

        print(f"[Fold {fold}] ckpt_path = {ckpt_path}")

        trainer = FinalTrainer(
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

        _ckpt_meta = load_inference_ckpt(str(ckpt_path), "cpu")
        _abl = normalize_ablation_mode(ablation_mode)
        # CPI prediction always uses the production/global CGI background.
        # The checkpoint metadata remains available for compatibility, but
        # cannot change the prediction graph.
        _cgi = CGI_GLOBAL
        print(
            f"[Fold {fold}] ablation_mode={_abl} | cgi_mode={_cgi} | "
            f"include CGI? {'CGI' in trainer.get_included_bg_rels(_abl)}"
        )

        df_all = trainer.predict_unlabeled_all(
            compound_file=args.compounds,
            external_cpd_features_file=args.feat,
            checkpoint_path=ckpt_path,
            candidate_file=args.candidate_file,
            ablation_mode=ablation_mode,
            cgi_mode=CGI_GLOBAL,
            external_cgi_file=args.cgi,
        )

        for c, p, pr in zip(
            df_all["compound"].astype(str),
            df_all["protein"].astype(str),
            df_all["prob"].astype(float),
        ):
            pair_sum[(c, p)] += pr
            pair_cnt[(c, p)] += 1
            all_compounds.add(c)
            all_proteins.add(p)

        try:
            fold_thr = resolve_cpi_threshold(_ckpt_meta, checkpoint_path=str(ckpt_path))
        except ValueError:
            missing_thresholds.append(str(ckpt_path))
            fold_thr = None

        if fold_thr is not None:
            fold_thrs.append(fold_thr)
            for c, p, pr in zip(
                df_all["compound"].astype(str),
                df_all["protein"].astype(str),
                df_all["prob"].astype(float),
            ):
                vote_seen[(c, p)] += 1
                if pr >= fold_thr:
                    vote_pos[(c, p)] += 1

    # ========= Output =========
    print("\n=== Unlabeled mode: online aggregation complete; generating matrices ===")

    recs = []
    for (c, p), s in pair_sum.items():
        n = pair_cnt[(c, p)]
        recs.append((c, p, s / max(n, 1)))
    grp = pd.DataFrame(recs, columns=["compound", "protein", "prob_mean"])

    all_compounds = sorted(all_compounds)
    all_proteins = sorted(all_proteins)

    # =========================================================
    # Determine matrix orientation from task_mode
    # target_prediction:   columns=compound, rows=protein
    # compound_screening: columns=protein, rows=compound
    # =========================================================
    if args.task_mode == "target_prediction":
        row_name = "protein"
        col_name = "compound"
        row_values = all_proteins
        col_values = all_compounds

        prob_mat = grp.pivot(index="protein", columns="compound", values="prob_mean").reindex(
            index=row_values,
            columns=col_values,
        )

    else:  # compound_screening
        row_name = "compound"
        col_name = "protein"
        row_values = all_compounds
        col_values = all_proteins

        prob_mat = grp.pivot(index="compound", columns="protein", values="prob_mean").reindex(
            index=row_values,
            columns=col_values,
        )

    prob_mat.index.name = row_name
    prob_mat.columns.name = None
    prob_mat_out = prob_mat.round(6)

    out_prob = os.path.join(args.out_dir, "ensemble_prob_matrix.csv")
    prob_mat_out.to_csv(out_prob)
    print(f"[Ensemble] Probability matrix ➔ {out_prob}")
    print(f"[Ensemble] prob_matrix shape: {row_name}s={prob_mat_out.shape[0]}, {col_name}s={prob_mat_out.shape[1]}")

    # Rank matrix: rank by column by default (descending within each column)
    prob_for_rank = prob_mat.fillna(-1.0)
    rank_mat = prob_for_rank.rank(ascending=False, method="min").astype("Int64")
    rank_mat.index.name = row_name
    rank_mat.columns.name = None

    out_rank = os.path.join(args.out_dir, "ensemble_rank_matrix.csv")
    rank_mat.to_csv(out_rank)
    print(f"[Ensemble] Rank matrix ➔ {out_rank}")
    print(f"[Ensemble] rank_matrix shape: {row_name}s={rank_mat.shape[0]}, {col_name}s={rank_mat.shape[1]}")

    if missing_thresholds:
        missing = ", ".join(missing_thresholds)
        raise ValueError(
            "Probability and rank outputs were saved, but CPI binary label/vote "
            f"generation requires threshold configuration for: {missing}. "
            "Provide a same-stem JSON sidecar for each weights-only checkpoint."
        )

    mean_thr = pd.Series(fold_thrs).mean()
    label_mat = (prob_mat >= float(mean_thr)).astype("Int64")
    label_mat.index.name = row_name
    label_mat.columns.name = None

    out_label = os.path.join(args.out_dir, "ensemble_label_matrix.csv")
    label_mat.to_csv(out_label)
    print(f"[Ensemble] Label matrix (thr={float(mean_thr):.4f}) ➔ {out_label}")
    print(f"[Ensemble] label_matrix shape: {row_name}s={label_mat.shape[0]}, {col_name}s={label_mat.shape[1]}")

    nfolds = len(fold_thrs)
    vote5_mat = pd.DataFrame(
        0,
        index=row_values,
        columns=col_values,
        dtype="Int64",
    )
    vote5_mat.index.name = row_name
    vote5_mat.columns.name = None

    for (c, p), pos_cnt in vote_pos.items():
        if vote_seen.get((c, p), 0) == nfolds and pos_cnt == nfolds:
            if args.task_mode == "target_prediction":
                # rows=protein, columns=compound
                vote5_mat.at[p, c] = 1
            else:
                # rows=compound, columns=protein
                vote5_mat.at[c, p] = 1

    out_vote5 = os.path.join(args.out_dir, "ensemble_label_vote5_matrix.csv")
    vote5_mat.to_csv(out_vote5)
    print(f"[Ensemble] Consensus label matrix (vote5; nfolds={nfolds}) ➔ {out_vote5}")
    print(f"[Ensemble] vote5_matrix shape: {row_name}s={vote5_mat.shape[0]}, {col_name}s={vote5_mat.shape[1]}")


if __name__ == "__main__":
    main()
