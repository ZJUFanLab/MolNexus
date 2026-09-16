from __future__ import annotations

import logging
import os
import traceback
import pandas as pd


def merge_hgt_cmi_summaries_if_ready(
    folder,
    folds=5,
    run_tag="CMI_finetune",
    model_name="HGT_CMI_CCS_finetune",
):
    """
    Automatically merge when all five fold summaries exist.
    Within a SLURM array, each fold attempts the merge upon completion;
    only the last task to finish and acquire the lock performs the merge.
    """
    os.makedirs(folder, exist_ok=True)

    summary_files = []
    for fold in range(1, folds + 1):
        fname = f"HGT_CMI_CCS_5fold_fold{fold}_summary_{run_tag}.csv"
        path = os.path.join(folder, fname)

        if not os.path.isfile(path):
            logging.info(f"[MERGE] Not ready: missing {path}")
            return False

        summary_files.append(path)

    lock_path = os.path.join(folder, f".merge_{run_tag}.lock")

    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(lock_fd)
    except FileExistsError:
        logging.info(f"[MERGE] Another process is merging or already merged: {lock_path}")
        return False

    try:
        dfs = []

        for fold, path in enumerate(summary_files, start=1):
            df = pd.read_csv(path)

            if "fold" not in df.columns:
                df.insert(0, "fold", fold)

            dfs.append(df)
            logging.info(f"[MERGE] Loaded fold {fold}: {path}")

        all_df = pd.concat(dfs, ignore_index=True)

        numeric_cols = [
            c for c in all_df.columns
            if c != "fold" and pd.api.types.is_numeric_dtype(all_df[c])
        ]

        means = all_df[numeric_cols].mean()
        stds = all_df[numeric_cols].std()

        summary_row = {"fold": "mean±SD"}

        for col in all_df.columns:
            if col == "fold":
                continue
            elif col in numeric_cols:
                summary_row[col] = f"{means[col]:.4f}±{stds[col]:.4f}"
            else:
                summary_row[col] = ""

        merged_df = pd.concat(
            [all_df, pd.DataFrame([summary_row])],
            ignore_index=True
        )

        merged_out_path = os.path.join(
            folder,
            f"HGT_CMI_CCS_5fold_merged_summary_{run_tag}.csv"
        )
        merged_df.to_csv(merged_out_path, index=False)
        logging.info(f"[MERGE] Saved merged summary to: {merged_out_path}")

        flat_row = {
            "model": model_name,
            "run_tag": run_tag,
        }

        for col in numeric_cols:
            flat_row[f"{col}_mean"] = means[col]
            flat_row[f"{col}_std"] = stds[col]

        flat_df = pd.DataFrame([flat_row])

        flat_out_path = os.path.join(
            folder,
            f"HGT_CMI_CCS_5fold_summary_mean_std_flat_{run_tag}.csv"
        )
        flat_df.to_csv(flat_out_path, index=False, float_format="%.4f")
        logging.info(f"[MERGE] Saved flat summary to: {flat_out_path}")

        return True

    except Exception:
        logging.error("[MERGE] Failed to merge fold summaries.")
        logging.error(traceback.format_exc())
        return False
