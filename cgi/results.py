from __future__ import annotations

import logging
import os
import traceback
import pandas as pd


def try_merge_hgt_cgi_summaries(
    folder,
    folds=5,
    run_tag=None,
    output_merged=None,
    output_flat=None,
):
    os.makedirs(folder, exist_ok=True)

    if run_tag is None or str(run_tag).strip() == "":
        run_tag = os.environ.get("RUN_TAG", "CGI_finetune")

    if output_merged is None:
        output_merged = f"HGT_CGI_CCS_5fold_merged_summary_{run_tag}.csv"

    if output_flat is None:
        output_flat = f"HGT_CGI_CCS_5fold_summary_mean_std_flat_{run_tag}.csv"

    expected_files = [
        os.path.join(
            folder,
            f"HGT_CGI_CCS_5fold_fold{fold}_summary_{run_tag}.csv"
        )
        for fold in range(1, folds + 1)
    ]

    missing = [p for p in expected_files if not os.path.isfile(p)]
    if missing:
        logging.info(
            f"[MERGE] Not all fold summaries are ready. "
            f"Missing {len(missing)}/{folds}. Skip merge for now."
        )
        for p in missing:
            logging.info(f"[MERGE] Missing: {p}")
        return False

    lock_file = os.path.join(folder, f".merge_HGT_CGI_CCS_5fold_{run_tag}.lock")

    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        logging.info("[MERGE] Another job is merging or has just started merging. Skip.")
        return False

    try:
        dfs = []

        for fold in range(1, folds + 1):
            path = os.path.join(
                folder,
                f"HGT_CGI_CCS_5fold_fold{fold}_summary_{run_tag}.csv"
            )

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

        merged_out_path = os.path.join(folder, output_merged)
        merged_df.to_csv(merged_out_path, index=False)
        logging.info(f"[MERGE] Saved merged summary to: {merged_out_path}")

        flat_row = {
            "model": "HGT_CGI_CCS",
            "run_tag": run_tag,
        }

        for col in numeric_cols:
            flat_row[f"{col}_mean"] = means[col]
            flat_row[f"{col}_std"] = stds[col]

        flat_df = pd.DataFrame([flat_row])

        flat_out_path = os.path.join(folder, output_flat)
        flat_df.to_csv(flat_out_path, index=False, float_format="%.4f")
        logging.info(f"[MERGE] Saved flat summary to: {flat_out_path}")
        logging.info(f"\n[MERGE] Summary flat:\n{flat_df.round(4)}")

        return True

    except Exception:
        logging.error("[MERGE] Failed to merge CGI fold summaries.")
        logging.error(traceback.format_exc())
        return False

    finally:
        try:
            os.remove(lock_file)
        except OSError:
            pass
