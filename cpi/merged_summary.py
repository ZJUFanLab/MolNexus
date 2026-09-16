#!/usr/bin/env python3
# merge_summaries_multi.py

import os
import pandas as pd

def merge_summaries_multi(folder, scenario_prefix_map, folds=5, output_filename="merged_summary_all_scenarios.csv"):
    """
    Merge fold1~foldN summary.csv files for multiple prefixes in folder,
    add a Scenario column, and append a mean±SD row for each Scenario.
    
    Parameters
    ----
    folder : str
        Path to the folder containing the CSV files
    scenario_prefix_map : dict
        Mapping from scenario names to filename prefixes, for example:
        {
            "Warm": "Warm",
            "CCS":  "CCS",
            "PCS":  "PCS",
            "DCS":  "DCS",
        }
    folds : int
        Number of folds; default: 5
    output_filename : str
        Output filename
    """
    all_results = []

    for scenario, prefix in scenario_prefix_map.items():
        dfs = []

        for fold in range(1, folds + 1):
            fname = f"{prefix}_fold{fold}_summary.csv"
            path = os.path.join(folder, fname)

            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")

            df = pd.read_csv(path)

            # Add the Scenario column
            df["Scenario"] = scenario

            dfs.append(df)
            print(f"Loaded [{scenario}] fold {fold}: {fname}")

        # Merge all folds for the current scenario
        scenario_df = pd.concat(dfs, ignore_index=True)

        # Numeric columns: compute mean/std only for actual numeric columns
        numeric_cols = scenario_df.select_dtypes(include="number").columns.tolist()

        # Construct the mean±SD summary row for the current scenario
        summary_row = {}

        # Retain all columns, initially setting them to empty
        for col in scenario_df.columns:
            summary_row[col] = ""

        # Set the identifier columns
        if "fold" in scenario_df.columns:
            summary_row["fold"] = "mean±SD"
        summary_row["Scenario"] = scenario

        # Fill numeric columns with mean±SD
        means = scenario_df[numeric_cols].mean()
        stds = scenario_df[numeric_cols].std()

        for col in numeric_cols:
            summary_row[col] = f"{means[col]:.4f}±{stds[col]:.4f}"

        summary_df = pd.DataFrame([summary_row])

        # Current scenario data + summary row
        scenario_result = pd.concat([scenario_df, summary_df], ignore_index=True)
        all_results.append(scenario_result)

    # Merge all scenarios
    result_df = pd.concat(all_results, ignore_index=True)

    # Optional: move the Scenario column to the front
    cols = result_df.columns.tolist()
    if "Scenario" in cols:
        cols.insert(0, cols.pop(cols.index("Scenario")))
        result_df = result_df[cols]

    # Save
    out_path = os.path.join(folder, output_filename)
    result_df.to_csv(out_path, index=False)
    print(f"Saved merged summary to: {out_path}")


if __name__ == "__main__":
    folder = "."
    scenario_prefix_map = {
        "Warm": "Warm",
        "CCS":  "CCS",
        "PCS":  "PCS",
        "DCS":  "DCS",
    }

    merge_summaries_multi(
        folder=folder,
        scenario_prefix_map=scenario_prefix_map,
        folds=5,
        output_filename="merged_summary_all_scenarios.csv"
    )
