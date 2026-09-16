from __future__ import annotations

import csv
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .config import COMMON_META_COLUMNS, LABEL_CANDIDATES

# Fixed relations formerly discovered from the legacy six-relation TSV.
DECODER_BASE_RELATIONS = (
    "CGI-D",
    "CGI-U",
    "GGI",
    "GPI",
    "MPI",
    "PPI",
)


def build_relation_id_map(*relation_groups):
    """Build the checkpoint-compatible, lexicographically sorted relation map."""
    relations = {relation for group in relation_groups for relation in group}
    return {relation: index for index, relation in enumerate(sorted(relations))}


def log_ram(tag=""):
    import logging
    import os
    import psutil
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    logging.info(f"[MEM]{f' {tag}' if tag else ''} RSS={rss:.2f} GB")


def normalize_delimiter(delimiter: str) -> str | None:
    if delimiter is None or delimiter.lower() == "auto":
        return None
    lookup = {
        "tab": "\t",
        "\\t": "\t",
        "tsv": "\t",
        "comma": ",",
        "csv": ",",
    }
    return lookup.get(delimiter, delimiter)


def sniff_delimiter(path: str, delimiter: str) -> str:
    explicit = normalize_delimiter(delimiter)
    if explicit is not None:
        return explicit
    with open(path, "r", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t" if str(path).lower().endswith((".tsv", ".txt")) else ","


def read_input_table(path: str, delimiter: str, max_rows: int):
    sep = sniff_delimiter(path, delimiter)
    nrows = max_rows if max_rows and max_rows > 0 else None
    df = pd.read_csv(path, sep=sep, dtype=str, nrows=nrows)
    if df.empty:
        raise ValueError(f"Input file has no rows: {path}")
    logging.info("Loaded input: %s rows, %s columns, sep=%r", len(df), len(df.columns), sep)
    return df


def parse_column_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def require_columns(df, columns: list[str], option_name: str) -> list[str]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{option_name} contains columns not found in input: {missing}")
    return columns


def mappable_label(value, rel2class: dict[str, int]) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    if text in rel2class:
        return rel2class[text]
    upper = text.upper().replace("_", "-")
    if upper in rel2class:
        return rel2class[upper]
    short = {"N": "CMI-N", "D": "CMI-D", "U": "CMI-U"}.get(upper)
    if short is not None:
        return rel2class[short]
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number in set(rel2class.values()) else None


def detect_label_column(df, requested: str | None, rel2class: dict[str, int]) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--label_col not found in input: {requested}")
        return requested

    for col in LABEL_CANDIDATES:
        if col not in df.columns:
            continue
        values = [x for x in df[col].dropna().astype(str).head(100).tolist() if x.strip()]
        if not values:
            continue
        mapped = [mappable_label(x, rel2class) for x in values]
        if any(x is not None for x in mapped):
            return col
    return None


def map_label_series(series, rel2class: dict[str, int], class2rel: dict[int, str]):
    label_ids = []
    bad_values = []
    for value in series.tolist():
        mapped = mappable_label(value, rel2class)
        if mapped is None:
            bad_values.append(value)
            label_ids.append(None)
        else:
            label_ids.append(mapped)
    if bad_values:
        preview = list(dict.fromkeys(str(x) for x in bad_values))[:10]
        raise ValueError(f"Label column contains unmappable values: {preview}")
    label_names = [class2rel[int(x)] for x in label_ids]
    return pd.Series(label_ids, index=series.index, dtype="int64"), pd.Series(label_names, index=series.index)


def first_existing(df, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_compound_metabolite_ids(df, args, input_format_hint: str):
    compound_id_col = args.compound_id_col
    metabolite_id_col = args.metabolite_id_col

    if compound_id_col and compound_id_col not in df.columns:
        raise ValueError(f"--compound_id_col not found: {compound_id_col}")
    if metabolite_id_col and metabolite_id_col not in df.columns:
        raise ValueError(f"--metabolite_id_col not found: {metabolite_id_col}")

    if compound_id_col and metabolite_id_col:
        return (
            df[compound_id_col].astype(str),
            df[metabolite_id_col].astype(str),
            compound_id_col,
            metabolite_id_col,
        )

    if {"head", "tail", "head_type", "tail_type"}.issubset(df.columns):
        head_type = df["head_type"].astype(str).str.lower()
        tail_type = df["tail_type"].astype(str).str.lower()
        compound_values = []
        metabolite_values = []
        for idx, row in df.iterrows():
            ht = str(row["head_type"]).lower()
            tt = str(row["tail_type"]).lower()
            if ht == "compound" and tt == "metabolite":
                compound_values.append(str(row["head"]))
                metabolite_values.append(str(row["tail"]))
            elif ht == "metabolite" and tt == "compound":
                compound_values.append(str(row["tail"]))
                metabolite_values.append(str(row["head"]))
            else:
                raise ValueError(
                    "Rows with head_type/tail_type must identify one compound "
                    f"and one metabolite; failed at input row index {idx}."
                )
        return (
            pd.Series(compound_values, index=df.index),
            pd.Series(metabolite_values, index=df.index),
            "head/tail_by_type",
            "head/tail_by_type",
        )

    if not compound_id_col:
        compound_id_col = first_existing(
            df,
            ["compound_id", "CID", "cid", "compound", "cpd_id", "cpd", "drug_id", "drug"],
        )
    if not metabolite_id_col:
        metabolite_id_col = first_existing(
            df,
            ["metabolite_id", "hmdb_id", "HMDB", "hmdb", "metabolite", "met_id", "met"],
        )

    if compound_id_col and metabolite_id_col:
        return (
            df[compound_id_col].astype(str),
            df[metabolite_id_col].astype(str),
            compound_id_col,
            metabolite_id_col,
        )

    if "head" in df.columns and "tail" in df.columns and input_format_hint != "kpgt_features":
        logging.warning("Using head as compound ID and tail as metabolite ID.")
        return df["head"].astype(str), df["tail"].astype(str), "head", "tail"

    compound_name_col = args.compound_name_col if args.compound_name_col in df.columns else None
    metabolite_name_col = args.metabolite_name_col if args.metabolite_name_col in df.columns else None
    if compound_name_col and metabolite_name_col:
        logging.warning("Using name columns as model IDs because ID columns were not found.")
        return (
            df[compound_name_col].astype(str),
            df[metabolite_name_col].astype(str),
            compound_name_col,
            metabolite_name_col,
        )

    if input_format_hint == "kpgt_features":
        logging.warning("No ID columns found; using row-specific synthetic IDs.")
        return (
            pd.Series([f"external_compound_{i}" for i in range(len(df))], index=df.index),
            pd.Series([f"external_metabolite_{i}" for i in range(len(df))], index=df.index),
            None,
            None,
        )

    raise ValueError(
        "Could not identify compound/metabolite ID columns. Provide "
        "--compound_id_col and --metabolite_id_col."
    )


def is_numeric_like(series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().all()


def select_prefixed_columns(df, prefix: str | None) -> list[str] | None:
    if not prefix:
        return None
    cols = [c for c in df.columns if c.startswith(prefix)]
    return cols or None


def default_feature_prefixes(entity: str) -> list[str]:
    if entity == "compound":
        return ["compound_kpgt_", "compound_feature_", "compound_feat_", "cpd_kpgt_", "cpd_feature_"]
    return ["metabolite_kpgt_", "metabolite_feature_", "metabolite_feat_", "met_kpgt_", "met_feature_"]


def infer_prefixed_feature_columns(df, entity: str) -> list[str] | None:
    for prefix in default_feature_prefixes(entity):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            return cols
    return None


def select_feature_columns(
    df,
    args,
    label_col: str | None,
    compound_id_col: str | None,
    metabolite_id_col: str | None,
    compound_dim: int,
    metabolite_dim: int,
):
    compound_cols = parse_column_list(args.compound_feature_cols)
    metabolite_cols = parse_column_list(args.metabolite_feature_cols)
    if compound_cols is not None:
        compound_cols = require_columns(df, compound_cols, "--compound_feature_cols")
    if metabolite_cols is not None:
        metabolite_cols = require_columns(df, metabolite_cols, "--metabolite_feature_cols")

    if compound_cols is None:
        compound_cols = select_prefixed_columns(df, args.compound_feature_prefix)
    if metabolite_cols is None:
        metabolite_cols = select_prefixed_columns(df, args.metabolite_feature_prefix)
    if compound_cols is None:
        compound_cols = infer_prefixed_feature_columns(df, "compound")
    if metabolite_cols is None:
        metabolite_cols = infer_prefixed_feature_columns(df, "metabolite")

    used_explicit_or_prefix = bool(compound_cols or metabolite_cols)

    if compound_cols is not None and len(compound_cols) != compound_dim:
        raise ValueError(
            f"Compound feature column count is {len(compound_cols)}, "
            f"but the model expects {compound_dim}."
        )
    if metabolite_cols is not None and len(metabolite_cols) != metabolite_dim:
        raise ValueError(
            f"Metabolite feature column count is {len(metabolite_cols)}, "
            f"but the model expects {metabolite_dim}."
        )
    if (compound_cols is None) ^ (metabolite_cols is None):
        raise ValueError(
            "Compound and metabolite feature columns must be provided together."
        )

    if compound_cols is None and metabolite_cols is None:
        exclude = set(COMMON_META_COLUMNS)
        exclude.update(c for c in [label_col, compound_id_col, metabolite_id_col] if c)
        if args.compound_name_col:
            exclude.add(args.compound_name_col)
        if args.metabolite_name_col:
            exclude.add(args.metabolite_name_col)
        numeric_cols = [
            c for c in df.columns
            if c not in exclude and c not in LABEL_CANDIDATES and is_numeric_like(df[c])
        ]
        expected = compound_dim + metabolite_dim
        if len(numeric_cols) == expected:
            compound_cols = numeric_cols[:compound_dim]
            metabolite_cols = numeric_cols[compound_dim:]
            logging.info(
                "Auto-split %s numeric feature columns into %s compound and %s metabolite features.",
                expected,
                compound_dim,
                metabolite_dim,
            )

    feature_cols = set((compound_cols or []) + (metabolite_cols or []))
    if label_col and label_col in feature_cols:
        raise ValueError(
            f"Label column {label_col!r} was also selected as a feature column. "
            "Labels are never allowed as model input."
        )
    return compound_cols, metabolite_cols, used_explicit_or_prefix


def features_to_matrix(df, columns: list[str] | None, entity: str):
    if not columns:
        return None
    matrix = df[columns].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        bad_cols = matrix.columns[matrix.isna().any()].tolist()[:10]
        raise ValueError(f"{entity} feature columns contain non-numeric values: {bad_cols}")
    return matrix.astype("float32").to_numpy()
