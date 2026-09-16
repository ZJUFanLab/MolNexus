from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .config import (
    COMMON_COMPOUND_FEATURE_PREFIXES, COMMON_COMPOUND_ID_COLS,
    COMMON_COMPOUND_NAME_COLS, COMMON_GENE_FEATURE_PREFIXES,
    COMMON_GENE_ID_COLS, COMMON_GENE_NAME_COLS, COMMON_LABEL_COLS,
)

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


def normalize_delimiter(delimiter: str) -> Optional[str]:
    if delimiter is None:
        return None
    value = str(delimiter).strip()
    if value == "" or value.lower() == "auto":
        return None
    if value.lower() == "tab" or value == "\\t":
        return "\t"
    return value


def read_input_table(path: str, delimiter: str) -> pd.DataFrame:
    sep = normalize_delimiter(delimiter)
    if sep is None:
        return pd.read_csv(path, sep=None, engine="python", dtype=str)
    return pd.read_csv(path, sep=sep, dtype=str)


def detect_col(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str], role: str) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{role} column was specified but not found: {explicit}")
        return explicit
    lower_to_actual = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        actual = lower_to_actual.get(candidate.lower())
        if actual is not None:
            return actual
    return None


def detect_label_col(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    return detect_col(df, explicit, COMMON_LABEL_COLS, "label")


def resolve_triplet_entity_cols(df: pd.DataFrame, args) -> Tuple[str, str]:
    if args.compound_id_col or args.gene_id_col:
        compound_col = detect_col(df, args.compound_id_col, COMMON_COMPOUND_ID_COLS, "compound id")
        gene_col = detect_col(df, args.gene_id_col, COMMON_GENE_ID_COLS, "gene id")
        if compound_col is None or gene_col is None:
            raise ValueError("Could not resolve compound/gene columns from explicit arguments.")
        return compound_col, gene_col

    if {"head", "tail"}.issubset(df.columns):
        if {"head_type", "tail_type"}.issubset(df.columns):
            head_types = set(df["head_type"].dropna().astype(str).str.lower().unique())
            tail_types = set(df["tail_type"].dropna().astype(str).str.lower().unique())
            if "compound" in head_types and "gene" in tail_types:
                return "head", "tail"
            if "gene" in head_types and "compound" in tail_types:
                return "tail", "head"
        return "head", "tail"

    compound_col = detect_col(df, None, COMMON_COMPOUND_ID_COLS, "compound id")
    gene_col = detect_col(df, None, COMMON_GENE_ID_COLS, "gene id")
    if compound_col is None:
        compound_col = detect_col(df, args.compound_name_col, COMMON_COMPOUND_NAME_COLS, "compound name")
    if gene_col is None:
        gene_col = detect_col(df, args.gene_name_col, COMMON_GENE_NAME_COLS, "gene name")
    if compound_col is None or gene_col is None:
        raise ValueError(
            "Could not auto-detect compound/gene columns. "
            "Set --compound_id_col and --gene_id_col."
        )
    return compound_col, gene_col


def parse_feature_cols(value: Optional[str]) -> Optional[List[str]]:
    if value is None or str(value).strip() == "":
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def feature_cols_by_prefix(df: pd.DataFrame, prefix: Optional[str], common_prefixes: Sequence[str]) -> List[str]:
    prefixes = []
    if prefix:
        prefixes.append(prefix)
    else:
        prefixes.extend(common_prefixes)
    for pref in prefixes:
        cols = [c for c in df.columns if c.startswith(pref)]
        if cols:
            return sorted(cols, key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)])
    return []


def detect_input_format(df: pd.DataFrame, args) -> str:
    if args.input_format != "auto":
        return args.input_format
    if args.compound_feature_cols or args.gene_feature_cols:
        return "kpgt_features"
    if args.compound_feature_prefix or args.gene_feature_prefix:
        return "kpgt_features"
    has_compound_feature = bool(feature_cols_by_prefix(df, None, COMMON_COMPOUND_FEATURE_PREFIXES))
    has_gene_feature = bool(feature_cols_by_prefix(df, None, COMMON_GENE_FEATURE_PREFIXES))
    if has_compound_feature and has_gene_feature:
        return "kpgt_features"
    return "triplet"


def to_clean_str_series(series: pd.Series, col: str) -> pd.Series:
    out = series.astype(str).str.strip()
    if (out == "").any() or out.isna().any():
        raise ValueError(f"Column {col} contains empty IDs.")
    return out


def normalize_label_value(value: Any, rel2class: Dict[str, int]) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text in rel2class:
        return int(rel2class[text])
    upper = text.upper()
    alias = {
        "N": "CGI-N",
        "D": "CGI-D",
        "U": "CGI-U",
        "CGI_N": "CGI-N",
        "CGI_D": "CGI-D",
        "CGI_U": "CGI-U",
        "NEG": "CGI-N",
        "NEGATIVE": "CGI-N",
        "DOWN": "CGI-D",
        "UP": "CGI-U",
    }
    if upper in alias and alias[upper] in rel2class:
        return int(rel2class[alias[upper]])
    try:
        label_id = int(float(text))
        if 0 <= label_id < len(rel2class):
            return label_id
    except ValueError:
        pass
    return None


def normalize_labels(series: pd.Series, rel2class: Dict[str, int]) -> np.ndarray:
    values = [normalize_label_value(x, rel2class) for x in series]
    bad = [str(series.iloc[i]) for i, v in enumerate(values) if v is None]
    if bad:
        examples = sorted(set(bad))[:10]
        raise ValueError(f"Unknown label values in label column: {examples}")
    return np.array(values, dtype=np.int64)
