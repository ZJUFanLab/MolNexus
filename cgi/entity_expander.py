from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
import torch
from .utils import (
    detect_col, detect_input_format, detect_label_col, feature_cols_by_prefix,
    normalize_labels, parse_feature_cols, resolve_triplet_entity_cols, to_clean_str_series,
)
from .config import (
    COMMON_COMPOUND_FEATURE_PREFIXES, COMMON_COMPOUND_ID_COLS,
    COMMON_COMPOUND_NAME_COLS, COMMON_GENE_FEATURE_PREFIXES,
    COMMON_GENE_ID_COLS, COMMON_GENE_NAME_COLS, DEFAULT_PATHS, TYPE_ORDER,
)


class TrainingEntityExpander:

    def create_entity_features(self):
            """
            Retrieve the corresponding features for each ID based on ent2type and id2ent, or initialize them randomly.
            """
            num_entities = len(self.ent2id)
            features_list = []

            # Feature tables and dimensions for different types
            tables = {
                'compound':   (self.cpd_features,        self.cpd_features.shape[1]),
                'metabolite': (self.metabolite_features, self.metabolite_features.shape[1]),
                'protein':    (self.protein_features,    self.protein_features.shape[1]),
                'gene':       (self.gene_features,       self.gene_features.shape[1]),
            }

            for idx in range(num_entities):
                ent = self.id2ent[idx]
                etype = self.ent2type[idx]
                df, dim = tables.get(etype, (None, None))

                if df is not None and ent in df.index:
                    arr = df.loc[ent].values
                else:
                    logging.warning(f"Entity '{ent}' of type '{etype}' has no features; initializing randomly")
                    arr = np.random.randn(dim)

                features_list.append(
                    torch.tensor(arr, dtype=torch.float32, device=self.device)
                )

            return features_list


def build_feature_override(
    df: pd.DataFrame,
    entity_ids: pd.Series,
    feature_cols: Sequence[str],
    entity_type: str,
    expected_dim: int,
) -> pd.DataFrame:
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing {entity_type} feature columns: {missing_cols[:10]}")
    features = df.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    if features.shape[1] != expected_dim:
        raise ValueError(
            f"{entity_type} feature dimension mismatch: got {features.shape[1]}, "
            f"expected {expected_dim} from checkpoint."
        )
    if features.isna().any().any():
        bad_cols = features.columns[features.isna().any()].tolist()[:10]
        raise ValueError(f"{entity_type} feature columns contain non-numeric or NaN values: {bad_cols}")
    out = features.astype("float32").copy()
    out.index = entity_ids.astype(str).values
    before = len(out)
    out = out[~out.index.duplicated(keep="first")]
    if len(out) < before:
        logging.warning(
            "%s feature input has duplicate entity IDs; keeping the first feature row for each ID.",
            entity_type,
        )
    return out


def prepare_external_input(
    raw_df: pd.DataFrame,
    args,
    expected_feat_dims: Dict[str, int],
    rel2class: Dict[str, int],
) -> Dict[str, Any]:
    input_format = detect_input_format(raw_df, args)
    label_col = detect_label_col(raw_df, args.label_col)
    compound_name_col = detect_col(raw_df, args.compound_name_col, COMMON_COMPOUND_NAME_COLS, "compound name")
    gene_name_col = detect_col(raw_df, args.gene_name_col, COMMON_GENE_NAME_COLS, "gene name")

    compound_override = None
    gene_override = None
    feature_cols: List[str] = []

    if input_format == "triplet":
        compound_col, gene_col = resolve_triplet_entity_cols(raw_df, args)
        compound_ids = to_clean_str_series(raw_df[compound_col], compound_col)
        gene_ids = to_clean_str_series(raw_df[gene_col], gene_col)
    else:
        compound_id_col = detect_col(raw_df, args.compound_id_col, COMMON_COMPOUND_ID_COLS, "compound id")
        gene_id_col = detect_col(raw_df, args.gene_id_col, COMMON_GENE_ID_COLS, "gene id")

        if compound_id_col is None and compound_name_col is not None:
            compound_id_col = compound_name_col
        if gene_id_col is None and gene_name_col is not None:
            gene_id_col = gene_name_col

        if compound_id_col is None:
            compound_ids = pd.Series([f"external_compound_{i}" for i in range(len(raw_df))], index=raw_df.index)
            logging.warning("No compound ID/name column found; using row-specific synthetic compound IDs.")
        else:
            compound_ids = to_clean_str_series(raw_df[compound_id_col], compound_id_col)

        if gene_id_col is None:
            gene_ids = pd.Series([f"external_gene_{i}" for i in range(len(raw_df))], index=raw_df.index)
            logging.warning("No gene ID/name column found; using row-specific synthetic gene IDs.")
        else:
            gene_ids = to_clean_str_series(raw_df[gene_id_col], gene_id_col)

        compound_cols = parse_feature_cols(args.compound_feature_cols)
        if compound_cols is None:
            compound_cols = feature_cols_by_prefix(
                raw_df,
                args.compound_feature_prefix,
                COMMON_COMPOUND_FEATURE_PREFIXES,
            )
        gene_cols = parse_feature_cols(args.gene_feature_cols)
        if gene_cols is None:
            gene_cols = feature_cols_by_prefix(raw_df, args.gene_feature_prefix, COMMON_GENE_FEATURE_PREFIXES)

        if not compound_cols or not gene_cols:
            raise ValueError(
                "KPGT feature input requires compound and gene feature columns. "
                "Set --compound_feature_prefix/--gene_feature_prefix or "
                "--compound_feature_cols/--gene_feature_cols."
            )

        protected = {c for c in [label_col, compound_id_col, gene_id_col, compound_name_col, gene_name_col] if c}
        for protected_col in protected:
            if protected_col in compound_cols or protected_col in gene_cols:
                raise ValueError(f"Metadata/label column was selected as a feature: {protected_col}")

        feature_cols = list(dict.fromkeys(list(compound_cols) + list(gene_cols)))
        compound_override = build_feature_override(
            raw_df,
            compound_ids,
            compound_cols,
            "compound",
            expected_feat_dims["compound"],
        )
        gene_override = build_feature_override(
            raw_df,
            gene_ids,
            gene_cols,
            "gene",
            expected_feat_dims["gene"],
        )

    pair_df = pd.DataFrame(
        {
            "__row_id": np.arange(len(raw_df), dtype=np.int64),
            "__compound_entity": compound_ids.astype(str).values,
            "__gene_entity": gene_ids.astype(str).values,
        }
    )

    y_true = None
    if label_col is not None:
        y_true = normalize_labels(raw_df[label_col], rel2class)

    meta_cols = [c for c in raw_df.columns if c not in set(feature_cols)]
    meta_df = raw_df.loc[:, meta_cols].copy()

    logging.info("Detected input_format=%s", input_format)
    logging.info("Detected label_col=%s", label_col)
    logging.info("Metadata columns retained in predictions.csv: %s", meta_cols)

    return {
        "input_format": input_format,
        "pair_df": pair_df,
        "meta_df": meta_df,
        "label_col": label_col,
        "y_true": y_true,
        "compound_override": compound_override,
        "gene_override": gene_override,
    }


def read_edge_table(path: str, name: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    df = pd.read_csv(path, dtype=str)
    required = {"head", "tail", "head_type", "tail_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} file missing required columns: {sorted(missing)}")
    return df


def read_feature_table(
    path: str,
    entity_type: str,
    expected_dim: int,
    gene_index_col: str = "gene",
) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{entity_type} feature file does not exist: {path}")

    if entity_type == "gene":
        expected_feature_cols = [f"feature_{i}" for i in range(expected_dim)]
        header = pd.read_csv(path, nrows=0)
        if gene_index_col in header.columns and set(expected_feature_cols).issubset(header.columns):
            df = pd.read_csv(path, usecols=[gene_index_col] + expected_feature_cols, dtype={gene_index_col: str})
            df = df.set_index(gene_index_col)
        else:
            df = pd.read_csv(path, index_col=0, dtype={0: str})
    else:
        df = pd.read_csv(path, index_col=0, dtype={0: str})

    df.index = df.index.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    if df.shape[1] != expected_dim:
        raise ValueError(
            f"{entity_type} feature dimension mismatch for {path}: "
            f"got {df.shape[1]}, expected {expected_dim} from checkpoint."
        )
    return df


def read_base_features(args, feat_dims: Dict[str, int]) -> Dict[str, pd.DataFrame]:
    return {
        "compound": read_feature_table(args.cpd_features_file, "compound", feat_dims["compound"]),
        "metabolite": read_feature_table(args.metabolite_features_file, "metabolite", feat_dims["metabolite"]),
        "protein": read_feature_table(args.protein_features_file, "protein", feat_dims["protein"]),
        "gene": read_feature_table(args.gene_features_file, "gene", feat_dims["gene"]),
    }


def collect_nodes_from_edges(nodes: set, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    for ent_col, type_col in [("head", "head_type"), ("tail", "tail_type")]:
        if ent_col in df.columns and type_col in df.columns:
            nodes.update(zip(df[ent_col].astype(str), df[type_col].astype(str)))


def build_entity_maps(
    edge_tables: Dict[str, pd.DataFrame],
    base_features: Dict[str, pd.DataFrame],
    pair_df: pd.DataFrame,
    compound_override: Optional[pd.DataFrame],
    gene_override: Optional[pd.DataFrame],
) -> Tuple[Dict[Tuple[str, str], int], Dict[int, str], Dict[int, str]]:
    nodes = set()
    for df in edge_tables.values():
        collect_nodes_from_edges(nodes, df)

    for node_type, feat_df in base_features.items():
        nodes.update((str(ent), node_type) for ent in feat_df.index.astype(str))

    nodes.update((str(x), "compound") for x in pair_df["__compound_entity"].astype(str))
    nodes.update((str(x), "gene") for x in pair_df["__gene_entity"].astype(str))
    if compound_override is not None:
        nodes.update((str(x), "compound") for x in compound_override.index.astype(str))
    if gene_override is not None:
        nodes.update((str(x), "gene") for x in gene_override.index.astype(str))

    ent_list: List[Tuple[str, str]] = []
    for node_type in TYPE_ORDER:
        ents = sorted([ent for ent, etype in nodes if etype == node_type])
        ent_list.extend((ent, node_type) for ent in ents)

    ent2id = {key: idx for idx, key in enumerate(ent_list)}
    id2ent = {idx: ent for (ent, _), idx in ent2id.items()}
    ent2type = {idx: etype for (_, etype), idx in ent2id.items()}
    return ent2id, id2ent, ent2type


def make_feature_list(
    ent2id: Dict[Tuple[str, str], int],
    id2ent: Dict[int, str],
    ent2type: Dict[int, str],
    base_features: Dict[str, pd.DataFrame],
    feat_dims: Dict[str, int],
    required_external: set,
    compound_override: Optional[pd.DataFrame],
    gene_override: Optional[pd.DataFrame],
    allow_missing_features_random: bool,
    random_seed: int,
    torch_module,
) -> List[Any]:
    rng = np.random.default_rng(random_seed)
    overrides = {
        "compound": compound_override if compound_override is not None else pd.DataFrame(),
        "gene": gene_override if gene_override is not None else pd.DataFrame(),
    }
    features_list: List[Any] = []
    missing_required: List[Tuple[str, str]] = []
    missing_background: List[Tuple[str, str]] = []

    for idx in range(len(id2ent)):
        ent = id2ent[idx]
        node_type = ent2type[idx]
        arr = None
        override_df = overrides.get(node_type)
        if override_df is not None and not override_df.empty and ent in override_df.index:
            arr = override_df.loc[ent].values
        elif ent in base_features[node_type].index:
            arr = base_features[node_type].loc[ent].values

        if arr is None:
            key = (ent, node_type)
            if key in required_external and not allow_missing_features_random:
                missing_required.append(key)
            else:
                missing_background.append(key)
                arr = rng.standard_normal(feat_dims[node_type]).astype("float32")

        if arr is not None:
            features_list.append(torch_module.tensor(arr, dtype=torch_module.float32))

    if missing_required:
        examples = missing_required[:20]
        raise ValueError(
            "External prediction entities are missing required features. "
            f"Examples: {examples}. For triplet input, make sure IDs match the training feature tables. "
            "For direct KPGT input, provide feature columns. "
            "Use --allow_missing_features_random only if random features are acceptable."
        )
    if missing_background:
        logging.warning(
            "Random-initialized %d background/non-required entities with missing features. Examples: %s",
            len(missing_background),
            missing_background[:10],
        )
    return features_list
