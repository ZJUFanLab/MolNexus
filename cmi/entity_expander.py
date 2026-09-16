from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .utils import sniff_delimiter


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


def read_feature_table(path: str, index_col=0, usecols=None):
    kwargs = {"index_col": index_col}
    if usecols is not None:
        kwargs["usecols"] = usecols
    df = pd.read_csv(path, **kwargs)
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors="coerce").astype("float32")


def read_base_tables(args):
    logging.info("Reading background edge tables and feature tables.")
    tables = {
        "ppi": pd.read_csv(args.ppi_file, dtype=str),
        "mpi": pd.read_csv(args.mpi_file, dtype=str),
        "gpi": pd.read_csv(args.gpi_file, dtype=str),
        "ggi": pd.read_csv(args.ggi_file, dtype=str),
        "cgi": pd.read_csv(args.cgi_file, dtype=str),
        "cpi": pd.read_csv(args.cpi_pos_file, dtype=str),
    }
    feature_cols = [f"feature_{i}" for i in range(1280)]
    features = {
        "compound": read_feature_table(args.compound_features_file, index_col=0),
        "metabolite": read_feature_table(args.metabolite_features_file, index_col=0),
        "protein": read_feature_table(args.protein_features_file, index_col=0),
        "gene": read_feature_table(args.gene_features_file, index_col="gene", usecols=["gene"] + feature_cols),
    }
    logging.info(
        "Feature dims: compound=%s metabolite=%s protein=%s gene=%s",
        features["compound"].shape[1],
        features["metabolite"].shape[1],
        features["protein"].shape[1],
        features["gene"].shape[1],
    )
    return tables, features


def collect_nodes_from_edges(*dfs) -> set[tuple[str, str]]:
    nodes = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        for entity_col, type_col in [("head", "head_type"), ("tail", "tail_type")]:
            if entity_col in df.columns and type_col in df.columns:
                nodes.update(zip(df[entity_col].astype(str), df[type_col].astype(str)))
    return nodes


def add_feature_index_nodes(nodes: set[tuple[str, str]], features: dict) -> None:
    for entity_type, table in features.items():
        for entity in table.index.astype(str):
            nodes.add((entity, entity_type))


def build_entity_maps(tables: dict, features: dict, compound_ids, metabolite_ids):
    nodes = collect_nodes_from_edges(
        tables["ppi"], tables["mpi"], tables["gpi"], tables["ggi"], tables["cgi"], tables["cpi"]
    )
    add_feature_index_nodes(nodes, features)
    nodes.update((str(x), "compound") for x in compound_ids)
    nodes.update((str(x), "metabolite") for x in metabolite_ids)

    type_order = ["compound", "metabolite", "protein", "gene"]
    ent_list = []
    for entity_type in type_order:
        entities = sorted(entity for entity, found_type in nodes if found_type == entity_type)
        ent_list.extend((entity, entity_type) for entity in entities)

    ent2id = {key: idx for idx, key in enumerate(ent_list)}
    id2ent = {idx: key[0] for key, idx in ent2id.items()}
    ent2type = {idx: key[1] for key, idx in ent2id.items()}
    logging.info("Entity counts: %s", {t: sum(1 for _, et in ent_list if et == t) for t in type_order})
    return ent2id, id2ent, ent2type


def upsert_external_features(features, entity_type: str, entity_ids, matrix) -> set[str]:
    if matrix is None:
        return set()
    table = features[entity_type]
    seen = {}
    conflicts = 0
    for entity_id, vector in zip(entity_ids.astype(str).tolist(), matrix):
        if entity_id in seen:
            if not np.allclose(seen[entity_id], vector, equal_nan=True):
                conflicts += 1
            continue
        seen[entity_id] = vector
    if conflicts:
        logging.warning(
            "%s duplicated IDs had conflicting input features; keeping the first occurrence.",
            conflicts,
        )
    if seen:
        insert_df = pd.DataFrame.from_dict(seen, orient="index", columns=table.columns).astype("float32")
        insert_df.index = insert_df.index.astype(str)
        features[entity_type] = pd.concat([table.drop(index=insert_df.index, errors="ignore"), insert_df], axis=0)
    return set(seen.keys())


def read_external_feature_table(
    path: str,
    id_col: str | None,
    expected_dim: int,
    delimiter: str,
    entity_type: str,
):
    sep = sniff_delimiter(path, delimiter)
    df = pd.read_csv(path, sep=sep, dtype=str)
    if df.empty:
        raise ValueError(f"External {entity_type} feature file is empty: {path}")

    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"External {entity_type} feature ID column not found: {id_col}")
        ids = df[id_col].astype(str)
        raw_features = df.drop(columns=[id_col])
    else:
        ids = df.iloc[:, 0].astype(str)
        raw_features = df.iloc[:, 1:]

    numeric = raw_features.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in numeric.columns if numeric[c].notna().all()]

    if len(numeric_cols) == expected_dim:
        matrix = numeric[numeric_cols]
    elif numeric.shape[1] == expected_dim and numeric.notna().all().all():
        matrix = numeric
    else:
        raise ValueError(
            f"External {entity_type} KPGT feature dimension mismatch for {path}: "
            f"found {len(numeric_cols)} fully numeric columns, model expects {expected_dim}. "
            "Use a file with one ID column followed by the exact KPGT feature columns, "
            "or remove numeric metadata columns."
        )

    out = matrix.astype("float32")
    out.index = ids
    duplicated = out.index.duplicated(keep="first")
    if duplicated.any():
        logging.warning(
            "External %s feature file has %s duplicated IDs; keeping first occurrence.",
            entity_type,
            int(duplicated.sum()),
        )
        out = out[~duplicated]
    return out


def merge_external_feature_file(
    features: dict,
    entity_type: str,
    path: str | None,
    id_col: str | None,
    expected_dim: int,
    delimiter: str,
):
    if not path:
        return
    external = read_external_feature_table(path, id_col, expected_dim, delimiter, entity_type)
    table = features[entity_type]
    external.columns = table.columns
    before = len(table)
    overlap = len(set(table.index.astype(str)).intersection(set(external.index.astype(str))))
    features[entity_type] = pd.concat(
        [table.drop(index=external.index, errors="ignore"), external],
        axis=0,
    )
    logging.info(
        "Merged external %s KPGT features: %s rows from %s (%s replaced existing rows, %s -> %s total rows).",
        entity_type,
        len(external),
        path,
        overlap,
        before,
        len(features[entity_type]),
    )


def validate_target_features(
    features: dict,
    compound_ids,
    metabolite_ids,
    allow_random: bool,
):
    missing_compounds = sorted(set(compound_ids.astype(str)) - set(features["compound"].index.astype(str)))
    missing_metabolites = sorted(set(metabolite_ids.astype(str)) - set(features["metabolite"].index.astype(str)))
    if (missing_compounds or missing_metabolites) and not allow_random:
        msg = []
        if missing_compounds:
            msg.append(f"missing compound KPGT features: {missing_compounds[:10]}")
        if missing_metabolites:
            msg.append(f"missing metabolite KPGT features: {missing_metabolites[:10]}")
        raise ValueError(
            "Target entities are missing required features (" + "; ".join(msg) + "). "
            "Provide --external_compound_features_file/--external_metabolite_features_file, "
            "provide feature columns/prefixes in the input file, fix ID matching, or use "
            "--allow_random_missing_external_features explicitly."
        )


def create_entity_features(ent2type: dict, id2ent: dict, features: dict, seed: int = 42):
    rng = np.random.default_rng(seed)
    result = []
    missing_by_type = {}
    for idx in range(len(id2ent)):
        entity = id2ent[idx]
        entity_type = ent2type[idx]
        table = features[entity_type]
        dim = table.shape[1]
        if entity in table.index:
            arr = table.loc[entity].to_numpy(dtype="float32")
        else:
            missing_by_type[entity_type] = missing_by_type.get(entity_type, 0) + 1
            arr = rng.standard_normal(dim).astype("float32")
        result.append(torch.tensor(arr, dtype=torch.float32))
    if missing_by_type:
        logging.warning(
            "Randomly initialized features for non-table entities: %s",
            missing_by_type,
        )
    return result


def map_target_pairs(compound_ids, metabolite_ids, ent2id: dict):
    rows = []
    missing = []
    for i, (compound_id, metabolite_id) in enumerate(zip(compound_ids.astype(str), metabolite_ids.astype(str))):
        compound_key = (str(compound_id), "compound")
        metabolite_key = (str(metabolite_id), "metabolite")
        if compound_key not in ent2id or metabolite_key not in ent2id:
            missing.append((i, compound_id, metabolite_id))
            continue
        rows.append((ent2id[compound_key], ent2id[metabolite_key]))
    if missing:
        raise ValueError(f"Could not map target pairs to entity IDs. First failures: {missing[:10]}")
    return np.asarray(rows, dtype=np.int64)
