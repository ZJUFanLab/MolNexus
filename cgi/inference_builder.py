from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .utils import DECODER_BASE_RELATIONS, build_relation_id_map


def build_rel2id(label_order: Sequence[str], expected_num_rels: int) -> Dict[str, int]:
    rel2id = build_relation_id_map(
        DECODER_BASE_RELATIONS,
        ("CPI",),
        label_order,
    )
    if len(rel2id) != expected_num_rels:
        raise ValueError(
            f"Relation count mismatch: configured CGI schema has {len(rel2id)} relations, "
            f"but checkpoint decoder expects {expected_num_rels}."
        )
    for rel in label_order:
        if rel not in rel2id:
            raise ValueError(f"Missing CGI relation in rel2id: {rel}")
    return rel2id


def included_background_rels(ablation_mode: str) -> set:
    mode = str(ablation_mode).strip().upper()
    full_rels = {"CPI", "PPI", "MPI", "GPI", "GGI"}
    if mode == "FULL":
        return full_rels.copy()
    if mode == "NO_CPI":
        return full_rels - {"CPI"}
    if mode == "NO_PPI":
        return full_rels - {"PPI"}
    if mode == "NO_MPI":
        return full_rels - {"MPI"}
    if mode == "NO_GPI":
        return full_rels - {"GPI"}
    if mode == "NO_GGI":
        return full_rels - {"GGI"}
    if mode == "NO_PROTEIN_EDGES":
        return full_rels - {"CPI", "PPI", "MPI", "GPI"}
    if mode == "NO_GENE_EDGES":
        return full_rels - {"GPI", "GGI"}
    if mode == "CGI_ONLY":
        return set()
    raise ValueError(f"Unknown ablation_mode: {ablation_mode}")


def map_edge_df(df: pd.DataFrame, ent2id: Dict[Tuple[str, str], int]) -> pd.DataFrame:
    mapped = df.copy()
    mapped["head_id"] = [
        ent2id.get((str(ent), str(etype)))
        for ent, etype in zip(mapped["head"].astype(str), mapped["head_type"].astype(str))
    ]
    mapped["tail_id"] = [
        ent2id.get((str(ent), str(etype)))
        for ent, etype in zip(mapped["tail"].astype(str), mapped["tail_type"].astype(str))
    ]
    mapped = mapped.dropna(subset=["head_id", "tail_id"]).copy()
    mapped[["head_id", "tail_id"]] = mapped[["head_id", "tail_id"]].astype(int)
    return mapped


def make_edges_by_df(df: pd.DataFrame, rel_name: str, torch_module) -> Dict[Tuple[str, str, str], Any]:
    edges = {}
    if df is None or df.empty:
        return edges
    df = df.sort_values(["head_type", "tail_type", "head_id", "tail_id"]).reset_index(drop=True)
    for (src_t, dst_t), sub in df.groupby(["head_type", "tail_type"], sort=True):
        head = torch_module.tensor(sub["head_id"].values, dtype=torch_module.long)
        tail = torch_module.tensor(sub["tail_id"].values, dtype=torch_module.long)
        if head.numel() > 0:
            edges[(str(src_t), rel_name, str(dst_t))] = torch_module.stack([head, tail], dim=0)
    return edges


def assign_edges_to_hetero(data, edge_dict: Dict[Tuple[str, str, str], Any]) -> None:
    for edge_type, edge_index in edge_dict.items():
        data[edge_type].edge_index = edge_index


def build_prediction_graph(
    training_module,
    edge_tables: Dict[str, pd.DataFrame],
    ent2id: Dict[Tuple[str, str], int],
    ablation_mode: str,
    torch_module,
) -> Tuple[Any, List[Tuple[str, str, str]]]:
    data = training_module.HeteroData()
    included = included_background_rels(ablation_mode)
    all_edges: Dict[Tuple[str, str, str], Any] = {}
    full_edge_types = set()
    for rel_name, table_key in [
        ("PPI", "ppi"),
        ("MPI", "mpi"),
        ("GPI", "gpi"),
        ("GGI", "ggi"),
        ("CPI", "cpi"),
    ]:
        mapped = map_edge_df(edge_tables[table_key], ent2id)
        relation_edges = make_edges_by_df(mapped, rel_name, torch_module)
        full_edge_types.update(relation_edges)
        if rel_name in included:
            all_edges.update(relation_edges)

    assign_edges_to_hetero(data, all_edges)

    return data, sorted(full_edge_types)


def map_prediction_pairs(
    pair_df: pd.DataFrame,
    ent2id: Dict[Tuple[str, str], int],
) -> pd.DataFrame:
    mapped = pair_df.copy()
    mapped["head_id"] = [
        ent2id.get((str(ent), "compound"))
        for ent in mapped["__compound_entity"].astype(str)
    ]
    mapped["tail_id"] = [
        ent2id.get((str(ent), "gene"))
        for ent in mapped["__gene_entity"].astype(str)
    ]
    if mapped["head_id"].isna().any() or mapped["tail_id"].isna().any():
        bad = mapped[mapped["head_id"].isna() | mapped["tail_id"].isna()].head(10)
        raise ValueError(f"Could not map external compound/gene pairs to entity IDs:\n{bad}")
    mapped[["head_id", "tail_id"]] = mapped[["head_id", "tail_id"]].astype(int)
    return mapped
