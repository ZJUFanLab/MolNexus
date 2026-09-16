from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .utils import DECODER_BASE_RELATIONS, build_relation_id_map


def map_background_edges(df, ent2id: dict):
    if df is None or df.empty:
        return df.copy()
    mapped = df.copy()
    mapped["key_head"] = list(zip(mapped["head"].astype(str), mapped["head_type"].astype(str)))
    mapped["key_tail"] = list(zip(mapped["tail"].astype(str), mapped["tail_type"].astype(str)))
    mapped["head_id"] = mapped["key_head"].map(ent2id)
    mapped["tail_id"] = mapped["key_tail"].map(ent2id)
    before = len(mapped)
    mapped = mapped.dropna(subset=["head_id", "tail_id"]).copy()
    dropped = before - len(mapped)
    if dropped:
        logging.warning("Dropped %s background edges with unmapped endpoints.", dropped)
    mapped[["head_id", "tail_id"]] = mapped[["head_id", "tail_id"]].astype(int)
    return mapped


def assign_edges_to_hetero(data, edge_dict: dict) -> None:
    for key, edge_index in edge_dict.items():
        data[key].edge_index = edge_index


def make_edges_by_df(df, rel_name: str, device):
    edge_dict = {}
    if df is None or df.empty:
        return edge_dict
    for (src_type, dst_type), sub in df.groupby(["head_type", "tail_type"]):
        idx = torch.tensor(sub["head_id"].values, dtype=torch.long, device=device)
        idy = torch.tensor(sub["tail_id"].values, dtype=torch.long, device=device)
        if idx.numel() > 0:
            edge_dict[(src_type, rel_name, dst_type)] = torch.stack([idx, idy], dim=0)
    return edge_dict


def build_background_graph(mapped_tables: dict, device):
    data = HeteroData()
    edge_dict = {}
    for rel in mapped_tables["cgi"]["relation"].dropna().unique():
        sub = mapped_tables["cgi"][mapped_tables["cgi"]["relation"] == rel]
        edge_dict.update(make_edges_by_df(sub, str(rel), device))
    edge_dict.update(make_edges_by_df(mapped_tables["ppi"], "PPI", device))
    edge_dict.update(make_edges_by_df(mapped_tables["mpi"], "MPI", device))
    edge_dict.update(make_edges_by_df(mapped_tables["gpi"], "GPI", device))
    edge_dict.update(make_edges_by_df(mapped_tables["ggi"], "GGI", device))
    edge_dict.update(make_edges_by_df(mapped_tables["cpi"], "CPI", device))
    assign_edges_to_hetero(data, edge_dict)
    edge_types = sorted(edge_dict.keys())
    logging.info("Background graph edge types (%s): %s", len(edge_types), edge_types)
    return data, edge_types


def build_rel2id(cgi_df, label_order: list[str]) -> dict[str, int]:
    cgi_rels = set(cgi_df["relation"].dropna().unique()) if cgi_df is not None and not cgi_df.empty else set()
    return build_relation_id_map(
        DECODER_BASE_RELATIONS,
        cgi_rels,
        ("CPI",),
        label_order,
    )
