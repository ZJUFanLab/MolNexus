# graph_builder.py

import logging
import os

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .config import ABLATION_RELATIONS, normalize_ablation_mode


def get_included_bg_rels(ablation_mode: str):
    """Return the background relation families selected by an ablation mode."""
    canonical_mode = normalize_ablation_mode(ablation_mode)
    return set(ABLATION_RELATIONS[canonical_mode])


def make_edges_by_df(df, rel_name, device):
    """
    Group a dataframe by (head_type, tail_type), then convert it to an edge_index dictionary.
    The logic matches the original _make_edges_by_df.
    """
    edge_dict = {}
    if df is None or df.empty:
        return edge_dict

    for (src_t, dst_t), sub in df.groupby(["head_type", "tail_type"]):
        idx = torch.tensor(sub["head_id"].values, dtype=torch.long, device=device)
        idy = torch.tensor(sub["tail_id"].values, dtype=torch.long, device=device)
        if idx.numel() > 0:
            edge_dict[(src_t, rel_name, dst_t)] = torch.stack([idx, idy], dim=0)

    return edge_dict


def assign_edges_to_hetero(data: HeteroData, edge_dict: dict):
    """
    Write {(src, rel, dst): edge_index} to HeteroData.
    The logic matches the original _assign_edges_to_hetero.
    """
    for k, e in edge_dict.items():
        data[k].edge_index = e


def build_hetero_graph(
    cpi_df,
    cgi_df,
    ppi_df,
    mpi_df,
    gpi_df,
    ggi_df,
    device,
):
    """
    Construct the complete heterogeneous graph:
      1) CPI
      2) CGI (by specific relation name)
      3) Background edges: PPI / MPI / GPI / GGI
    The logic matches the original build_hetero_graph.
    """
    data = HeteroData()

    def _make_edges(df, rel_name):
        edge_dict = {}
        if df is None or df.empty:
            return edge_dict
        for (src_t, dst_t), sub in df.groupby(["head_type", "tail_type"]):
            idx = torch.tensor(sub["head_id"].values, dtype=torch.long, device=device)
            idy = torch.tensor(sub["tail_id"].values, dtype=torch.long, device=device)
            if idx.numel() > 0:
                edge_dict[(src_t, rel_name, dst_t)] = torch.stack([idx, idy], dim=0)
        return edge_dict

    edges = {}

    edges.update(_make_edges(cpi_df, "CPI"))

    if cgi_df is not None and not cgi_df.empty:
        for rel in cgi_df["relation"].unique():
            edges.update(_make_edges(cgi_df[cgi_df["relation"] == rel], rel))

    for df, rel in [
        (ppi_df, "PPI"),
        (mpi_df, "MPI"),
        (gpi_df, "GPI"),
        (ggi_df, "GGI"),
    ]:
        edges.update(_make_edges(df, rel))

    assign_edges_to_hetero(data, edges)
    return data


def build_bg_graph(
    included_rels: set,
    cgi_df_subset,
    ppi_df,
    mpi_df,
    gpi_df,
    ggi_df,
    device,
):
    """
    Construct the background graph from included_rels and cgi_df_subset.
    The logic matches the original build_bg_graph.
    """
    data = HeteroData()
    edges = {}

    if "CGI" in included_rels and cgi_df_subset is not None and not cgi_df_subset.empty:
        for rel in cgi_df_subset["relation"].unique():
            edges.update(
                make_edges_by_df(
                    cgi_df_subset[cgi_df_subset["relation"] == rel],
                    rel,
                    device,
                )
            )

    if "PPI" in included_rels:
        edges.update(make_edges_by_df(ppi_df, "PPI", device))
    if "MPI" in included_rels:
        edges.update(make_edges_by_df(mpi_df, "MPI", device))
    if "GPI" in included_rels:
        edges.update(make_edges_by_df(gpi_df, "GPI", device))
    if "GGI" in included_rels:
        edges.update(make_edges_by_df(ggi_df, "GGI", device))

    assign_edges_to_hetero(data, edges)
    return data


def add_cpi_edges(data: HeteroData, cpi_df_subset, device):
    """
    Append CPI edges to an existing HeteroData object.
    The logic matches the original add_cpi_edges.
    """
    if cpi_df_subset is None or cpi_df_subset.empty:
        return data

    edge_dict = make_edges_by_df(cpi_df_subset, "CPI", device)
    assign_edges_to_hetero(data, edge_dict)
    return data


def split_cgi_by_compound(all_cgi_df, train_cpi_df, val_cpi_df):
    """
    Split CGI according to each compound split.
    The logic matches the original split_cgi_by_compound.
    """
    train_comp_ids = set(
        train_cpi_df.loc[train_cpi_df["head_type"] == "compound", "head_id"].tolist()
    )
    val_comp_ids = set(
        val_cpi_df.loc[val_cpi_df["head_type"] == "compound", "head_id"].tolist()
    )

    cgi_train = all_cgi_df[
        all_cgi_df["head_type"].eq("compound") & all_cgi_df["head_id"].isin(train_comp_ids)
    ].copy()

    cgi_val = all_cgi_df[
        all_cgi_df["head_type"].eq("compound") & all_cgi_df["head_id"].isin(val_comp_ids)
    ].copy()

    extra = all_cgi_df[~all_cgi_df["head_type"].eq("compound")].copy()
    if not extra.empty:
        cgi_train = pd.concat([cgi_train, extra], ignore_index=True)
        cgi_val = pd.concat([cgi_val, extra], ignore_index=True)

    return cgi_train, cgi_val


def log_cgi_usage(tag: str, data: HeteroData, train_cgi_df, ext_cgi_mapped, compounds, ent2id):
    """
    Log the presence of CGI relations in the graph and the number of CGI incident edges for each compound.
    The logic matches the original _log_cgi_usage.
    """
    cgi_rels = set()

    if train_cgi_df is not None and (not train_cgi_df.empty) and ("relation" in train_cgi_df.columns):
        cgi_rels |= set(train_cgi_df["relation"].unique().tolist())

    if ext_cgi_mapped is not None and (not ext_cgi_mapped.empty) and ("relation" in ext_cgi_mapped.columns):
        cgi_rels |= set(ext_cgi_mapped["relation"].unique().tolist())

    cgi_keys = [k for k in data.edge_types if k[1] in cgi_rels]
    tot_edges = sum(int(data[k].edge_index.size(1)) for k in cgi_keys) if len(cgi_keys) else 0

    logging.info(
        f"[{tag}] CGI relations present in graph: "
        f"{cgi_keys if cgi_keys else 'NONE'} | total CGI edges = {tot_edges}"
    )

    verbose = os.environ.get("LOG_CGI_DETAIL", "0") == "1"

    for c in compounds:
        key = (c, "compound")
        if key not in ent2id:
            if verbose:
                logging.info(
                    f"[{tag}] compound '{c}' not in ent2id (likely filtered or missing features)"
                )
            continue

        hid = ent2id[key]
        cnt = 0

        for (src_t, rel, dst_t) in cgi_keys:
            eidx = data[(src_t, rel, dst_t)].edge_index
            if eidx.numel() == 0:
                continue
            try:
                cnt += int((eidx[0] == hid).sum().item())
            except Exception:
                pass

        if cnt > 0 or verbose:
            logging.info(f"[{tag}] compound '{c}' (id={hid}) incident CGI edges: {cnt}")
