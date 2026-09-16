from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import HeteroData


def get_edge_index_dict_safe(data: HeteroData):
    """Return the edge indices stored in ``data``, or an empty dict."""
    edge_dict = {}
    for etype in data.edge_types:
        store = data[etype]
        if "edge_index" in store:
            edge_dict[etype] = store.edge_index
    return edge_dict


class TrainingGraphBuilder:

    def get_included_bg_rels(self, ablation_mode: str):
            ablation_mode = ablation_mode.strip().upper()
            assert ablation_mode in self.ABLATION_MODES, f"Unknown ablation mode: {ablation_mode}"

            full_rels = {"CPI", "PPI", "MPI", "GPI", "GGI"}

            if ablation_mode == "FULL":
                return full_rels.copy()
            if ablation_mode == "NO_CPI":
                return full_rels - {"CPI"}
            if ablation_mode == "NO_PPI":
                return full_rels - {"PPI"}
            if ablation_mode == "NO_MPI":
                return full_rels - {"MPI"}
            if ablation_mode == "NO_GPI":
                return full_rels - {"GPI"}
            if ablation_mode == "NO_GGI":
                return full_rels - {"GGI"}
            if ablation_mode == "NO_PROTEIN_EDGES":
                return full_rels - {"CPI", "PPI", "MPI", "GPI"}
            if ablation_mode == "NO_GENE_EDGES":
                return full_rels - {"GPI", "GGI"}
            if ablation_mode == "CGI_ONLY":
                return set()
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")

    def get_full_bg_edge_types(self):
            """Return edge types defined by the complete background graph."""
            edge_types = set()

            def _collect_edge_types(df, rel_name):
                if df is None or df.empty:
                    return
                type_pairs = df[["head_type", "tail_type"]].drop_duplicates()
                for src_t, dst_t in type_pairs.itertuples(index=False, name=None):
                    edge_types.add((str(src_t), rel_name, str(dst_t)))

            _collect_edge_types(self.ppi_df, "PPI")
            _collect_edge_types(self.mpi_df, "MPI")
            _collect_edge_types(self.gpi_df, "GPI")
            _collect_edge_types(self.ggi_df, "GGI")
            _collect_edge_types(self.cpi_pos_df, "CPI")

            return sorted(edge_types)

    def _make_edges_by_df(self, df, rel_name, device=None):
            if device is None:
                device = self.device
            edge_dict = {}
            if df is None or df.empty:
                return edge_dict
            for (src_t, dst_t), sub in df.groupby(['head_type','tail_type']):
                idx = torch.tensor(sub['head_id'].values, dtype=torch.long, device=device)
                idy = torch.tensor(sub['tail_id'].values, dtype=torch.long, device=device)
                if idx.numel() > 0:
                    edge_dict[(src_t, rel_name, dst_t)] = torch.stack([idx, idy], dim=0)
            return edge_dict

    def _assign_edges_to_hetero(self, data: HeteroData, edge_dict: dict):
            """
            Write {(src, rel, dst): edge_index} directly to the internal HeteroData storage.
            Note: data.edge_index_dict = {...} and update(...) can no longer be used.
            """
            for k, e in edge_dict.items():
                data[k].edge_index = e

    def build_bg_graph(self, included_rels: set):
            data = HeteroData()
            device = self.device
            edges = {}

            if "PPI" in included_rels:
                edges.update(self._make_edges_by_df(self.ppi_df, "PPI", device))
            if "MPI" in included_rels:
                edges.update(self._make_edges_by_df(self.mpi_df, "MPI", device))
            if "GPI" in included_rels:
                edges.update(self._make_edges_by_df(self.gpi_df, "GPI", device))
            if "GGI" in included_rels:
                edges.update(self._make_edges_by_df(self.ggi_df, "GGI", device))
            if "CPI" in included_rels:
                edges.update(self._make_edges_by_df(self.cpi_pos_df, "CPI", device))

            self._assign_edges_to_hetero(data, edges)
            return data

    def add_cpi_edges(self, data: HeteroData, cpi_df_subset):
            if cpi_df_subset is None or cpi_df_subset.empty:
                return data
            device = self.device
            edge_dict = self._make_edges_by_df(cpi_df_subset, 'CPI', device)
            self._assign_edges_to_hetero(data, edge_dict)
            return data

    def map_triplets(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Map (head, relation, tail) to ID triplets based on head_type/tail_type.
            """
            df = df.copy()

            # Map head_id and tail_id
            # Use a vectorized approach directly: generate key columns, then map
            df["key_head"] = list(zip(df["head"], df["head_type"]))
            df["key_tail"] = list(zip(df["tail"], df["tail_type"]))
            df["head_id"] = df["key_head"].map(self.ent2id)
            df["tail_id"] = df["key_tail"].map(self.ent2id)
            df = df.dropna(subset=["head_id", "tail_id"])
            df[["head_id", "tail_id"]] = df[["head_id", "tail_id"]].astype(int)
            df['relation_id'] = df['relation'].map(self.rel2id)

            return df
