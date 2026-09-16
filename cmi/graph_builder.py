from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import HeteroData


class TrainingGraphBuilder:

    def build_hetero_graph(self, cpi_df, cgi_df):
            data = HeteroData()
            device = self.device

            def _make_edges(df, rel_name):
                edge_dict = {}
                if df is None or df.empty:
                    return edge_dict
                for (src_t, dst_t), sub in df.groupby(['head_type','tail_type']):
                    idx = torch.tensor(sub['head_id'].values, dtype=torch.long, device=device)
                    idy = torch.tensor(sub['tail_id'].values, dtype=torch.long, device=device)
                    if idx.numel() > 0:
                        edge_dict[(src_t, rel_name, dst_t)] = torch.stack([idx, idy], dim=0)
                return edge_dict

            edges = {}
            # 1) CPI
            edges.update(_make_edges(cpi_df, 'CPI'))
            # 2) CGI (possibly including CGI-U/CGI-D and others)
            if cgi_df is not None and not cgi_df.empty:
                for rel in cgi_df['relation'].unique():
                    edges.update(_make_edges(cgi_df[cgi_df['relation'] == rel], rel))
            # 3) Background edges
            for df, rel in [(self.ppi_df,'PPI'), (self.mpi_df,'MPI'),
                            (self.gpi_df,'GPI'), (self.ggi_df,'GGI')]:
                edges.update(_make_edges(df, rel))

            # Write to HeteroData in one operation
            self._assign_edges_to_hetero(data, edges)
            return data

    def get_included_bg_rels(self, ablation_mode: str):
            ablation_mode = ablation_mode.strip().upper()
            assert ablation_mode in self.ABLATION_MODES, \
                f"Unknown ablation mode: {ablation_mode}"

            full_rels = {"CPI", "CGI", "MPI", "PPI", "GPI", "GGI"}

            if ablation_mode == "FULL":
                return full_rels.copy()
            if ablation_mode == "NO_CPI":
                return full_rels - {"CPI"}
            if ablation_mode == "NO_CGI":
                return full_rels - {"CGI"}
            if ablation_mode == "NO_MPI":
                return full_rels - {"MPI"}
            if ablation_mode == "NO_PPI":
                return full_rels - {"PPI"}
            if ablation_mode == "NO_GPI":
                return full_rels - {"GPI"}
            if ablation_mode == "NO_GGI":
                return full_rels - {"GGI"}
            if ablation_mode == "NO_PROTEIN_RELATED":
                return full_rels - {"CPI", "GPI", "MPI", "PPI"}
            if ablation_mode == "NO_GENE_RELATED":
                return full_rels - {"CGI", "GGI", "GPI"}
            if ablation_mode == "CMI_ONLY":
                return set()

            raise ValueError(f"Unknown ablation mode: {ablation_mode}")

    def get_full_bg_edge_types(self):
            """Return HGT metadata edge types for the complete background graph."""
            edge_types = set()

            def _collect_edge_types(df, rel_name):
                if df is None or df.empty:
                    return
                pairs = df[["head_type", "tail_type"]].drop_duplicates()
                for src_t, dst_t in pairs.itertuples(index=False, name=None):
                    edge_types.add((str(src_t), str(rel_name), str(dst_t)))

            _collect_edge_types(self.ppi_df, "PPI")
            _collect_edge_types(self.mpi_df, "MPI")
            _collect_edge_types(self.gpi_df, "GPI")
            _collect_edge_types(self.ggi_df, "GGI")
            _collect_edge_types(self.cpi_pos_df, "CPI")

            if (
                self.cgi_df is not None
                and not self.cgi_df.empty
                and "relation" in self.cgi_df.columns
            ):
                for rel in self.cgi_df["relation"].dropna().unique():
                    _collect_edge_types(
                        self.cgi_df[self.cgi_df["relation"] == rel],
                        str(rel),
                    )

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

    def build_bg_graph(self, included_rels: set, cgi_df_subset=None):
            data = HeteroData()
            device = self.device
            edges = {}

            if "CGI" in included_rels and cgi_df_subset is not None and not cgi_df_subset.empty:
                for rel in cgi_df_subset['relation'].unique():
                    edges.update(self._make_edges_by_df(
                        cgi_df_subset[cgi_df_subset['relation'] == rel], rel, device))

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

    def split_cgi_by_compound(self, all_cgi_df, train_cpi_df, val_cpi_df):
            train_comp_ids = set(train_cpi_df.loc[train_cpi_df['head_type']=='compound','head_id'].tolist())
            val_comp_ids   = set(val_cpi_df.loc[val_cpi_df['head_type']=='compound','head_id'].tolist())
            cgi_train = all_cgi_df[ all_cgi_df['head_type'].eq('compound') & all_cgi_df['head_id'].isin(train_comp_ids) ].copy()
            cgi_val   = all_cgi_df[ all_cgi_df['head_type'].eq('compound') & all_cgi_df['head_id'].isin(val_comp_ids)   ].copy()
            # If there are CGI entries with head_type != 'compound', retain both sides as needed
            extra = all_cgi_df[~all_cgi_df['head_type'].eq('compound')].copy()
            if not extra.empty:
                cgi_train = pd.concat([cgi_train, extra], ignore_index=True)
                cgi_val   = pd.concat([cgi_val,   extra], ignore_index=True)
            return cgi_train, cgi_val

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

    def build_train_val_test_data(self):
            train_data = pd.concat([self.train_df, self.ppi_df, self.mpi_df, self.gpi_df, self.ggi_df, self.cgi_df], ignore_index=True)
            val_data   = pd.concat([self.val_df, self.ppi_df, self.mpi_df, self.gpi_df, self.ggi_df, self.cgi_df], ignore_index=True)
            train_data = self.map_triplets(train_data)
            val_data   = self.map_triplets(val_data)
            logging.info(f"Train triplets: {train_data.head()}")
            logging.info(f"Validation triplets: {val_data.head()}")
            train_cpd = train_data[train_data['relation_id'] == self.rel2id['CPI']]
            val_cpd   = val_data[  val_data['relation_id']   == self.rel2id['CPI']]

            return train_data, val_data, train_cpd, val_cpd
