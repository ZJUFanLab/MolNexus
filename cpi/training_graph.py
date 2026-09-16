from __future__ import annotations

from .graph_builder import (
    add_cpi_edges as add_cpi_edges_fn,
    build_bg_graph as build_bg_graph_fn,
    build_hetero_graph as build_hetero_graph_fn,
    get_included_bg_rels as get_included_bg_rels_fn,
    log_cgi_usage as log_cgi_usage_fn,
    split_cgi_by_compound as split_cgi_by_compound_fn,
)


class CPITrainingGraph:

    def get_included_bg_rels(self, ablation_mode: str):
            return get_included_bg_rels_fn(ablation_mode)

    def build_hetero_graph(self, cpi_df, cgi_df):
            return build_hetero_graph_fn(
                cpi_df,
                cgi_df,
                self.ppi_df,
                self.mpi_df,
                self.gpi_df,
                self.ggi_df,
                self.device,
            )

    def build_bg_graph(self, included_rels: set, cgi_df_subset=None):
            return build_bg_graph_fn(
                included_rels,
                cgi_df_subset,
                self.ppi_df,
                self.mpi_df,
                self.gpi_df,
                self.ggi_df,
                self.device,
            )

    def add_cpi_edges(self, data, cpi_df_subset):
            return add_cpi_edges_fn(data, cpi_df_subset, self.device)

    def split_cgi_by_compound(self, all_cgi_df, train_cpi_df, val_cpi_df):
            return split_cgi_by_compound_fn(all_cgi_df, train_cpi_df, val_cpi_df)

    def _log_cgi_usage(self, tag, data, ext_cgi_mapped, compounds):
            return log_cgi_usage_fn(tag, data, self.cgi_df, ext_cgi_mapped, compounds, self.ent2id)
