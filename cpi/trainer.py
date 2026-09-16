# trainer.py

import copy
import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import (
    ALLOWED_TYPES,
    DEFAULT_LOG_FILE,
    DEFAULT_SUMMARY_PREFIX,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_HGT_EMB_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_HGT_DROPOUT,
    DEFAULT_MLP_DROPOUT,
    DEFAULT_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MODE,
    SUPPORTED_TRAIN_MODES,
    TRAIN_MODE_CONFIGS,
    normalize_ablation_mode,
)

from .graph_builder import (
    add_cpi_edges as add_cpi_edges_fn,
    build_bg_graph as build_bg_graph_fn,
    build_hetero_graph as build_hetero_graph_fn,
    get_included_bg_rels as get_included_bg_rels_fn,
    log_cgi_usage as log_cgi_usage_fn,
    split_cgi_by_compound as split_cgi_by_compound_fn,
)
from .model import HGT, build_decoder
from .predictor import (
    predict_unlabeled_all_impl,
    predict_unlabeled_topk_impl,
    save_predictions_impl,
    test_external_impl,
)
from .utils import (
    DECODER_BASE_RELATIONS,
    _read_table,
    build_relation_id_map,
    compute_ranking_metrics,
    log_ram,
)

from .training_data import CPITrainingData
from .training_graph import CPITrainingGraph
from .training_inference import CPITrainingInference
from .training_loop import CPITrainingLoop

class FinalTrainer(CPITrainingLoop, CPITrainingData, CPITrainingGraph, CPITrainingInference):

    def __init__(
            self,
            ppi_file,
            mpi_file,
            gpi_file,
            ggi_file,
            cgi_file,
            cpd_features_file,
            protein_features_file,
            metabolite_features_file,
            gene_features_file,
            train_target_file=None,
            val_target_file=None,
        ):
            self.ALLOWED_TYPES = ALLOWED_TYPES
            self.ablation_mode = normalize_ablation_mode(
                os.environ.get("ABLATION_MODE", "FULL")
            )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

            if (train_target_file is None) != (val_target_file is None):
                raise ValueError("train_target_file and val_target_file must be provided together")

            self.train_df = _read_table(train_target_file, dtype=str) if train_target_file is not None else None
            self.val_df = _read_table(val_target_file, dtype=str) if val_target_file is not None else None
            self.ppi_df = _read_table(ppi_file, dtype=str)
            self.mpi_df = _read_table(mpi_file, dtype=str)
            self.gpi_df = _read_table(gpi_file, dtype=str)
            self.ggi_df = _read_table(ggi_file, dtype=str)
            self.cgi_df = _read_table(cgi_file, dtype=str)

            self.cpd_features = _read_table(cpd_features_file, index_col=0, dtype={0: str})
            self.protein_features = _read_table(protein_features_file, index_col=0, dtype={0: str})
            self.protein_features = self.protein_features.apply(pd.to_numeric, errors="coerce").astype("float32")
            self.metabolite_features = _read_table(
                metabolite_features_file,
                index_col=0,
                dtype={0: str},
                encoding="utf-8",
            )
            self.cpd_features = self.cpd_features.apply(pd.to_numeric, errors="coerce").astype("float32")
            self.metabolite_features = self.metabolite_features.apply(pd.to_numeric, errors="coerce").astype("float32")

            feature_cols = [f"feature_{i}" for i in range(1280)]
            self.gene_features = _read_table(
                gene_features_file,
                usecols=["gene"] + feature_cols,
                index_col="gene",
            ).astype("float32")
            log_ram("after read csv")

            USE_EXTERNAL_INDEX = False

            if USE_EXTERNAL_INDEX:
                entity_df = _read_table(
                    "entity2id.tsv",
                    sep="\t",
                )
                self.ent2id = {
                    (row["entity"], row["entity_type"]): int(row["entity_id"])
                    for _, row in entity_df.iterrows()
                }
                self.id2ent = {
                    int(row["entity_id"]): row["entity"]
                    for _, row in entity_df.iterrows()
                }
                self.ent2type = {
                    int(row["entity_id"]): row["entity_type"]
                    for _, row in entity_df.iterrows()
                }
            else:
                def _collect_nodes(*dfs):
                    S = set()
                    for df in dfs:
                        if df is None or df.empty:
                            continue
                        for a, b in [("head", "head_type"), ("tail", "tail_type")]:
                            if a in df.columns and b in df.columns:
                                S.update(zip(df[a].astype(str), df[b].astype(str)))
                    return S

                log_ram("after build entity maps")

                nodes_from_edges = _collect_nodes(
                    self.train_df,
                    self.val_df,
                    self.ppi_df,
                    self.mpi_df,
                    self.gpi_df,
                    self.ggi_df,
                    self.cgi_df,
                )

                for ent in self.cpd_features.index.astype(str):
                    nodes_from_edges.add((ent, "compound"))
                for ent in self.metabolite_features.index.astype(str):
                    nodes_from_edges.add((ent, "metabolite"))
                for ent in self.protein_features.index.astype(str):
                    nodes_from_edges.add((ent, "protein"))
                for ent in self.gene_features.index.astype(str):
                    nodes_from_edges.add((ent, "gene"))

                type_order = ["compound", "metabolite", "protein", "gene"]
                ent_list = []
                for t in type_order:
                    ents = sorted([e for (e, et) in nodes_from_edges if et == t])
                    ent_list.extend([(e, t) for e in ents])

                self.ent2id = {k: i for i, k in enumerate(ent_list)}
                self.id2ent = {i: k[0] for k, i in self.ent2id.items()}
                self.ent2type = {i: k[1] for k, i in self.ent2id.items()}

            self.entity_features = self.create_entity_features()

            cgi_rels = set(self.cgi_df["relation"].unique()) if not self.cgi_df.empty else set()
            self.rel2id = build_relation_id_map(
                DECODER_BASE_RELATIONS,
                cgi_rels,
                ("CPI",),
            )
            self.id2rel = {v: k for k, v in self.rel2id.items()}
            self.num_rels_total = len(self.rel2id)

            def _map_bg_edges(df):
                df["key_head"] = list(zip(df["head"], df["head_type"]))
                df["key_tail"] = list(zip(df["tail"], df["tail_type"]))
                df["head_id"] = df["key_head"].map(self.ent2id)
                df["tail_id"] = df["key_tail"].map(self.ent2id)
                df = df.dropna(subset=["head_id", "tail_id"])
                df[["head_id", "tail_id"]] = df[["head_id", "tail_id"]].astype(int)
                return df

            self.ppi_df = _map_bg_edges(self.ppi_df)
            self.mpi_df = _map_bg_edges(self.mpi_df)
            self.gpi_df = _map_bg_edges(self.gpi_df)
            self.ggi_df = _map_bg_edges(self.ggi_df)
            self.cgi_df = _map_bg_edges(self.cgi_df)

            if self.train_df is not None:
                logging.info(f"Train set size: {len(self.train_df)}, Valid set size: {len(self.val_df)}")
            logging.info(f"PPI size: {len(self.ppi_df)}, MPI size: {len(self.mpi_df)}")
            logging.info(f"GPI size: {len(self.gpi_df)}, GGI size: {len(self.ggi_df)}")
            logging.info(f"CGI size: {len(self.cgi_df)}")
            log_ram("after bg edges mapped")

            assert len(self.ent2id) == len(self.entity_features), (
                f"Entity count {len(self.ent2id)} does not match feature list length {len(self.entity_features)}"
            )

            self.train_mode = os.environ.get("TRAIN_MODE", DEFAULT_TRAIN_MODE)
            assert self.train_mode in SUPPORTED_TRAIN_MODES, (
                f"Unknown TRAIN_MODE={self.train_mode}, supported={SUPPORTED_TRAIN_MODES}"
            )

            mode_cfg = TRAIN_MODE_CONFIGS[self.train_mode]

            self.learning_rate = mode_cfg.get("learning_rate", DEFAULT_LEARNING_RATE)
            self.weight_decay = mode_cfg.get("weight_decay", DEFAULT_WEIGHT_DECAY)

            self.hgt_emb_dim = mode_cfg.get("hgt_emb_dim", DEFAULT_HGT_EMB_DIM)
            self.num_heads = mode_cfg.get("num_heads", DEFAULT_NUM_HEADS)
            self.num_layers = mode_cfg.get("num_layers", DEFAULT_NUM_LAYERS)
            self.num_epochs = mode_cfg.get("num_epochs", DEFAULT_NUM_EPOCHS)

            self.hgt_dropout = mode_cfg.get("hgt_dropout", DEFAULT_HGT_DROPOUT)
            self.mlp_dropout = mode_cfg.get("mlp_dropout", DEFAULT_MLP_DROPOUT)
            self.mlp_hidden_dim = mode_cfg.get("mlp_hidden_dim", DEFAULT_MLP_HIDDEN_DIM)

            ckpt_base_dir = mode_cfg.get("ckpt_dir")
            self.CKPT_DIR = os.path.join(ckpt_base_dir, self.ablation_mode)
            self.log_file = mode_cfg.get("log_file", DEFAULT_LOG_FILE)
            summary_prefix = mode_cfg.get("summary_prefix", DEFAULT_SUMMARY_PREFIX)
            self.summary_prefix = f"{summary_prefix}_{self.ablation_mode}"


            logging.info(
                f"[TRAIN_MODE] mode={self.train_mode}, "
                f"ablation_mode={self.ablation_mode}, "
                f"lr={self.learning_rate}, wd={self.weight_decay}, "
                f"hgt_emb_dim={self.hgt_emb_dim}, num_heads={self.num_heads}, "
                f"num_layers={self.num_layers}, num_epochs={self.num_epochs}, "
                f"hgt_dropout={self.hgt_dropout}, mlp_dropout={self.mlp_dropout}, "
                f"mlp_hidden_dim={self.mlp_hidden_dim}, ckpt_dir={self.CKPT_DIR}"
            )
