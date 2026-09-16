from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from .config import CLASS2REL as CGI_CLASS2REL
from .config import LABEL_ORDER as CGI_LABEL_ORDER
from .config import REL2CLASS as CGI_REL2CLASS
from .config import PROJECT_ROOT
from .entity_expander import TrainingEntityExpander
from .finetune import (
    _is_decoder_relation_weight, _remap_relation_weight,
    build_cpi_pretrain_rel2id, build_finetune_optimizer,
    load_cpi_pretrained_for_cgi, set_encoder_trainable,
    set_frozen_encoder_eval_mode,
)
from .graph_builder import TrainingGraphBuilder
from .metrics import compute_ranking_metrics
from .model import *
from .results import try_merge_hgt_cgi_summaries
from .scoring import CGITrainingScoring, CGI_SCORE_BATCH_SIZE
from .training_loop import CGITrainingLoop
from .utils import DECODER_BASE_RELATIONS, build_relation_id_map, log_ram

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

logging.basicConfig(filename=str(PROJECT_ROOT / "logs" / "training_CGI.log"), filemode="w", level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)


class FinalTrainer(CGITrainingLoop, CGITrainingScoring, TrainingGraphBuilder, TrainingEntityExpander):

    CKPT_DIR = "result_CGI"

    def __init__(self,
                    train_target_file, val_target_file, 
                    ppi_file, mpi_file, gpi_file, ggi_file,
                    cpi_pos_file,
                    cpd_features_file,
                    protein_features_file,
                    metabolite_features_file,
                    gene_features_file):
                        
            self.ALLOWED_TYPES = {'compound','protein','metabolite','gene'}
            # Support specifying the output directory via CKPT_DIR in sbatch; retain the original default if unset.
            self.CKPT_DIR = os.environ.get("CKPT_DIR", self.CKPT_DIR).strip()
            os.makedirs(self.CKPT_DIR, exist_ok=True)

            # Define device centrally in init
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision('high')  # PyTorch 2.x
                except Exception:
                    pass
        
            self.train_df = pd.read_csv(train_target_file, dtype=str)
            self.val_df   = pd.read_csv(val_target_file,   dtype=str)
            self.train_df["label"] = self.train_df["relation"].map(CGI_REL2CLASS)
            self.val_df["label"] = self.val_df["relation"].map(CGI_REL2CLASS)

            if self.train_df["label"].isna().any():
                bad = self.train_df[self.train_df["label"].isna()]["relation"].unique()
                raise ValueError(f"Unknown CGI relation in train_df: {bad}")

            if self.val_df["label"].isna().any():
                bad = self.val_df[self.val_df["label"].isna()]["relation"].unique()
                raise ValueError(f"Unknown CGI relation in val_df: {bad}")

            self.train_df["label"] = self.train_df["label"].astype(int)
            self.val_df["label"] = self.val_df["label"].astype(int)
            self.ppi_df = pd.read_csv(ppi_file, dtype=str)
            self.mpi_df = pd.read_csv(mpi_file, dtype=str)
            self.gpi_df = pd.read_csv(gpi_file, dtype=str)
            self.ggi_df = pd.read_csv(ggi_file, dtype=str)
            self.cpi_pos_df = pd.read_csv(cpi_pos_file, dtype=str)
            self.cpd_features = pd.read_csv(cpd_features_file, index_col=0, dtype={0: str})
            self.protein_features = pd.read_csv(protein_features_file, index_col=0, dtype={0: str})
            self.protein_features = self.protein_features.apply(pd.to_numeric, errors='coerce').astype('float32')
            self.metabolite_features = pd.read_csv(metabolite_features_file, index_col=0, dtype={0: str}, encoding='utf-8')
            self.cpd_features = self.cpd_features.apply(pd.to_numeric, errors='coerce').astype('float32')
            self.metabolite_features = self.metabolite_features.apply(pd.to_numeric, errors='coerce').astype('float32')
            feature_cols = [f"feature_{i}" for i in range(1280)]
            self.gene_features = pd.read_csv(
                gene_features_file,
                usecols=["gene"] + feature_cols,
                index_col="gene"
            ).astype("float32")
            log_ram("after read csv")

        
            USE_EXTERNAL_INDEX = False  # Change to True to restore the original external index

            if USE_EXTERNAL_INDEX:
                # Load mappings from a pregenerated entity index file
                entity_df = pd.read_csv("entity2id.tsv", sep="\t")
                # Construct three mappings
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
                # Generate automatically from all edges and features
                def _collect_nodes(*dfs):
                    S = set()
                    for df in dfs:
                        if df is None or df.empty: 
                            continue
                        for a,b in [('head','head_type'), ('tail','tail_type')]:
                            if a in df.columns and b in df.columns:
                                S.update(zip(df[a].astype(str), df[b].astype(str)))
                    return S

                log_ram("after build entity maps")

                # Include training/validation CPI, background data, and CGI here
                raw_train = pd.read_csv(train_target_file, dtype=str)
                raw_val   = pd.read_csv(val_target_file,   dtype=str)
                nodes_from_edges = _collect_nodes(
                    raw_train,
                    raw_val,
                    self.ppi_df,
                    self.mpi_df,
                    self.gpi_df,
                    self.ggi_df,
                    self.cpi_pos_df,
                )
                # Also include feature-table indices (using their respective types)
                for ent in self.cpd_features.index.astype(str):
                    nodes_from_edges.add((ent,'compound'))
                for ent in self.metabolite_features.index.astype(str):
                    nodes_from_edges.add((ent,'metabolite'))
                for ent in self.protein_features.index.astype(str):
                    nodes_from_edges.add((ent,'protein'))
                for ent in self.gene_features.index.astype(str):
                    nodes_from_edges.add((ent,'gene'))

                # Stable ordering: group by type, sort by entity name, then combine into global IDs
                type_order = ['compound','metabolite','protein','gene']
                ent_list = []
                for t in type_order:
                    ents = sorted([e for (e,et) in nodes_from_edges if et==t])
                    ent_list.extend([(e,t) for e in ents])

                self.ent2id = {k:i for i,k in enumerate(ent_list)}
                self.id2ent = {i:k[0] for k,i in self.ent2id.items()}
                self.ent2type = {i:k[1] for k,i in self.ent2id.items()}

            self.entity_features = self.create_entity_features()
            #self.entity_features = [feat.to(self.device) for feat in self.entity_features]

            self.rel2id = build_relation_id_map(
                DECODER_BASE_RELATIONS,
                ("CPI",),
                CGI_LABEL_ORDER,
            )
            self.id2rel = {v: k for k, v in self.rel2id.items()}
            self.num_rels_total = len(self.rel2id)  # Store for later use by the decoder


            # Correctly map head_id/tail_id in the background graph
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
            self.cpi_pos_df = _map_bg_edges(self.cpi_pos_df)

            logging.info(f"Train set size: {len(self.train_df)}, Valid set size: {len(self.val_df)}")
            logging.info(f"PPI size: {len(self.ppi_df)}, MPI size: {len(self.mpi_df)}")
            logging.info(f"GPI size: {len(self.gpi_df)}, GGI size: {len(self.ggi_df)}")

            logging.info(f"CPI positive background size: {len(self.cpi_pos_df)}")
            log_ram("after bg edges mapped")


            # Default hyperparameters
            self.learning_rate = 1e-3
            self.hgt_emb_dim = 512
            self.num_heads = 4
            self.dropout = 0.2
            self.num_layers = 1

            assert len(self.ent2id) == len(self.entity_features), \
                f"Entity count {len(self.ent2id)} does not match feature list length {len(self.entity_features)}"

    ABLATION_MODES = {
        "FULL",
        "NO_CPI",
        "NO_PPI",
        "NO_MPI",
        "NO_GPI",
        "NO_GGI",
        "NO_PROTEIN_EDGES",
        "NO_GENE_EDGES",
        "CGI_ONLY",
    }
