from __future__ import annotations

import copy
import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from checkpoint_io import (
    load_checkpoint_bundle as _load_training_checkpoint,
    remove_checkpoint_bundle as _remove_checkpoint_and_config,
    remove_inference_config as _remove_checkpoint_config,
    save_inference_config as _save_checkpoint_config,
    torch_load_checkpoint as _torch_load_checkpoint,
)

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from .config import LABEL_ORDER as CGI_LABEL_ORDER
from .finetune import (
    build_cpi_pretrain_rel2id, build_finetune_optimizer,
    load_cpi_pretrained_for_cgi, set_encoder_trainable,
    set_frozen_encoder_eval_mode,
)
from .graph_builder import get_edge_index_dict_safe
from .model import HGT, build_decoder
from .scoring import CGI_SCORE_BATCH_SIZE
from .utils import log_ram


def _metric_of_legacy_checkpoint(path):
    try:
        checkpoint = _torch_load_checkpoint(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return float(checkpoint.get("metric", float("-inf")))
    except Exception:
        pass
    return float("-inf")


class CGITrainingLoop:

    def train_and_test(self, fold, resume: bool = False):
            device = self.device
            start_epoch = 0
            best_val_loss = float('inf')
            # Initialize: maximize the metric
            best_metric = float('-inf')
            best_epoch = -1
            best_path = None
            saved_checkpoint_metrics = {}

            # Inspect the first five entities
            for i in range(5):
                t = self.ent2type[i]
                f = self.entity_features[i].shape
                print(f"Entity {self.id2ent[i]:10s} type={t:10s} feat_shape={f}")

            # 1) Prepare train/val CGI target triplets
            train_cgi = self.map_triplets(self.train_df)
            val_cgi   = self.map_triplets(self.val_df)

            # Training CPI positive edges used only for graph construction (label == 1)
            #train_cpi_pos = train_cpi[
            #    (train_cpi['relation_id'] == self.rel2id['CPI']) &
            #    (train_cpi['label'].astype(int) == 1)
            #].copy()
    #
            ## Deduplicate to avoid adding edges repeatedly (retain directionality without making edges bidirectional)
            #train_cpi_pos = train_cpi_pos.drop_duplicates(subset=['head_id','tail_id'])
            train_cgi_edges = train_cgi.drop_duplicates(
                subset=["head_id", "relation_id", "tail_id"]
            ).copy()

            # 2) Construct the training and validation graphs
            # CGI targets remain supervised triplets and are not message-passing edges.
            ABLATION_MODE = os.environ.get("ABLATION_MODE", "FULL").strip().upper()
            included_rels = self.get_included_bg_rels(ABLATION_MODE)
            logging.info(
                f"[CFG] ABLATION_MODE={ABLATION_MODE}, "
                f"included_rels={sorted(included_rels)}"
            )

            USE_TRAIN_CGI_IN_VAL_GRAPH = os.environ.get("VAL_USE_TRAIN_CGI", "1") == "1"

            import copy

            bg_data = self.build_bg_graph(included_rels)

            #data_train = copy.deepcopy(bg_data)
            #for rel in CGI_LABEL_ORDER:
            #    sub = train_cgi_edges[train_cgi_edges["relation"] == rel]
            #    self._assign_edges_to_hetero(
            #        data_train,
            #        self._make_edges_by_df(sub, rel, device)
            #    )
    #
            #data_val = copy.deepcopy(bg_data)
            #if USE_TRAIN_CGI_IN_VAL_GRAPH:
            #    for rel in CGI_LABEL_ORDER:
            #        sub = train_cgi_edges[train_cgi_edges["relation"] == rel]
            #        self._assign_edges_to_hetero(
            #            data_val,
            #            self._make_edges_by_df(sub, rel, device)
            #        )
            data_train = copy.deepcopy(bg_data)
            data_val = copy.deepcopy(bg_data)

            logging.info(
                "[CFG] Train CGI target edges are NOT added into HGT message-passing graph; "
                "they are used only as supervised triplets."
            )

            # Validation CGI targets must never appear in the validation graph.
            val_edge_index_dict = get_edge_index_dict_safe(data_val)
            for rel in CGI_LABEL_ORDER:
                etype = ("compound", rel, "gene")
                if etype in val_edge_index_dict:
                    E = val_edge_index_dict[etype]
                    val_pairs = set(zip(val_cgi["head_id"].tolist(), val_cgi["tail_id"].tolist()))
                    graph_pairs = set(zip(
                        E[0].detach().cpu().tolist(),
                        E[1].detach().cpu().tolist(),
                    ))
                    leak = val_pairs.intersection(graph_pairs)
                    assert len(leak) == 0, f"Data leakage: validation-set {rel} CGI appears in the validation graph!"

            # Move to the device
            #for d in (data_train, data_val):
            #    for k, e in d.edge_index_dict.items():
            #        d.edge_index_dict[k] = e.to(device)
            data_train = data_train.to(device)
            data_val   = data_val.to(device)
            log_ram("after build hetero graphs")


            def _log_graph_stats(name, data):
                edge_index_dict = get_edge_index_dict_safe(data)
                keys = list(edge_index_dict.keys())
                logging.info(f"[{name}] edge types: {keys}")
                if not keys:
                    logging.info(f"[{name}] graph contains no message-passing edges.")
                    return
                for etype in keys:
                    eidx = edge_index_dict[etype]
                    logging.info(f"[{name}] {etype}: E={eidx.size(1)}")

            _log_graph_stats("TRAIN", data_train)
            _log_graph_stats("VAL",   data_val)

            train_edge_index_dict = get_edge_index_dict_safe(data_train)
            val_edge_index_dict = get_edge_index_dict_safe(data_val)

            if ABLATION_MODE == "CGI_ONLY":
                assert not train_edge_index_dict, \
                    "CGI_ONLY should not contain background edges in TRAIN graph."
                assert not val_edge_index_dict, \
                    "CGI_ONLY should not contain background edges in VAL graph."
                logging.info(
                    "[CHECK] CGI_ONLY: TRAIN/VAL graphs contain no background edges; "
                    "CGI-N/CGI-D/CGI-U remain supervised targets."
                )
            elif ABLATION_MODE == "NO_PROTEIN_EDGES":
                removed_rels = {"CPI", "PPI", "MPI", "GPI"}
                train_active_rels = {etype[1] for etype in train_edge_index_dict}
                val_active_rels = {etype[1] for etype in val_edge_index_dict}

                assert not train_active_rels.intersection(removed_rels), \
                    "NO_PROTEIN_EDGES TRAIN graph contains a removed relation."
                assert not val_active_rels.intersection(removed_rels), \
                    "NO_PROTEIN_EDGES VAL graph contains a removed relation."
                assert train_active_rels.issubset({"GGI"}), \
                    f"NO_PROTEIN_EDGES TRAIN graph contains unexpected relations: {sorted(train_active_rels)}"
                assert val_active_rels.issubset({"GGI"}), \
                    f"NO_PROTEIN_EDGES VAL graph contains unexpected relations: {sorted(val_active_rels)}"
                assert train_edge_index_dict, \
                    "NO_PROTEIN_EDGES TRAIN graph should retain GGI edges."
                assert val_edge_index_dict, \
                    "NO_PROTEIN_EDGES VAL graph should retain GGI edges."
                logging.info(
                    "[CHECK] NO_PROTEIN_EDGES: CPI/PPI/MPI/GPI are absent from "
                    "TRAIN/VAL graphs; only GGI remains."
                )
            elif ABLATION_MODE == "NO_GENE_EDGES":
                removed_rels = {"GPI", "GGI"}
                expected_rels = {"CPI", "PPI", "MPI"}
                train_active_rels = {etype[1] for etype in train_edge_index_dict}
                val_active_rels = {etype[1] for etype in val_edge_index_dict}

                assert not train_active_rels.intersection(removed_rels), \
                    "NO_GENE_EDGES TRAIN graph contains a removed relation."
                assert not val_active_rels.intersection(removed_rels), \
                    "NO_GENE_EDGES VAL graph contains a removed relation."
                assert train_active_rels.issubset(expected_rels), \
                    f"NO_GENE_EDGES TRAIN graph contains unexpected relations: {sorted(train_active_rels)}"
                assert val_active_rels.issubset(expected_rels), \
                    f"NO_GENE_EDGES VAL graph contains unexpected relations: {sorted(val_active_rels)}"
                assert train_edge_index_dict, \
                    "NO_GENE_EDGES TRAIN graph should retain CPI/PPI/MPI edges."
                assert val_edge_index_dict, \
                    "NO_GENE_EDGES VAL graph should retain CPI/PPI/MPI edges."
                logging.info(
                    "[CHECK] NO_GENE_EDGES: GPI/GGI are absent from TRAIN/VAL "
                    "graphs; CPI/PPI/MPI remain."
                )
            else:
                assert train_edge_index_dict, "Train graph has no edges!"
                assert val_edge_index_dict, "Val graph has no edges!"


            # 4) Prepare the corresponding CPI triplet tensor
            # Use map_triplets to convert the DataFrame to an ID tensor
            def _to_tensor(df):
                arr = df[['head_id','relation_id','tail_id']].values.astype(int)
                return torch.from_numpy(arr).long().to(device)

            train_labels = torch.tensor(
                train_cgi["label"].astype(int).values,
                dtype=torch.long
            ).to(device)

            val_labels = torch.tensor(
                val_cgi["label"].astype(int).values,
                dtype=torch.long
            ).to(device)
        
            # Added section
            feat_dims = {
                'compound':   self.cpd_features.shape[1],
                'metabolite': self.metabolite_features.shape[1],
                'protein':    self.protein_features.shape[1],
                'gene':       self.gene_features.shape[1],
            }
        
            # Keep HGT relation-specific parameters identical across ablations.
            node_types = list(feat_dims.keys())
            train_edge_index_dict = get_edge_index_dict_safe(data_train)
            val_edge_index_dict = get_edge_index_dict_safe(data_val)
            active_edge_types = sorted(
                set(train_edge_index_dict) | set(val_edge_index_dict)
            )
            edge_types = self.get_full_bg_edge_types()
            metadata = (node_types, edge_types)
            logging.info(
                f"[ABLATION] mode={ABLATION_MODE}, "
                f"active_edge_types={active_edge_types}"
            )
            logging.info(f"[ABLATION] fixed FULL metadata edge_types={edge_types}")

            # ======== Prepare auxiliary LP sampling: collect positive samples and type-to-node-ID sets from background edges ========
            # Mapping from relations to types (src_t, rel, dst_t)
            rel2types = {}
            for (src_t, rel, dst_t) in edge_types:
                rel2types[rel] = (src_t, dst_t)
            # ID list for each type
            type2ids = {t: torch.tensor([i for i, tp in self.ent2type.items() if tp == t],
                                        device=device, dtype=torch.long)
                        for t in node_types}

            # Collect positive triplets for background relations
            # Note: do not add training CGI target edges here, avoiding overlap between the auxiliary LP loss and the main task.
            aux_pos_by_rel = {}
            def _collect_pos(df, rel_name):
                if df is None or df.empty: 
                    return
                rid = self.rel2id.get(rel_name, None)
                if rid is None: 
                    return
                arr = df[['head_id','tail_id']].values.astype(int)
                if arr.shape[0] == 0:
                    return
                trip = np.concatenate([
                    arr[:, [0]], 
                    np.full((arr.shape[0], 1), rid, dtype=int),
                    arr[:, [1]]
                ], axis=1)  # [E, 3]
                aux_pos_by_rel[rel_name] = torch.from_numpy(trip).long().to(device)

            # Add only background edges included in the current ablation selection
            if "PPI" in included_rels: _collect_pos(self.ppi_df, "PPI")
            if "MPI" in included_rels: _collect_pos(self.mpi_df, "MPI")
            if "GPI" in included_rels: _collect_pos(self.gpi_df, "GPI")
            if "GGI" in included_rels: _collect_pos(self.ggi_df, "GGI")
            if "CPI" in included_rels: _collect_pos(self.cpi_pos_df, "CPI")

            self._aux_lp_pack = {
                "rel2types": rel2types,
                "type2ids": type2ids,
                "aux_pos_by_rel": aux_pos_by_rel
            }

            log_ram("before training")


            #metadata = data.metadata()
            num_rels = self.num_rels_total

            # ===== Unified decoder configuration =====
            decoder_name = os.environ.get("DECODER", "mlp+complex").lower()
            decoder_kwargs = {}
            if "mlp" in decoder_name:
                decoder_kwargs["mlp_hidden_dim"] = int(os.environ.get("MLP_HIDDEN", "512"))
            if "transe" in decoder_name:
                decoder_kwargs["transe_p"] = int(os.environ.get("TRANSE_P", "1"))
                # For composite decoders, the initial gate value can be set (0 => 0.5)
            decoder_kwargs["gate_init"] = float(os.environ.get("GATE_INIT", "0.0"))
            decoder = build_decoder(decoder_name, self.hgt_emb_dim, num_rels, return_logits=True, **decoder_kwargs)
            logging.info(f"[CFG] DECODER={decoder_name}, args={decoder_kwargs}")
            # ====================================

            model = HGT(
                hgt_emb_dim=self.hgt_emb_dim,
                metadata=metadata,         # Note: HGT obtains relation names from rel2id.keys()
                num_heads=self.num_heads,
                ent2type=self.ent2type,
                feat_dims=feat_dims,
                num_layers=self.num_layers,
                num_rels=num_rels,
                #dropout=self.dropout
                return_logits=True,
                decoder=decoder
            ).to(device)

            # ===== CPI -> CGI pretrained transfer learning =====
            # Usage:
            #   export PRETRAIN_CKPT=/path/to/CPI_pretrained_checkpoint.pt
            # If PRETRAIN_CKPT is unset, fall back to training from scratch.
            pretrain_ckpt = os.environ.get("PRETRAIN_CKPT", "").strip()
            if pretrain_ckpt:
                pre_rel2id = build_cpi_pretrain_rel2id(None)
                load_cpi_pretrained_for_cgi(
                    model=model,
                    ckpt_path=pretrain_ckpt,
                    pre_rel2id=pre_rel2id,
                    cur_rel2id=self.rel2id,
                    device=device
                )
            else:
                logging.info("[TRANSFER] PRETRAIN_CKPT is empty. CGI will train from scratch.")
            # ==================================

            # ===== Transfer-learning/control-experiment mode =====
            # full: fine-tune all parameters
            # decoder_only: freeze the HGT encoder and train only the decoder
            # warmup_unfreeze: train only the decoder for the initial epochs, then unfreeze the full model
            finetune_mode = os.environ.get("FINETUNE_MODE", "full").lower()
            freeze_epochs = int(os.environ.get("FREEZE_EPOCHS", "10"))
            finetune_lr = float(os.environ.get("FINETUNE_LR", str(self.learning_rate)))

            if finetune_mode == "full":
                set_encoder_trainable(model, True)
                optimizer = build_finetune_optimizer(model, finetune_lr)
                logging.info(f"[FINETUNE] mode=full: train encoder + decoder, lr={finetune_lr}")

            elif finetune_mode == "decoder_only":
                set_encoder_trainable(model, False)
                optimizer = build_finetune_optimizer(model, finetune_lr)
                logging.info(f"[FINETUNE] mode=decoder_only: freeze encoder, train decoder only, lr={finetune_lr}")

            elif finetune_mode == "warmup_unfreeze":
                set_encoder_trainable(model, False)
                optimizer = build_finetune_optimizer(model, finetune_lr)
                logging.info(
                    f"[FINETUNE] mode=warmup_unfreeze: freeze encoder for first {freeze_epochs} epochs, "
                    f"warmup_lr={finetune_lr}"
                )

            else:
                raise ValueError(f"Unknown FINETUNE_MODE: {finetune_mode}")
            class_counts = train_cgi["label"].value_counts().reindex(
                range(len(CGI_LABEL_ORDER)),
                fill_value=0
            ).values.astype(float)

            class_weights = class_counts.sum() / (
                len(CGI_LABEL_ORDER) * np.maximum(class_counts, 1.0)
            )

            class_weights = torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=device
            )

            criterion = nn.CrossEntropyLoss(weight=class_weights)

            logging.info(f"CGI class counts: {class_counts.tolist()}")
            logging.info(f"CGI class weights: {class_weights.detach().cpu().numpy().tolist()}")

            # If resuming, restore the model and optimizer from the best checkpoint
            if resume:
                ckpt_dir = os.path.join(self.CKPT_DIR, f"fold{fold}")
                pattern = os.path.join(ckpt_dir, f"best_checkpoint_fold{fold}_*.pt")
                checkpoints = glob.glob(pattern)
                if checkpoints:
                    def _metric_of(p):
                        return _metric_of_legacy_checkpoint(p)

                    # Prefer the best checkpoint by metric; if no metric can be read, fall back to the most recent
                    best_ckpt = max(checkpoints, key=_metric_of)
                    if _metric_of(best_ckpt) == float('-inf'):
                        best_ckpt = max(checkpoints, key=os.path.getmtime)

                    logging.info(f"Resuming training from checkpoint {best_ckpt}")
                    state = _load_training_checkpoint(best_ckpt, map_location=device)
                    state_dict = state['state_dict']

                    rel_w = state_dict.get('decoder.rel_embedding.weight')
                    if rel_w is not None and rel_w.shape[0] != self.num_rels_total:
                        logging.warning(f"num_rels mismatch (ckpt={rel_w.shape[0]}, current={self.num_rels_total}). decoder.rel_embedding will be reinitialized using the current configuration.")

                    # Use non-strict loading with a warning for shape changes and similar cases
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if missing or unexpected:
                        logging.warning(f"While loading ckpt: missing_keys={missing}, unexpected_keys={unexpected}")

                    if 'optimizer_state_dict' in state:
                        optimizer.load_state_dict(state['optimizer_state_dict'])
                        # Move tensors in the optimizer state to the current device
                        for s in optimizer.state.values():
                            for k, v in s.items():
                                if isinstance(v, torch.Tensor):
                                    s[k] = v.to(device)

                        start_epoch = int(state.get('epoch', 0))
                        best_val_loss = float(state.get('val_loss', best_val_loss))
                        best_metric = float(state.get('metric', best_metric))
                        best_epoch = int(state.get('epoch', best_epoch))

                        logging.info(f"Resumed at epoch {start_epoch}, val_loss={best_val_loss:.4f}, metric={best_metric:.4f}")
                    else:
                        logging.warning(
                            "Loaded weights-only checkpoint; optimizer state, epoch, and best validation metric "
                            "are not available, so training will continue with a fresh optimizer from epoch 0."
                        )
                else:
                    logging.info("No usable checkpoint found; training from scratch")

            # Automatically reduce the global LR when validation loss does not decrease
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            #    optimizer, mode='min',
            #    factor=0.8, patience=20,
            #    min_lr=1e-4,
            #    verbose=True
            #)

            # Training-loss convergence detection
            #train_loss_history = []
            #convergence_tol       = 1e-3   # Convergence threshold: loss difference between consecutive epochs
            #convergence_start     = 50      # Start checking from epoch 11
            #loss_converged    = False
            ## Counter for consecutive epochs meeting the threshold
            #small_delta_count = 0

            num_epochs = 1000
            log_gate_every = int(os.environ.get("LOG_GATE_EVERY", "1"))  # Print every N epochs; 0 disables printing
            patience = num_epochs
            early_stop_counter = 0
            save_dir =  os.path.join(self.CKPT_DIR, f"fold{fold}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            encoder_unfrozen = (finetune_mode == "full")

            for epoch in range(start_epoch, num_epochs):
                model.train()

                # In decoder_only mode, the encoder is frozen; after model.train() each epoch, switch encoder-related BN/modules back to eval,
                # preventing BatchNorm running_mean/running_var from being updated.
                if finetune_mode == "decoder_only":
                    set_frozen_encoder_eval_mode(model)

                # warmup_unfreeze: train only the decoder for the first freeze_epochs, then unfreeze the encoder and rebuild the optimizer.
                if finetune_mode == "warmup_unfreeze":
                    if epoch < freeze_epochs:
                        set_frozen_encoder_eval_mode(model)
                    elif not encoder_unfrozen:
                        set_encoder_trainable(model, True)
                        unfreeze_lr = float(os.environ.get("UNFREEZE_LR", "1e-4"))
                        optimizer = build_finetune_optimizer(model, unfreeze_lr)
                        encoder_unfrozen = True
                        logging.info(
                            f"[FINETUNE] epoch {epoch+1}: unfreeze encoder, new_lr={unfreeze_lr}"
                        )

                optimizer.zero_grad()

                # Forward pass
                h_train = model(data_train, self.entity_features)

                main_loss_value = self.backward_cgi_loss_batched(
                    model=model,
                    h=h_train,
                    cgi_df=train_cgi,
                    labels=train_labels,
                    criterion=criterion,
                    batch_size=CGI_SCORE_BATCH_SIZE,
                )

                logging.info(
                    f"[Train] Epoch {epoch+1}, "
                    f"batched main_loss={main_loss_value:.4f}, "
                    f"batch_size={CGI_SCORE_BATCH_SIZE}"
                )

                # ======== Auxiliary LP loss (safe version) ========
                lambda_aux = 0       # Weight; values from 0.05~0.2 can be tried
                aux_batch_per_rel = 0  # Number of edges sampled per relation per epoch
                rel2types = self._aux_lp_pack["rel2types"]
                type2ids  = self._aux_lp_pack["type2ids"]
                aux_pos_by_rel = self._aux_lp_pack["aux_pos_by_rel"]

                aux_loss = torch.tensor(0.0, device=device)

                if (lambda_aux > 0) and (aux_batch_per_rel > 0):
                    per_rel_losses = []
                    for rel_name, pos_all in aux_pos_by_rel.items():
                        if pos_all is None or pos_all.numel() == 0:
                            continue
                        b = min(aux_batch_per_rel, pos_all.shape[0])
                        if b <= 0:
                            continue

                        # Sample positive edges
                        idx = torch.randint(pos_all.shape[0], (b,), device=device)
                        pos_batch = pos_all[idx]  # [b,3]

                        # Corrupt negative samples
                        src_t, dst_t = rel2types[rel_name]
                        cand_dst_ids = type2ids[dst_t]
                        neg_batch = pos_batch.clone()
                        neg_batch[:, 2] = cand_dst_ids[torch.randint(
                            0, cand_dst_ids.numel(), (b,), device=device
                        )]

                        # Score
                        pos_s = model.get_score(h_train, pos_batch)
                        neg_s = model.get_score(h_train, neg_batch)
                        target = torch.ones_like(pos_s, device=device)

                        # Valid b >= 1; reduction='mean'
                        per_rel_losses.append(F.margin_ranking_loss(
                            pos_s, neg_s, target, margin=1.0, reduction='mean'
                        ))

                    if len(per_rel_losses):
                        aux_loss = torch.stack(per_rel_losses).mean() * float(lambda_aux)

                loss_value = main_loss_value
                # Clip gradients to suppress large early-stage fluctuations
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    h_val = model(data_val, self.entity_features)

                    logits_val_cpu = self.score_cgi_multiclass_batched_no_grad(
                        model=model,
                        h=h_val,
                        cgi_df=val_cgi,
                        batch_size=CGI_SCORE_BATCH_SIZE,
                    )

                    logits_val = logits_val_cpu.to(device)
                    val_loss = criterion(logits_val, val_labels)

                    probs_val = torch.softmax(logits_val_cpu, dim=1).numpy()
                    y_pred = probs_val.argmax(axis=1)
                    labels_np = val_labels.detach().cpu().numpy()

                    acc = accuracy_score(labels_np, y_pred)
                    balanced_acc = balanced_accuracy_score(labels_np, y_pred)
                    macro_precision = precision_score(
                        labels_np, y_pred, average="macro", zero_division=0
                    )
                    macro_recall = recall_score(
                        labels_np, y_pred, average="macro", zero_division=0
                    )
                    macro_f1 = f1_score(
                        labels_np, y_pred, average="macro", zero_division=0
                    )
                    weighted_f1 = f1_score(
                        labels_np, y_pred, average="weighted", zero_division=0
                    )

                    metric = macro_f1
                #metric = - val_loss.item()

                # Get the learning rate from the optimizer
                current_lr = optimizer.param_groups[0]['lr']

                logging.info(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"lr: {current_lr:.6f}, "
                    f"train_loss={loss_value:.4f}, "
                    f"val_loss={val_loss.item():.4f}, "
                    f"ACC: {acc:.4f}, "
                    f"BalancedACC: {balanced_acc:.4f}, "
                    f"MacroPrecision: {macro_precision:.4f}, "
                    f"MacroRecall: {macro_recall:.4f}, "
                    f"MacroF1: {macro_f1:.4f}, "
                    f"WeightedF1: {weighted_f1:.4f}, "
                    f"Metric: {metric:.4f}"
                )

                # ==== Print the gate distribution (only valid for composite decoders with a gate) ====
                if log_gate_every > 0 and ((epoch + 1) % log_gate_every == 0):
                    dec = getattr(model, "decoder", None)
                    # Unified approach: print whenever the decoder has rel_gate (only DualChannelDecoderGate does)
                    if dec is not None and hasattr(dec, "rel_gate"):
                        with torch.no_grad():
                            alpha = torch.sigmoid(dec.rel_gate.weight.squeeze(1)).detach().cpu().numpy()  # [num_rels]
                        import numpy as _np
                        q25, q50, q75 = _np.quantile(alpha, [0.25, 0.5, 0.75]).tolist()
                        msg = (f"[Gate] epoch {epoch+1}: "
                            f"mean={alpha.mean():.3f}, std={alpha.std():.3f}, "
                            f"min={alpha.min():.3f}, q25/50/75={q25:.3f}/{q50:.3f}/{q75:.3f}, max={alpha.max():.3f}")
                        logging.info(msg)

                        # Inspect CPI separately
                        rid_cpi = self.rel2id.get("CPI", None)
                        if rid_cpi is not None:
                            logging.info(f"[Gate] epoch {epoch+1}: alpha[CPI]={alpha[rid_cpi]:.3f}")

                        # Print several relations with the smallest/largest values (three each)
                        order = _np.argsort(alpha)
                        lows  = [(self.id2rel[i], float(alpha[i])) for i in order[:3]]
                        highs = [(self.id2rel[i], float(alpha[i])) for i in order[-3:]]
                        logging.info(f"[Gate] epoch {epoch+1}: lowest3={lows}; highest3={highs}")
                # ==========================================================

                # Save the best model only after training loss has converged and val_loss is minimal (metric is maximal)
                #if loss_converged and val_loss.item() < best_val_loss:
                #if val_loss.item() < best_val_loss:
                    # Save a complete checkpoint whenever val_loss decreases
                    #best_val_loss = val_loss.item()
                if metric > best_metric:
                    best_metric = float(metric)
                    best_val_loss = float(val_loss.item())
                    best_epoch = epoch + 1

                    fname = f"best_checkpoint_fold{fold}_epoch{best_epoch:03d}_val{best_val_loss:.4f}_metric{best_metric:.4f}_CGI_finetune.pt"
                    checkpoint_path = os.path.join(save_dir, fname)
                    best_path = checkpoint_path
                    saved_checkpoint_metrics[checkpoint_path] = best_metric
                    torch.save(model.state_dict(), checkpoint_path)
                    if "transe" in decoder_name:
                        _save_checkpoint_config(
                            checkpoint_path,
                            {"decoder_kwargs": {"transe_p": int(decoder_kwargs["transe_p"])}},
                        )
                    else:
                        _remove_checkpoint_config(checkpoint_path)

                    early_stop_counter = 0

                    # Retain only the three checkpoints with the highest metrics
                    pattern = os.path.join(save_dir, f"best_checkpoint_fold{fold}_*.pt")
                    all_ckpts = glob.glob(pattern)

                    def _metric_of(p):
                        return saved_checkpoint_metrics.get(p, _metric_of_legacy_checkpoint(p))

                    all_ckpts.sort(key=_metric_of, reverse=True)  # Descending order
                    for p in all_ckpts[3:]:
                        _remove_checkpoint_and_config(p)
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}.")
                    break

            # Determine the best checkpoint path
            pattern = os.path.join(save_dir, f"best_checkpoint_fold{fold}_*.pt")
            all_ckpts = glob.glob(pattern)
            if not all_ckpts:
                # If no checkpoint has been saved, save the final epoch as a fallback
                last_fname = f"best_checkpoint_fold{fold}_epoch{epoch:03d}_val{val_loss:.4f}_metric{metric:.4f}_CGI_finetune.pt"
                best_path = os.path.join(save_dir, last_fname)
                saved_checkpoint_metrics[best_path] = float(metric.item() if hasattr(metric, "item") else metric)
                torch.save(model.state_dict(), best_path)
                if "transe" in decoder_name:
                    _save_checkpoint_config(
                        best_path,
                        {"decoder_kwargs": {"transe_p": int(decoder_kwargs["transe_p"])}},
                    )
                else:
                    _remove_checkpoint_config(best_path)
                logging.info("No better checkpoint found; saved the final-epoch model as the checkpoint")
            elif best_path is None:
                # Load the checkpoint with the highest metric among legacy checkpoints; pure weights fall back to mtime.
                def _metric_of(p):
                    return saved_checkpoint_metrics.get(p, _metric_of_legacy_checkpoint(p))

                best_path = max(all_ckpts, key=_metric_of)
                if _metric_of(best_path) == float('-inf'):
                    best_path = max(all_ckpts, key=os.path.getmtime)
            assert best_path, "Could not determine the best checkpoint path"

            # Load the best model and save the evaluation results
            ckpt = _load_training_checkpoint(best_path, map_location=device)
            best_model = HGT(
                hgt_emb_dim=self.hgt_emb_dim,
                metadata=metadata,
                num_heads=self.num_heads,
                ent2type=self.ent2type,
                feat_dims=feat_dims,         
                num_layers=self.num_layers,
                num_rels=num_rels,
                #dropout=self.dropout
                return_logits=True,
                decoder=decoder
            ).to(device)
            best_model.load_state_dict(ckpt['state_dict'])

            # To continue training from a legacy full checkpoint, the optimizer can be restored:
            # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
            # Recompute embeddings with the best model and save all predictions from that model
            best_model.eval()
            with torch.no_grad():
                h_val_best = best_model(data_val, self.entity_features)
            out_csv = os.path.join(save_dir, f"fold_{fold}_val_predictions.csv")
            self.save_predictions(
                best_model,
                h_val_best,
                val_cgi,
                val_labels,
                out_csv
            )

            # Recompute all metrics with the best model
            logits_best_cpu = self.score_cgi_multiclass_batched_no_grad(
                model=best_model,
                h=h_val_best,
                cgi_df=val_cgi,
                batch_size=CGI_SCORE_BATCH_SIZE,
            )

            probs_best = torch.softmax(logits_best_cpu, dim=1).numpy()
            y_pred_best = probs_best.argmax(axis=1)
            y_true_best = val_labels.cpu().numpy()

            acc_best = accuracy_score(y_true_best, y_pred_best)
            balanced_acc_best = balanced_accuracy_score(y_true_best, y_pred_best)
            macro_precision_best = precision_score(
                y_true_best, y_pred_best, average="macro", zero_division=0
            )
            macro_recall_best = recall_score(
                y_true_best, y_pred_best, average="macro", zero_division=0
            )
            macro_f1_best = f1_score(
                y_true_best, y_pred_best, average="macro", zero_division=0
            )
            weighted_f1_best = f1_score(
                y_true_best, y_pred_best, average="weighted", zero_division=0
            )

            report = classification_report(
                y_true_best,
                y_pred_best,
                target_names=CGI_LABEL_ORDER,
                output_dict=True,
                zero_division=0,
            )

            cm = confusion_matrix(
                y_true_best,
                y_pred_best,
                labels=list(range(len(CGI_LABEL_ORDER)))
            )

            pd.DataFrame(report).transpose().to_csv(
                os.path.join(save_dir, f"fold_{fold}_classification_report.csv")
            )

            pd.DataFrame(
                cm,
                index=[f"true_{x}" for x in CGI_LABEL_ORDER],
                columns=[f"pred_{x}" for x in CGI_LABEL_ORDER],
            ).to_csv(os.path.join(save_dir, f"fold_{fold}_confusion_matrix.csv"))

            return {
                'Best Epoch': best_epoch,
                'ACC': acc_best,
                'BalancedACC': balanced_acc_best,
                'MacroPrecision': macro_precision_best,
                'MacroRecall': macro_recall_best,
                'MacroF1': macro_f1_best,
                'WeightedF1': weighted_f1_best,
            }
