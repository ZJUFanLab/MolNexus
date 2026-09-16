from __future__ import annotations

import copy
import glob
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from checkpoint_io import (
    load_checkpoint_bundle as _load_training_checkpoint,
    remove_checkpoint_bundle as _remove_checkpoint_and_config,
    save_inference_config as _save_checkpoint_config,
    torch_load_checkpoint as _torch_load_checkpoint,
)

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

from .config import CGI_GLOBAL, CGI_SPLIT, normalize_ablation_mode
from .inference_builder import resolve_cpi_threshold
from .model import HGT, build_decoder
from .utils import compute_ranking_metrics


def _metric_of_legacy_checkpoint(path):
    try:
        checkpoint = _torch_load_checkpoint(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return float(checkpoint.get("metric", float("-inf")))
    except Exception:
        pass
    return float("-inf")


class CPITrainingLoop:

    def train_and_test(self, fold, resume: bool = False):
            device = self.device
            start_epoch = 0
            best_val_loss = float("inf")
            best_metric = float("-inf")
            best_epoch = -1
            best_path = None
            saved_checkpoint_metrics = {}

            for i in range(5):
                t = self.ent2type[i]
                f = self.entity_features[i].shape
                print(f"Entity {self.id2ent[i]:10s} type={t:10s} feat_shape={f}")

            train_cpi = self.map_triplets(self.train_df)
            val_cpi = self.map_triplets(self.val_df)

            train_cpi_pos = train_cpi[
                (train_cpi["relation_id"] == self.rel2id["CPI"])
                & (train_cpi["label"].astype(int) == 1)
            ].copy()
            train_cpi_pos = train_cpi_pos.drop_duplicates(subset=["head_id", "tail_id"])

            CGI_MODE = os.environ.get("CGI_MODE", CGI_GLOBAL)
            included_rels = self.get_included_bg_rels(self.ablation_mode)
            logging.info(
                f"[CFG] CGI_MODE={CGI_MODE}, ABLATION_MODE={self.ablation_mode}, "
                f"included_rels={sorted(included_rels)}"
            )

            USE_TRAIN_CPI_IN_VAL_GRAPH = os.environ.get("VAL_USE_TRAIN_CPI", "1") == "1"

            if CGI_MODE == CGI_GLOBAL:
                all_deg = self.map_triplets(self.cgi_df)
                cgi_for_bg = all_deg if ("CGI" in included_rels) else None

                bg_data = self.build_bg_graph(included_rels, cgi_for_bg)
                data_train = self.add_cpi_edges(copy.deepcopy(bg_data), train_cpi_pos)

                if USE_TRAIN_CPI_IN_VAL_GRAPH:
                    data_val = self.add_cpi_edges(copy.deepcopy(bg_data), train_cpi_pos)
                else:
                    data_val = copy.deepcopy(bg_data)

            elif CGI_MODE == CGI_SPLIT:
                all_deg = self.map_triplets(self.cgi_df)
                cgi_train, cgi_val = self.split_cgi_by_compound(all_deg, train_cpi_pos, val_cpi)

                cgi_train = cgi_train if ("CGI" in included_rels) else None
                cgi_val = cgi_val if ("CGI" in included_rels) else None

                bg_train = self.build_bg_graph(included_rels, cgi_train)
                data_train = self.add_cpi_edges(copy.deepcopy(bg_train), train_cpi_pos)

                bg_val = self.build_bg_graph(included_rels, cgi_val)
                if USE_TRAIN_CPI_IN_VAL_GRAPH:
                    data_val = self.add_cpi_edges(copy.deepcopy(bg_val), train_cpi_pos)
                else:
                    data_val = copy.deepcopy(bg_val)
            else:
                raise ValueError(f"Unknown CGI_MODE: {CGI_MODE}")

            if ("compound", "CPI", "protein") in data_val.edge_types:
                E = data_val[("compound", "CPI", "protein")].edge_index
                val_pairs = set(zip(val_cpi["head_id"].tolist(), val_cpi["tail_id"].tolist()))
                leak = val_pairs.intersection(set(zip(E[0].tolist(), E[1].tolist())))
                assert len(leak) == 0, "Data leakage: validation-set CPI appears in the validation graph!"

            data_train = data_train.to(device)
            data_val = data_val.to(device)
            log_ram("after build hetero graphs")

            def _log_graph_stats(name, data):
                keys = list(data.edge_types)
                logging.info(f"[{name}] edge types: {keys}")
                for etype in keys:
                    eidx = data[etype].edge_index
                    logging.info(f"[{name}] {etype}: E={eidx.size(1)}")

            _log_graph_stats("TRAIN", data_train)
            _log_graph_stats("VAL", data_val)

            assert len(data_train.edge_types) > 0, "Train graph has no edges!"
            if len(data_val.edge_types) == 0:
                if self.ablation_mode == "CPI_ONLY" and not USE_TRAIN_CPI_IN_VAL_GRAPH:
                    logging.info(
                        "[VAL] CPI_ONLY with VAL_USE_TRAIN_CPI=0 uses feature-only "
                        "validation propagation"
                    )
                else:
                    raise AssertionError("Validation graph has no edges")

            def _to_tensor(df):
                arr = df[["head_id", "relation_id", "tail_id"]].values.astype(int)
                return torch.from_numpy(arr).long().to(device)

            train_triplets_cpi = _to_tensor(train_cpi).to(device)
            val_triplets_cpi = _to_tensor(val_cpi).to(device)

            train_labels = torch.tensor(
                self.train_df["label"].astype(int).values,
                dtype=torch.float32,
            ).to(device)
            val_labels = torch.tensor(
                self.val_df["label"].astype(int).values,
                dtype=torch.float32,
            ).to(device)

            feat_dims = {
                "compound": self.cpd_features.shape[1],
                "metabolite": self.metabolite_features.shape[1],
                "protein": self.protein_features.shape[1],
                "gene": self.gene_features.shape[1],
            }

            node_types = list(feat_dims.keys())
            edge_types_train = list(data_train.edge_types)
            edge_types_val = list(data_val.edge_types)
            edge_types = sorted(set(edge_types_train) | set(edge_types_val))
            metadata = (node_types, edge_types)

            rel2types = {}
            for (src_t, rel, dst_t) in edge_types:
                rel2types[rel] = (src_t, dst_t)

            type2ids = {
                t: torch.tensor(
                    [i for i, tp in self.ent2type.items() if tp == t],
                    device=device,
                    dtype=torch.long,
                )
                for t in node_types
            }

            aux_pos_by_rel = {}

            def _collect_pos(df, rel_name):
                if df is None or df.empty:
                    return
                rid = self.rel2id.get(rel_name, None)
                if rid is None:
                    return
                arr = df[["head_id", "tail_id"]].values.astype(int)
                if arr.shape[0] == 0:
                    return
                trip = np.concatenate(
                    [
                        arr[:, [0]],
                        np.full((arr.shape[0], 1), rid, dtype=int),
                        arr[:, [1]],
                    ],
                    axis=1,
                )
                aux_pos_by_rel[rel_name] = torch.from_numpy(trip).long().to(device)

            if "PPI" in included_rels:
                _collect_pos(self.ppi_df, "PPI")
            if "MPI" in included_rels:
                _collect_pos(self.mpi_df, "MPI")
            if "GPI" in included_rels:
                _collect_pos(self.gpi_df, "GPI")
            if "GGI" in included_rels:
                _collect_pos(self.ggi_df, "GGI")
            if "CGI" in included_rels and ("relation" in self.cgi_df.columns) and (not self.cgi_df.empty):
                for r in self.cgi_df["relation"].unique():
                    _collect_pos(self.cgi_df[self.cgi_df["relation"] == r], r)

            self._aux_lp_pack = {
                "rel2types": rel2types,
                "type2ids": type2ids,
                "aux_pos_by_rel": aux_pos_by_rel,
            }

            log_ram("before training")

            num_rels = self.num_rels_total

            decoder_name = os.environ.get("DECODER", "mlp+complex").lower()
            decoder_kwargs = {}
            if "mlp" in decoder_name:
                decoder_kwargs["mlp_hidden_dim"] = self.mlp_hidden_dim
                decoder_kwargs["mlp_dropout"] = self.mlp_dropout
            if "transe" in decoder_name:
                decoder_kwargs["transe_p"] = int(os.environ.get("TRANSE_P", "1"))
            decoder_kwargs["gate_init"] = float(os.environ.get("GATE_INIT", "0.0"))

            decoder = build_decoder(
                decoder_name,
                self.hgt_emb_dim,
                num_rels,
                return_logits=True,
                **decoder_kwargs,
            )
            logging.info(f"[CFG] DECODER={decoder_name}, args={decoder_kwargs}")

            model = HGT(
                hgt_emb_dim=self.hgt_emb_dim,
                metadata=metadata,
                num_heads=self.num_heads,
                ent2type=self.ent2type,
                feat_dims=feat_dims,
                num_layers=self.num_layers,
                num_rels=num_rels,
                return_logits=True,
                decoder=decoder,
            ).to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            pos = float(self.train_df["label"].astype(int).sum())
            neg = float(len(self.train_df) - pos)
            pos_weight = torch.tensor(neg / max(pos, 1.0), device=device, dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            def _load_ablation_checkpoint(path, map_location):
                metadata = _load_training_checkpoint(path, map_location=map_location)
                checkpoint_ablation = metadata.get("ablation_mode")
                if checkpoint_ablation is None:
                    raise ValueError(
                        f"Checkpoint {path} has no ablation_mode metadata. "
                        "Weights-only CPI checkpoints require the same-name .json inference config."
                    )
                checkpoint_ablation = normalize_ablation_mode(checkpoint_ablation)
                if checkpoint_ablation != self.ablation_mode:
                    raise ValueError(
                        f"Checkpoint {path} uses ablation mode {checkpoint_ablation!r}, "
                        f"but the current run uses {self.ablation_mode!r}"
                    )
                return metadata

            if resume:
                ckpt_dir = os.path.join(self.CKPT_DIR, f"fold{fold}")
                pattern = os.path.join(ckpt_dir, f"best_checkpoint_fold{fold}_*.pt")
                checkpoints = glob.glob(pattern)
                if checkpoints:
                    def _metric_of(p):
                        return _metric_of_legacy_checkpoint(p)

                    best_ckpt = max(checkpoints, key=_metric_of)
                    if _metric_of(best_ckpt) == float("-inf"):
                        best_ckpt = max(checkpoints, key=os.path.getmtime)

                    logging.info(f"Resuming training from checkpoint {best_ckpt}")
                    state = _load_ablation_checkpoint(best_ckpt, map_location=device)

                    state_dict = state["state_dict"]
                    rel_w = state_dict.get("decoder.rel_embedding.weight")
                    if rel_w is not None and rel_w.shape[0] != self.num_rels_total:
                        logging.warning(
                            f"num_rels mismatch (ckpt={rel_w.shape[0]}, current={self.num_rels_total}). decoder.rel_embedding will be reinitialized using the current configuration."
                        )

                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if missing or unexpected:
                        logging.warning(f"While loading ckpt: missing_keys={missing}, unexpected_keys={unexpected}")

                    if "optimizer_state_dict" in state:
                        optimizer.load_state_dict(state["optimizer_state_dict"])
                        for s in optimizer.state.values():
                            for k, v in s.items():
                                if isinstance(v, torch.Tensor):
                                    s[k] = v.to(device)

                        start_epoch = int(state.get("epoch", 0))
                        best_val_loss = float(state.get("val_loss", best_val_loss))
                        best_metric = float(state.get("metric", best_metric))
                        best_epoch = int(state.get("epoch", best_epoch))

                        logging.info(
                            f"Resumed at epoch {start_epoch}, val_loss={best_val_loss:.4f}, metric={best_metric:.4f}"
                        )
                    else:
                        logging.warning(
                            "Loaded weights-only checkpoint; optimizer state, epoch, and best validation metric "
                            "are not available, so training will continue with a fresh optimizer from epoch 0."
                        )
                else:
                    logging.info("No usable checkpoint found; training from scratch")

            num_epochs = self.num_epochs
            log_gate_every = int(os.environ.get("LOG_GATE_EVERY", "1"))
            patience = num_epochs
            early_stop_counter = 0

            save_dir = os.path.join(self.CKPT_DIR, f"fold{fold}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for epoch in range(start_epoch, num_epochs):
                model.train()
                optimizer.zero_grad()

                h_train = model(data_train, self.entity_features)
                scores = model.get_score(h_train, train_triplets_cpi)

                logging.info(
                    f"[Train] Epoch {epoch+1}, scores mean: {scores.mean().item():.4f}, std: {scores.std().item():.4f}"
                )

                bce_loss = criterion(scores, train_labels)

                lambda_aux = 0
                aux_batch_per_rel = 0
                rel2types = self._aux_lp_pack["rel2types"]
                type2ids = self._aux_lp_pack["type2ids"]
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

                        idx = torch.randint(pos_all.shape[0], (b,), device=device)
                        pos_batch = pos_all[idx]

                        src_t, dst_t = rel2types[rel_name]
                        cand_dst_ids = type2ids[dst_t]
                        neg_batch = pos_batch.clone()
                        neg_batch[:, 2] = cand_dst_ids[
                            torch.randint(0, cand_dst_ids.numel(), (b,), device=device)
                        ]

                        pos_s = model.get_score(h_train, pos_batch)
                        neg_s = model.get_score(h_train, neg_batch)
                        target = torch.ones_like(pos_s, device=device)

                        per_rel_losses.append(
                            F.margin_ranking_loss(
                                pos_s,
                                neg_s,
                                target,
                                margin=1.0,
                                reduction="mean",
                            )
                        )

                    if len(per_rel_losses):
                        aux_loss = torch.stack(per_rel_losses).mean() * float(lambda_aux)

                loss = bce_loss + aux_loss
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    h_val = model(data_val, self.entity_features)
                    scores_val = model.get_score(h_val, val_triplets_cpi)
                    val_loss = criterion(scores_val, val_labels)

                    if getattr(model.decoder, "return_logits", False):
                        probs_val = torch.sigmoid(scores_val).detach().cpu().numpy()
                    else:
                        probs_val = scores_val.detach().cpu().numpy()

                    labels_np = val_labels.cpu().numpy()
                    roc_auc = roc_auc_score(labels_np, probs_val)
                    aupr = average_precision_score(labels_np, probs_val)

                P = labels_np.sum().astype(float)
                N = float(len(labels_np) - P)
                pos_ratio = P / (P + N + 1e-9)

                fpr, tpr, thr_roc = roc_curve(labels_np, probs_val)
                if len(thr_roc) > 0:
                    accs = (tpr * P + (1.0 - fpr) * N) / (P + N + 1e-9)
                    best_idx_acc = int(np.argmax(accs))
                    thr_acc = float(thr_roc[best_idx_acc])
                else:
                    thr_acc = 0.5

                prec, rec, thr_pr = precision_recall_curve(labels_np, probs_val)
                if len(thr_pr) > 0:
                    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
                    best_idx_f1 = int(np.nanargmax(f1s))
                    thr_f1 = float(thr_pr[best_idx_f1])
                else:
                    thr_f1 = 0.5

                if len(thr_roc) > 0:
                    j_scores = tpr - fpr
                    best_idx_j = int(np.argmax(j_scores))
                    thr_j = float(thr_roc[best_idx_j])
                else:
                    thr_j = 0.5

                if 0.40 <= pos_ratio <= 0.60:
                    chosen_thr = thr_acc
                    chosen_policy = "ACC"
                else:
                    chosen_thr = thr_f1
                    chosen_policy = "F1"

                y_pred = (probs_val > chosen_thr).astype(int)
                precision = precision_score(labels_np, y_pred)
                recall = recall_score(labels_np, y_pred)
                f1 = f1_score(labels_np, y_pred)
                acc = accuracy_score(labels_np, y_pred)

                hits1, hits10, mrr = compute_ranking_metrics(model, h_val, val_triplets_cpi, val_labels)

                logging.info(
                    f"[Val] pos_ratio={pos_ratio:.3f} | thr_acc={thr_acc:.3f}, thr_f1={thr_f1:.3f}, thr_j={thr_j:.3f} | "
                    f"chosen={chosen_policy}({chosen_thr:.3f})"
                )

                best_thr_for_ckpt = chosen_thr
                thr_acc_for_ckpt = thr_acc
                thr_f1_for_ckpt = thr_f1
                thr_j_for_ckpt = thr_j
                chosen_policy_for_ckpt = chosen_policy

                metric = (roc_auc + aupr) / 2
                current_lr = optimizer.param_groups[0]["lr"]

                logging.info(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"lr: {current_lr:.6f}, "
                    f"train_loss={loss.item():.4f}, "
                    f"val_loss={val_loss.item():.4f}, "
                    f"Hits1: {hits1:.4f}, Hits@10: {hits10:.4f}, MRR: {mrr:.4f}, "
                    f"ROC-AUC: {roc_auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUPR: {aupr:.4f}, ACC: {acc:.4f}, "
                    f"Metric: {metric:.4f}"
                )

                if log_gate_every > 0 and ((epoch + 1) % log_gate_every == 0):
                    dec = getattr(model, "decoder", None)
                    if dec is not None and hasattr(dec, "rel_gate"):
                        with torch.no_grad():
                            alpha = torch.sigmoid(dec.rel_gate.weight.squeeze(1)).detach().cpu().numpy()
                        import numpy as _np

                        q25, q50, q75 = _np.quantile(alpha, [0.25, 0.5, 0.75]).tolist()
                        msg = (
                            f"[Gate] epoch {epoch+1}: "
                            f"mean={alpha.mean():.3f}, std={alpha.std():.3f}, "
                            f"min={alpha.min():.3f}, q25/50/75={q25:.3f}/{q50:.3f}/{q75:.3f}, max={alpha.max():.3f}"
                        )
                        logging.info(msg)

                        rid_cpi = self.rel2id.get("CPI", None)
                        if rid_cpi is not None:
                            logging.info(f"[Gate] epoch {epoch+1}: alpha[CPI]={alpha[rid_cpi]:.3f}")

                        order = _np.argsort(alpha)
                        lows = [(self.id2rel[i], float(alpha[i])) for i in order[:3]]
                        highs = [(self.id2rel[i], float(alpha[i])) for i in order[-3:]]
                        logging.info(f"[Gate] epoch {epoch+1}: lowest3={lows}; highest3={highs}")

                if metric > best_metric:
                    best_metric = float(metric)
                    best_val_loss = float(val_loss.item())
                    best_epoch = epoch + 1

                    fname = (
                        f"best_checkpoint_fold{fold}_epoch{best_epoch:03d}_val{best_val_loss:.4f}_metric{best_metric:.4f}"
                        f".pt"
                    )
                    checkpoint_path = os.path.join(save_dir, fname)
                    best_path = checkpoint_path
                    saved_checkpoint_metrics[checkpoint_path] = best_metric
                    torch.save(model.state_dict(), checkpoint_path)
                    _save_checkpoint_config(
                        checkpoint_path,
                        {
                            "threshold": float(best_thr_for_ckpt),
                            "decoder_kwargs": (
                                {"transe_p": int(decoder_kwargs["transe_p"])}
                                if "transe" in decoder_name
                                else {}
                            ),
                            "ablation_mode": self.ablation_mode,
                            "cgi_mode": CGI_MODE,
                            "val_use_train_cpi": int(USE_TRAIN_CPI_IN_VAL_GRAPH),
                            "metadata_node_types": node_types,
                            "metadata_edge_types": edge_types,
                        },
                    )

                    early_stop_counter = 0

                    pattern = os.path.join(save_dir, f"best_checkpoint_fold{fold}_*.pt")
                    all_ckpts = glob.glob(pattern)

                    def _metric_of(p):
                        return saved_checkpoint_metrics.get(p, _metric_of_legacy_checkpoint(p))

                    all_ckpts.sort(key=_metric_of, reverse=True)
                    for p in all_ckpts[3:]:
                        _remove_checkpoint_and_config(p)
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}.")
                    break

            pattern = os.path.join(save_dir, f"best_checkpoint_fold{fold}_*.pt")
            all_ckpts = glob.glob(pattern)
            if not all_ckpts:
                last_fname = (
                    f"best_checkpoint_fold{fold}_epoch{epoch:03d}_val{val_loss:.4f}_metric{metric:.4f}"
                    f".pt"
                )
                best_path = os.path.join(save_dir, last_fname)
                saved_checkpoint_metrics[best_path] = float(metric.item() if hasattr(metric, "item") else metric)
                torch.save(model.state_dict(), best_path)
                _save_checkpoint_config(
                    best_path,
                    {
                        "threshold": float(best_thr_for_ckpt),
                        "decoder_kwargs": (
                            {"transe_p": int(decoder_kwargs["transe_p"])}
                            if "transe" in decoder_name
                            else {}
                        ),
                        "ablation_mode": self.ablation_mode,
                        "cgi_mode": CGI_MODE,
                        "val_use_train_cpi": int(USE_TRAIN_CPI_IN_VAL_GRAPH),
                        "metadata_node_types": node_types,
                        "metadata_edge_types": edge_types,
                    },
                )
                logging.info("No better checkpoint found; saved the final-epoch model as the checkpoint")
            elif best_path is None:
                def _metric_of(p):
                    return saved_checkpoint_metrics.get(p, _metric_of_legacy_checkpoint(p))

                best_path = max(all_ckpts, key=_metric_of)
                if _metric_of(best_path) == float("-inf"):
                    best_path = max(all_ckpts, key=os.path.getmtime)
            assert best_path, "Could not determine the best checkpoint path"

            ckpt = _load_ablation_checkpoint(best_path, map_location=device)
            best_model = HGT(
                hgt_emb_dim=self.hgt_emb_dim,
                metadata=metadata,
                num_heads=self.num_heads,
                ent2type=self.ent2type,
                feat_dims=feat_dims,
                num_layers=self.num_layers,
                num_rels=num_rels,
                return_logits=True,
                decoder=decoder,
            ).to(device)
            best_model.load_state_dict(ckpt["state_dict"])

            thr = resolve_cpi_threshold(ckpt, checkpoint_path=best_path)
            logging.info(f"[CKPT] loaded CPI decision threshold={thr:.3f}")

            best_model.eval()
            with torch.no_grad():
                h_val_best = best_model(data_val, self.entity_features)
            out_csv = os.path.join(save_dir, f"fold_{fold}_val_predictions.csv")
            self.save_predictions(best_model, h_val_best, val_triplets_cpi, val_labels, out_csv, threshold=thr)

            with torch.no_grad():
                scores_best = best_model.get_score(h_val_best, val_triplets_cpi)

            if getattr(best_model.decoder, "return_logits", False):
                probs_best = torch.sigmoid(scores_best).cpu().numpy()
            else:
                probs_best = scores_best.cpu().numpy()

            y_pred_best = (probs_best > thr).astype(int)

            roc_auc_best = roc_auc_score(val_labels.cpu(), probs_best)
            precision_best = precision_score(val_labels.cpu(), y_pred_best)
            recall_best = recall_score(val_labels.cpu(), y_pred_best)
            f1_best = f1_score(val_labels.cpu(), y_pred_best)
            aupr_best = average_precision_score(val_labels.cpu(), probs_best)
            acc_best = accuracy_score(val_labels.cpu(), y_pred_best)
            hits1_best, hits10_best, mrr_best = compute_ranking_metrics(
                best_model,
                h_val_best,
                val_triplets_cpi,
                val_labels,
            )

            return {
                "ablation_mode": self.ablation_mode,
                "Best Epoch": best_epoch,
                "Hits1": hits1_best,
                "Hits@10": hits10_best,
                "MRR": mrr_best,
                "ROC-AUC": roc_auc_best,
                "Precision": precision_best,
                "Recall": recall_best,
                "F1": f1_best,
                "AUPR": aupr_best,
                "ACC": acc_best,
            }
