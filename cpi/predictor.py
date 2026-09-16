# predictor.py

import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .entity_expander import (
    expand_candidate_proteins_from_file,
    expand_compounds_and_proteins_from_frames,
    expand_compounds_from_list,
    expand_entities_from_cgi_frame,
    load_external_compound_features,
)
from .graph_builder import log_cgi_usage
from .inference_builder import prepare_inference_runtime, resolve_cpi_threshold
from .utils import _batched_score, _make_triplets_for_head, _read_table, compute_ranking_metrics


def save_predictions_impl(trainer, model, h, triplets, labels, out_csv, threshold=0.5):
    """
    Corresponds to the original FinalTrainer.save_predictions
    """
    model.eval()
    with torch.no_grad():
        scores = model.get_score(h, triplets.to(h.device))
        if getattr(model.decoder, "return_logits", False):
            probs = torch.sigmoid(scores)
        else:
            probs = scores

        probs_np = probs.detach().cpu().numpy()
        y_pred = (probs_np > threshold).astype(int)
        true = labels.cpu().numpy()

        recs = []
        for (h_id, _, t_id), y_t, y_p, sc in zip(
            triplets.cpu().numpy(),
            true,
            y_pred,
            probs_np,
        ):
            if trainer.ent2type.get(h_id) == "compound" and trainer.ent2type.get(t_id) == "protein":
                recs.append(
                    {
                        "compound": trainer.id2ent[h_id],
                        "protein": trainer.id2ent[t_id],
                        "true_label": int(y_t),
                        "pred_label": int(y_p),
                        "pred_score": float(sc),
                    }
                )

        pd.DataFrame(recs).to_csv(out_csv, index=False)
        logging.info(f"Saved {len(recs)} predictions to {out_csv}")


def test_external_impl(
    trainer,
    external_cgi_file: str,
    external_cpi_file: str,
    external_cpd_features_file: str,
    checkpoint_path: str,
    ablation_mode: str | None = None,
    cgi_mode: str | None = None,
):
    """
    Corresponds to the original FinalTrainer.test_external
    """
    device = trainer.device

    external_feats = load_external_compound_features(external_cpd_features_file)

    external_cgi_raw = _read_table(external_cgi_file, dtype=str)
    external_cpi_raw = _read_table(external_cpi_file, dtype=str)

    expand_compounds_and_proteins_from_frames(
        trainer,
        [external_cgi_raw, external_cpi_raw],
        external_compound_feature_table=external_feats,
    )

    before = len(external_cpi_raw)
    miss_head = external_cpi_raw[
        ~external_cpi_raw.apply(lambda r: (r["head"], r["head_type"]) in trainer.ent2id, axis=1)
    ]
    miss_tail = external_cpi_raw[
        ~external_cpi_raw.apply(lambda r: (r["tail"], r["tail_type"]) in trainer.ent2id, axis=1)
    ]
    logging.info(f"[CPI] raw={before}, miss_head={len(miss_head)}, miss_tail={len(miss_tail)}")

    external_cgi_mapped = trainer.map_triplets(external_cgi_raw)
    external_cpi_mapped = trainer.map_triplets(external_cpi_raw)

    if external_cpi_mapped.empty:
        logging.warning(
            "[test_external] external_cpi_mapped is empty: "
            "Check whether the external CPI columns (head,head_type,tail,tail_type,relation,label) match the entities and relations."
        )
        empty_df = pd.DataFrame(columns=["compound", "protein", "true_label", "pred_score", "pred_label"])
        nan_metrics = {
            k: float("nan")
            for k in ["Hits@1", "Hits@10", "MRR", "ROC-AUC", "Precision", "Recall", "F1", "AUPR", "ACC"]
        }
        return empty_df, nan_metrics

    arr = external_cpi_mapped[["head_id", "relation_id", "tail_id"]].values.astype(int)
    test_triplets = torch.from_numpy(arr).long().to(device)

    UNLABELED = "label" not in external_cpi_mapped.columns
    if UNLABELED:
        test_labels = None
    else:
        test_labels = torch.tensor(
            external_cpi_mapped["label"].astype(int).values,
            dtype=torch.float32,
            device=device,
        )

    runtime = prepare_inference_runtime(
        trainer,
        checkpoint_path=checkpoint_path,
        ablation_mode=ablation_mode,
        cgi_mode=cgi_mode,
        ext_cgi_mapped=external_cgi_mapped,
        external_cpi_mapped=external_cpi_mapped,
    )

    ckpt = runtime["ckpt"]
    data_test = runtime["data_test"]
    model = runtime["model"]
    model.eval()
    logging.info(f"[PREDICT] model.training={model.training}")

    try:
        log_cgi_usage(
            "TEST_EXTERNAL/TEST",
            data_test,
            trainer.cgi_df,
            external_cgi_mapped,
            compounds=list(
                sorted(
                    set(
                        external_cgi_raw.loc[
                            external_cgi_raw.get("head_type") == "compound",
                            "head",
                        ].astype(str).tolist()
                    )
                )
            )
            if ("head_type" in external_cgi_raw.columns)
            else [],
            ent2id=trainer.ent2id,
        )
    except Exception as _e:
        logging.warning(f"[TEST_EXTERNAL] _log_cgi_usage failed: {str(_e)}")

    if test_triplets.numel() == 0:
        logging.warning("[test_external] test_triplets is empty; skipping evaluation.")
        empty_df = pd.DataFrame(columns=["compound", "protein", "true_label", "pred_score", "pred_label"])
        nan_metrics = {
            k: float("nan")
            for k in ["Hits@1", "Hits@10", "MRR", "ROC-AUC", "Precision", "Recall", "F1", "AUPR", "ACC"]
        }
        return empty_df, nan_metrics

    with torch.inference_mode():
        h_test = model(data_test, trainer.entity_features)
        scores = model.get_score(h_test, test_triplets)

    if getattr(model.decoder, "return_logits", False):
        probs = torch.sigmoid(scores).detach().cpu().numpy()
    else:
        probs = scores.detach().cpu().numpy()

    thr = resolve_cpi_threshold(ckpt, checkpoint_path=checkpoint_path)

    if UNLABELED:
        preds_np = (probs > thr).astype(int)
        recs = []
        for (h_id, _, t_id), p, y_pred in zip(
            test_triplets.detach().cpu().numpy(),
            probs,
            preds_np,
        ):
            recs.append(
                {
                    "compound": trainer.id2ent[int(h_id)],
                    "protein": trainer.id2ent[int(t_id)],
                    "true_label": -1,
                    "pred_score": float(p),
                    "pred_label": int(y_pred),
                }
            )
        df_preds = pd.DataFrame(recs)
        metrics = {
            "Hits@1": float("nan"),
            "Hits@10": float("nan"),
            "MRR": float("nan"),
            "ROC-AUC": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "AUPR": float("nan"),
            "ACC": float("nan"),
        }
        return df_preds, metrics

    labels_np = test_labels.detach().cpu().numpy().astype(int)
    preds_np = (probs > thr).astype(int)

    recs = []
    for (h_id, _, t_id), y_true, p, y_pred in zip(
        test_triplets.detach().cpu().numpy(),
        labels_np,
        probs,
        preds_np,
    ):
        recs.append(
            {
                "compound": trainer.id2ent[int(h_id)],
                "protein": trainer.id2ent[int(t_id)],
                "true_label": int(y_true),
                "pred_score": float(p),
                "pred_label": int(y_pred),
            }
        )
    df_preds = pd.DataFrame(recs)

    try:
        roc_auc = roc_auc_score(labels_np, probs)
    except Exception:
        roc_auc = float("nan")
    try:
        aupr = average_precision_score(labels_np, probs)
    except Exception:
        aupr = float("nan")
    try:
        precision = precision_score(labels_np, preds_np)
    except Exception:
        precision = float("nan")
    try:
        recall = recall_score(labels_np, preds_np)
    except Exception:
        recall = float("nan")
    try:
        f1 = f1_score(labels_np, preds_np)
    except Exception:
        f1 = float("nan")
    acc = accuracy_score(labels_np, preds_np)

    hits1, hits10, mrr = compute_ranking_metrics(model, h_test, test_triplets, test_labels)
    metrics = {
        "Hits@1": hits1,
        "Hits@10": hits10,
        "MRR": mrr,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUPR": aupr,
        "ACC": acc,
    }
    return df_preds, metrics


def predict_unlabeled_topk_impl(
    trainer,
    compound_file: str,
    external_cpd_features_file: str | None,
    checkpoint_path: str,
    candidate_file: str,
    topk: int = 20,
    ablation_mode: str | None = None,
    cgi_mode: str | None = None,
    external_cgi_file: str | None = None,
    batch_size: int = 131072,
    chunk_size: int = 32768,
):
    """
    Corresponds to the original FinalTrainer.predict_unlabeled_topk
    """
    dev = trainer.device

    df_list = _read_table(compound_file, dtype=str)
    if "compound" in df_list.columns:
        compounds = df_list["compound"].astype(str).unique().tolist()
    elif {"head", "head_type"} <= set(df_list.columns):
        compounds = df_list.loc[df_list["head_type"].eq("compound"), "head"].astype(str).unique().tolist()
    else:
        raise ValueError("compound_file must contain 'compound' column, or ('head','head_type') with head_type='compound'")
    logging.info(f"[INPUT] num_compounds_in_list={len(compounds)}")

    ext_feats = load_external_compound_features(external_cpd_features_file)
    expand_compounds_from_list(
        trainer,
        compounds,
        external_feature_table=ext_feats,
    )

    ext_cgi_mapped = None
    if external_cgi_file and os.path.exists(external_cgi_file):
        raw = _read_table(external_cgi_file, dtype=str)
        if not raw.empty:
            expand_entities_from_cgi_frame(trainer, raw)
            ext_cgi_mapped = trainer.map_triplets(raw)
            ext_cgi_mapped = ext_cgi_mapped.drop_duplicates(subset=["head_id", "relation", "tail_id"])

    runtime = prepare_inference_runtime(
        trainer,
        checkpoint_path=checkpoint_path,
        ablation_mode=ablation_mode,
        cgi_mode=cgi_mode,
        ext_cgi_mapped=ext_cgi_mapped,
        compounds=compounds,
        use_train_cpi=False,
    )

    data_test = runtime["data_test"]
    model = runtime["model"]
    rid_cpi = runtime["rid_cpi"]
    model.eval()
    logging.info(f"[PREDICT] model.training={model.training}")

    with torch.inference_mode():
        h_test = model(data_test, trainer.entity_features)

    ids, raw_count, unique_count = expand_candidate_proteins_from_file(trainer, candidate_file)
    cand_all = torch.tensor(ids, dtype=torch.long, device=dev)
    logging.info(f"[CAND:file] raw_proteins={raw_count}, unique_proteins={unique_count}")

    kg_proteins_total = sum(1 for _i, _t in trainer.ent2type.items() if _t == "protein")
    logging.info(
        f"[CAND] num_candidates={int(cand_all.numel())}, KG_proteins_total={kg_proteins_total}"
    )

    rows = []
    for c in compounds:
        key = (c, "compound")
        if key not in trainer.ent2id:
            continue
        hid = trainer.ent2id[key]

        cand = cand_all

        scores_all = []
        for i in range(0, cand.numel(), chunk_size):
            tails = cand[i : i + chunk_size]
            trip = _make_triplets_for_head(hid, rid_cpi, tails, dev)
            with torch.inference_mode():
                s = _batched_score(model, h_test, trip, batch=batch_size)
                scores_all.append(s)
        scores_all = torch.cat(scores_all, 0)

        k = min(topk, scores_all.numel())
        top_idx = torch.topk(scores_all, k=k, largest=True).indices
        for j in range(k):
            rows.append(
                {
                    "compound": c,
                    "rank": int(j + 1),
                    "protein_id": trainer.id2ent[int(cand[top_idx[j]].item())],
                    "score_logit": float(scores_all[top_idx[j]].item()),
                }
            )

    return pd.DataFrame(rows)


def predict_unlabeled_all_impl(
    trainer,
    compound_file: str,
    external_cpd_features_file: str | None,
    checkpoint_path: str,
    candidate_file: str,
    ablation_mode: str | None = None,
    cgi_mode: str | None = None,
    external_cgi_file: str | None = None,
    batch_size: int = 131072,
    chunk_size: int = 32768,
):
    """
    Corresponds to the original FinalTrainer.predict_unlabeled_all
    """
    dev = trainer.device

    df_list = _read_table(compound_file, dtype=str)
    if "compound" in df_list.columns:
        compounds = df_list["compound"].astype(str).unique().tolist()
    elif {"head", "head_type"} <= set(df_list.columns):
        compounds = df_list.loc[df_list["head_type"].eq("compound"), "head"].astype(str).unique().tolist()
    else:
        raise ValueError("compound_file must contain 'compound' column, or ('head','head_type') with head_type='compound'")
    logging.info(f"[INPUT] num_compounds_in_list={len(compounds)}")

    ext_feats = load_external_compound_features(external_cpd_features_file)
    newly_added = expand_compounds_from_list(
        trainer,
        compounds,
        external_feature_table=ext_feats,
    )
    logging.info(f"[UNLABELED-ALL] compounds from list = {len(compounds)}, newly added = {len(newly_added)}")

    ext_cgi_mapped = None
    if external_cgi_file and os.path.exists(external_cgi_file):
        new_comp_set = set(newly_added)
        raw = _read_table(external_cgi_file, dtype=str)
        if not raw.empty:
            mask_new = (
                ((raw.get("head_type") == "compound") & (raw.get("head").astype(str).isin(new_comp_set)))
                | ((raw.get("tail_type") == "compound") & (raw.get("tail").astype(str).isin(new_comp_set)))
            )
            raw = raw[mask_new].copy()

        if not raw.empty:
            added_from_ext_cgi = expand_entities_from_cgi_frame(trainer, raw)
            ext_cgi_mapped = trainer.map_triplets(raw)
            ext_cgi_mapped = ext_cgi_mapped.drop_duplicates(subset=["head_id", "relation", "tail_id"])
            logging.info(f"[UNLABELED-ALL] ext CGI edges = {len(ext_cgi_mapped) if ext_cgi_mapped is not None else 0}")
            for _t in ["compound", "protein", "metabolite", "gene"]:
                logging.info(f"[UNLABELED-ALL] newly added via ext CGI - {_t}: {len(added_from_ext_cgi.get(_t, []))}")

    runtime = prepare_inference_runtime(
        trainer,
        checkpoint_path=checkpoint_path,
        ablation_mode=ablation_mode,
        cgi_mode=cgi_mode,
        ext_cgi_mapped=ext_cgi_mapped,
        compounds=compounds,
        use_train_cpi=False,
    )

    data_test = runtime["data_test"]
    model = runtime["model"]
    rid_cpi = runtime["rid_cpi"]
    model.eval()
    logging.info(f"[PREDICT] model.training={model.training}")

    try:
        log_cgi_usage("UNLABELED-ALL/TEST", data_test, trainer.cgi_df, ext_cgi_mapped, compounds, trainer.ent2id)
    except Exception as _e:
        logging.warning(f"[UNLABELED-ALL] _log_cgi_usage failed: {str(_e)}")

    with torch.inference_mode():
        h_test = model(data_test, trainer.entity_features)

    ids, raw_count, unique_count = expand_candidate_proteins_from_file(trainer, candidate_file)
    cand_all = torch.tensor(ids, dtype=torch.long, device=dev)
    logging.info(f"[CAND:file] raw_proteins={raw_count}, unique_proteins={unique_count}")

    kg_proteins_total = sum(1 for _i, _t in trainer.ent2type.items() if _t == "protein")
    logging.info(
        f"[CAND] num_candidates={int(cand_all.numel())}, KG_proteins_total={kg_proteins_total}"
    )

    rows = []
    for c in compounds:
        key = (c, "compound")
        if key not in trainer.ent2id:
            continue
        hid = trainer.ent2id[key]

        cand = cand_all

        scores_all = []
        for i in range(0, cand.numel(), chunk_size):
            tails = cand[i : i + chunk_size]
            trip = _make_triplets_for_head(hid, rid_cpi, tails, dev)
            with torch.inference_mode():
                s = _batched_score(model, h_test, trip, batch=batch_size)
                scores_all.append(s)
        scores_all = torch.cat(scores_all, 0)

        order = torch.argsort(scores_all, descending=True)
        invrank = torch.empty_like(order)
        invrank[order] = torch.arange(1, order.numel() + 1, device=order.device)

        if getattr(model.decoder, "return_logits", False):
            probs_all = torch.sigmoid(scores_all)
        else:
            probs_all = scores_all

        for idx in range(cand.numel()):
            pid = int(cand[idx].item())
            rows.append(
                {
                    "compound": c,
                    "protein": trainer.id2ent[pid],
                    "prob": float(probs_all[idx].item()),
                    "logit": float(scores_all[idx].item()),
                    "rank_in_fold": int(invrank[idx].item()),
                }
            )

    return pd.DataFrame(rows)
