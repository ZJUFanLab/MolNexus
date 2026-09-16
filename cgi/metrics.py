from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch


def compute_ranking_metrics(model, h, triplets, labels):
    """
    triplets: Tensor[T,3]  (head,rel,tail)
    labels:   Tensor[T]    (0/1)
    Return Hits@1, Hits@10, and MRR (grouped by head)
    """
    triplets = triplets.to(h.device)
    model.eval()
    with torch.no_grad():
        scores_tensor = model.get_score(h, triplets)       # on GPU
        labels_tensor = labels.to(h.device)
    
    # Move only the triplet indices and final aggregate metrics to the CPU
    trip_np = triplets.cpu().numpy()

    # Group by head
    head2idx = {}
    for i, trip in enumerate(trip_np):
        head = trip[0]
        head2idx.setdefault(head, []).append(i)

    hits1 = hits10 = mrr = 0.0
    for idxs in head2idx.values():
        # Score and rank the candidate relations for this head separately
        # First select the corresponding positions on the GPU
        s = scores_tensor[idxs]        # Tensor on GPU
        l = labels_tensor[idxs]        # Tensor on GPU
        # Sort directly with torch
        order = torch.argsort(-s)
        # Find the GPU positions of all positive samples; flatten ensures a 1D tensor
        pos = (l == 1).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        # Each element in pos corresponds to a position where label == 1
        # Look up their ranks in order
        #ranks = [(order == p).nonzero(as_tuple=False)[0,0].item() for p in pos]
        #rank = min(ranks)
        inv = torch.empty_like(order)                         # Shape: n
        inv[order] = torch.arange(order.numel(), device=order.device)
        # inv[i] is now the rank of candidate i (0 indicates first place)
        rank = int(inv[pos].min().item())                    # Use the highest-ranked of multiple positive samples
        if rank == 0:  hits1 += 1
        if rank < 10: hits10 += 1
        mrr += 1.0 / (rank + 1)
    n = len(head2idx)
    return hits1/n, hits10/n, mrr/n


def write_predictions(
    output_dir: Path,
    meta_df: pd.DataFrame,
    avg_probs: np.ndarray,
    pred_ids: np.ndarray,
    class2rel: Dict[int, str],
    fold_probs: List[np.ndarray],
    save_fold_probs: bool,
) -> Path:
    out_df = meta_df.reset_index(drop=True).copy()
    for class_id in range(avg_probs.shape[1]):
        rel_name = class2rel[int(class_id)]
        out_df[f"prob_{rel_name}"] = avg_probs[:, class_id]
    out_df["pred_label_id"] = pred_ids.astype(int)
    out_df["pred_label_name"] = [class2rel[int(i)] for i in pred_ids]

    if save_fold_probs:
        for fold_idx, probs in enumerate(fold_probs):
            for class_id in range(probs.shape[1]):
                rel_name = class2rel[int(class_id)]
                out_df[f"fold{fold_idx}_prob_{rel_name}"] = probs[:, class_id]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def write_evaluation(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_order: Sequence[str],
) -> Tuple[Path, Path, Path]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        log_loss,
        top_k_accuracy_score,
        brier_score_loss,
        roc_auc_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.preprocessing import label_binarize

    labels = list(range(len(label_order)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(label_order),
        output_dict=True,
        zero_division=0,
    )
    report_path = output_dir / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df = pd.DataFrame(cm, index=list(label_order), columns=list(label_order))
    cm_df.index.name = "true_label"
    cm_df.columns.name = "pred_label"
    cm_df.to_csv(cm_path)

    eps = 1e-15
    y_prob_safe = np.clip(y_prob, eps, 1.0 - eps)
    y_prob_safe = y_prob_safe / y_prob_safe.sum(axis=1, keepdims=True)

    y_true_bin = label_binarize(y_true, classes=labels)

    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "BalancedACC": balanced_accuracy_score(y_true, y_pred),
        "MacroPrecision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "MacroRecall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "MacroF1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "WeightedF1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "LogLoss": log_loss(y_true, y_prob_safe, labels=labels),
        "Top2Accuracy": top_k_accuracy_score(y_true, y_prob_safe, k=2, labels=labels),
        "BrierScore": np.mean([
            brier_score_loss(y_true_bin[:, i], y_prob_safe[:, i])
            for i in range(len(labels))
        ]),
    }

    try:
        metrics["MacroROC_AUC_OVR"] = roc_auc_score(
            y_true,
            y_prob_safe,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        metrics["MacroROC_AUC_OVR"] = np.nan

    try:
        metrics["WeightedROC_AUC_OVR"] = roc_auc_score(
            y_true,
            y_prob_safe,
            labels=labels,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        metrics["WeightedROC_AUC_OVR"] = np.nan

    try:
        metrics["MacroPR_AUC"] = average_precision_score(
            y_true_bin,
            y_prob_safe,
            average="macro",
        )
    except ValueError:
        metrics["MacroPR_AUC"] = np.nan

    try:
        metrics["WeightedPR_AUC"] = average_precision_score(
            y_true_bin,
            y_prob_safe,
            average="weighted",
        )
    except ValueError:
        metrics["WeightedPR_AUC"] = np.nan

    summary_path = output_dir / "evaluation_summary.csv"
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)

    return report_path, cm_path, summary_path
