from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


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


def make_prediction_output(
    input_df,
    feature_cols: set[str],
    compound_ids,
    metabolite_ids,
    label_col: str | None,
    true_label_ids,
    true_label_names,
    mean_probs,
    fold_probs,
    fold_names,
    label_order,
    save_fold_probs: bool,
):
    output = input_df.drop(columns=list(feature_cols), errors="ignore").copy()
    if "compound_id" not in output.columns:
        output.insert(0, "compound_id", compound_ids.astype(str).values)
    if "metabolite_id" not in output.columns:
        insert_at = 1 if "compound_id" in output.columns else 0
        output.insert(insert_at, "metabolite_id", metabolite_ids.astype(str).values)

    if label_col is not None:
        output["true_label_id"] = true_label_ids.astype(int).values
        output["true_label_name"] = true_label_names.astype(str).values

    for class_id, label_name in enumerate(label_order):
        output[f"prob_class_{class_id}"] = mean_probs[:, class_id]
        output[f"prob_{label_name}"] = mean_probs[:, class_id]

    pred_ids = mean_probs.argmax(axis=1).astype(int)
    output["pred_label_id"] = pred_ids
    output["pred_label_name"] = [label_order[i] for i in pred_ids]

    if save_fold_probs:
        for fold_idx, probs in enumerate(fold_probs):
            for class_id in range(len(label_order)):
                output[f"fold{fold_idx}_prob_class_{class_id}"] = probs[:, class_id]
    return output, pred_ids


def save_evaluation(output_dir: Path, true_ids, pred_ids, label_order, mean_probs=None):
    labels = list(range(len(label_order)))

    report = classification_report(
        true_ids,
        pred_ids,
        target_names=label_order,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(
        true_ids,
        pred_ids,
        labels=labels,
    )
    pd.DataFrame(cm, index=label_order, columns=label_order).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    summary = {
        "ACC": accuracy_score(true_ids, pred_ids),
        "BalancedACC": balanced_accuracy_score(true_ids, pred_ids),
        "MacroPrecision": precision_score(true_ids, pred_ids, average="macro", zero_division=0),
        "MacroRecall": recall_score(true_ids, pred_ids, average="macro", zero_division=0),
        "MacroF1": f1_score(true_ids, pred_ids, average="macro", zero_division=0),
        "WeightedF1": f1_score(true_ids, pred_ids, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(true_ids, pred_ids),
    }

    if mean_probs is not None:
        true_onehot = np.eye(len(label_order), dtype=float)[true_ids]

        summary["LogLoss"] = log_loss(true_ids, mean_probs, labels=labels)

        top2 = np.argsort(mean_probs, axis=1)[:, -2:]
        summary["Top2Accuracy"] = float(np.mean([
            int(true_ids[i]) in top2[i] for i in range(len(true_ids))
        ]))

        summary["BrierScore"] = float(
            np.mean(np.sum((mean_probs - true_onehot) ** 2, axis=1))
        )

        try:
            summary["MacroROC_AUC_OVR"] = roc_auc_score(
                true_ids,
                mean_probs,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
            summary["WeightedROC_AUC_OVR"] = roc_auc_score(
                true_ids,
                mean_probs,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            summary["MacroROC_AUC_OVR"] = np.nan
            summary["WeightedROC_AUC_OVR"] = np.nan

        try:
            summary["MacroPR_AUC"] = average_precision_score(
                true_onehot,
                mean_probs,
                average="macro",
            )
            summary["WeightedPR_AUC"] = average_precision_score(
                true_onehot,
                mean_probs,
                average="weighted",
            )
        except ValueError:
            summary["MacroPR_AUC"] = np.nan
            summary["WeightedPR_AUC"] = np.nan

    pd.DataFrame([summary]).to_csv(output_dir / "evaluation_summary.csv", index=False)
