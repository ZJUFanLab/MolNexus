# utils.py

import glob
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

from checkpoint_io import torch_load_checkpoint


# Fixed relations formerly discovered from the legacy six-relation TSV.
DECODER_BASE_RELATIONS = (
    "CGI-D",
    "CGI-U",
    "GGI",
    "GPI",
    "MPI",
    "PPI",
)


def build_relation_id_map(*relation_groups):
    """Build the checkpoint-compatible, lexicographically sorted relation map."""
    relations = {relation for group in relation_groups for relation in group}
    return {relation: index for index, relation in enumerate(sorted(relations))}


def setup_logging(log_file: str, filemode: str = "w", level: int = logging.DEBUG) -> None:
    """
    Encapsulate the logging configuration from the original script in a function for explicit use by train.py / predict.py.
    The default behavior matches the original training script:
      - File logging level: DEBUG
      - Console logging level: INFO
      - Output to stdout
    """
    root_logger = logging.getLogger()

    # Avoid adding handlers repeatedly
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    root_logger.setLevel(level)

    file_handler = logging.FileHandler(log_file, mode=filemode)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(console_formatter)
    root_logger.addHandler(console)


def set_seed(seed: int = 42) -> None:
    """
    Encapsulate the random-seed setup from the original script in a function.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _read_table(path, **kwargs):
    """
    Select the delimiter automatically:
      - When the caller does not explicitly provide sep/delimiter,
        .tsv/.tsv.gz/.tab use '\\t'; otherwise use the pandas default (comma).
    Pass all other arguments directly to pandas.read_csv.
    """
    p = str(path).lower()
    has_sep = ("sep" in kwargs) or ("delimiter" in kwargs)
    if not has_sep:
        if p.endswith((".tsv", ".tsv.gz", ".tab")):
            kwargs["sep"] = "\t"
    return pd.read_csv(path, **kwargs)


def log_ram(tag: str = "") -> None:
    import psutil

    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    logging.info(f"[MEM]{f' {tag}' if tag else ''} RSS={rss:.2f} GB")


def _batched_score(model, h, triplets, batch: int = 131072):
    out = []
    with torch.no_grad():
        for i in range(0, triplets.size(0), batch):
            s = model.get_score(h, triplets[i:i + batch])
            out.append(s.detach())
    return torch.cat(out, 0)


def _make_triplets_for_head(head_id: int, rid_cpi: int, cand_tails: torch.LongTensor, device):
    H = torch.full((cand_tails.numel(), 1), int(head_id), dtype=torch.long, device=device)
    R = torch.full((cand_tails.numel(), 1), int(rid_cpi), dtype=torch.long, device=device)
    T = cand_tails.view(-1, 1).long()
    return torch.cat([H, R, T], dim=1)


def find_checkpoint(ckpt_dir, fold, best_epoch):
    """
    Find a checkpoint by fold and Best Epoch.
    Training output filenames use epoch{best_epoch:03d} directly (one-based), so no -1 adjustment is needed.
    """
    pattern = os.path.join(
        ckpt_dir,
        f"fold{fold}",
        f"best_checkpoint_fold{fold}_epoch{int(best_epoch):03d}_*.pt",
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Could not find a checkpoint for Fold {fold}; pattern: {pattern}")
    return matches[0]


def find_best_checkpoint_by_metric(ckpt_dir, fold):
    """
    When the summary has no Best Epoch, select the checkpoint with the highest saved metric.
    """
    pattern = os.path.join(ckpt_dir, f"fold{fold}", f"best_checkpoint_fold{fold}_*.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Could not find a checkpoint for Fold {fold}; pattern: {pattern}")

    def _metric_of(p):
        try:
            checkpoint = torch_load_checkpoint(p, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                return float(checkpoint.get("metric", float("-inf")))
        except Exception:
            pass
        return float("-inf")

    matches.sort(key=lambda p: (_metric_of(p), os.path.getmtime(p)), reverse=True)
    return matches[0]


def compute_ranking_metrics(model, h, triplets, labels):
    triplets = triplets.to(h.device)
    model.eval()
    with torch.no_grad():
        scores_tensor = model.get_score(h, triplets)
        labels_tensor = labels.to(h.device)

    trip_np = triplets.cpu().numpy()
    head2idx = {}
    for i, trip in enumerate(trip_np):
        head = int(trip[0])
        head2idx.setdefault(head, []).append(i)

    hits1 = hits10 = mrr = 0.0
    heads_with_pos = 0

    for idxs in head2idx.values():
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=scores_tensor.device)
        s = scores_tensor.index_select(0, idxs_t)
        l = labels_tensor.index_select(0, idxs_t)

        pos = (l == 1).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        heads_with_pos += 1

        order = torch.argsort(-s)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        rank = int(inv[pos].min().item())
        if rank == 0:
            hits1 += 1
        if rank < 10:
            hits10 += 1
        mrr += 1.0 / (rank + 1)

    if heads_with_pos == 0:
        return float("nan"), float("nan"), float("nan")
    return hits1 / heads_with_pos, hits10 / heads_with_pos, mrr / heads_with_pos
