# metrics.py

import torch


def compute_ranking_metrics(model, h, triplets, labels):
    triplets = triplets.to(h.device)
    model.eval()
    with torch.no_grad():
        scores_tensor = model.get_score(h, triplets)   # [T] on device
        labels_tensor = labels.to(h.device)            # [T]

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
            continue  # Skip heads without positive samples
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
