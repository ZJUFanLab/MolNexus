# model.py

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from .config import DEFAULT_TRAIN_MODE, DEFAULT_HGT_DROPOUT, TRAIN_MODE_CONFIGS


class MLPDecoder(nn.Module):
    """
    Standalone MLP-only decoder (without any bilinear terms).
    """

    def __init__(self, combined_dim, num_rels, mlp_hidden_dim=512, mlp_dropout=0.2, return_logits=True):
        super().__init__()
        self.return_logits = return_logits
        self.rel_embedding = nn.Embedding(num_rels, combined_dim)
        nn.init.xavier_uniform_(self.rel_embedding.weight, gain=nn.init.calculate_gain("relu"))
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim * 3, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )


    def forward(self, h, triplets):
        heads = h[triplets[:, 0]]
        tails = h[triplets[:, 2]]
        rels = self.rel_embedding(triplets[:, 1])
        x = torch.cat([heads, rels, tails], dim=1)
        logits = self.mlp(x).squeeze(1)
        return logits if self.return_logits else torch.sigmoid(logits)


class DistMultDecoder(nn.Module):
    """
    DistMult: score = sum(h * r * t)
    """

    def __init__(self, combined_dim, num_rels, return_logits=True):
        super().__init__()
        self.return_logits = return_logits
        self.rel_diag = nn.Embedding(num_rels, combined_dim)
        nn.init.xavier_uniform_(self.rel_diag.weight, gain=1.0)

    def forward(self, h, triplets):
        heads = h[triplets[:, 0]]
        tails = h[triplets[:, 2]]
        rels = self.rel_diag(triplets[:, 1])
        logits = (heads * rels * tails).sum(dim=1)
        return logits if self.return_logits else torch.sigmoid(logits)


class ComplExDecoder(nn.Module):
    """
    ComplEx: score = Re(<h, r, conj(t)>)
    This assumes the HGT output dimension d is even; the first d/2 values are treated as the real part and the remaining d/2 as the imaginary part.
    """

    def __init__(self, combined_dim, num_rels, return_logits=True):
        super().__init__()
        assert combined_dim % 2 == 0, "ComplEx requires an even dimension (set hgt_emb_dim to an even number, such as 512)"
        self.return_logits = return_logits
        k = combined_dim // 2
        self.rel_re = nn.Embedding(num_rels, k)
        self.rel_im = nn.Embedding(num_rels, k)
        nn.init.xavier_uniform_(self.rel_re.weight, gain=1.0)
        nn.init.xavier_uniform_(self.rel_im.weight, gain=1.0)

    def _split_ri(self, x):
        return torch.chunk(x, 2, dim=1)

    def forward(self, h, triplets):
        h_h = h[triplets[:, 0]]
        h_t = h[triplets[:, 2]]
        r_re = self.rel_re(triplets[:, 1])
        r_im = self.rel_im(triplets[:, 1])

        hh_re, hh_im = self._split_ri(h_h)
        tt_re, tt_im = self._split_ri(h_t)

        logits = (
            hh_re * r_re * tt_re
            + hh_im * r_re * tt_im
            + hh_re * r_im * tt_im
            - hh_im * r_im * tt_re
        ).sum(dim=1)
        return logits if self.return_logits else torch.sigmoid(logits)


class TransEDecoder(nn.Module):
    """
    TransE: score = -||h + r - t||_p (used as logits; includes learnable scaling gamma)
    """

    def __init__(self, combined_dim, num_rels, p=1, return_logits=True):
        super().__init__()
        self.return_logits = return_logits
        self.p = p
        self.rel = nn.Embedding(num_rels, combined_dim)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.xavier_uniform_(self.rel.weight, gain=1.0)

    def forward(self, h, triplets):
        heads = h[triplets[:, 0]]
        tails = h[triplets[:, 2]]
        rels = self.rel(triplets[:, 1])
        dist = torch.norm(heads + rels - tails, p=self.p, dim=1)
        logits = -self.gamma * dist
        return logits if self.return_logits else torch.sigmoid(logits)


class DualChannelDecoder(nn.Module):
    """
    Dual-channel wrapper: combine two decoders that return logits using a learnable weight alpha.
    """

    def __init__(self, dec_a: nn.Module, dec_b: nn.Module):
        super().__init__()
        self.dec_a = dec_a
        self.dec_b = dec_b
        self._alpha = nn.Parameter(torch.tensor(0.5))
        self.return_logits = True

    def forward(self, h, triplets):
        a = self.dec_a(h, triplets)
        b = self.dec_b(h, triplets)
        alpha = torch.sigmoid(self._alpha)
        logits = alpha * a + (1.0 - alpha) * b
        return logits


class DualChannelDecoderGate(nn.Module):
    """
    Combine two decoders that return logits using a relation-level gate:
      logits = sigma(g_r) * dec_a + (1 - sigma(g_r)) * dec_b
    where g_r is the learnable scalar (Embedding) corresponding to relation r.
    """

    def __init__(self, dec_a: nn.Module, dec_b: nn.Module, num_rels: int, gate_init: float = 0.0):
        super().__init__()
        self.dec_a = dec_a
        self.dec_b = dec_b
        self.rel_gate = nn.Embedding(num_rels, 1)
        nn.init.constant_(self.rel_gate.weight, float(gate_init))
        self.return_logits = True

    def forward(self, h, triplets):
        a = self.dec_a(h, triplets)
        b = self.dec_b(h, triplets)
        alpha = torch.sigmoid(self.rel_gate(triplets[:, 1]).squeeze(1))
        logits = alpha * a + (1.0 - alpha) * b
        return logits


def build_decoder(name: str, combined_dim: int, num_rels: int, return_logits: bool = True, **kwargs) -> nn.Module:
    """
    name:
      'mlp', 'distmult', 'complex', 'transe',
      'mlp+distmult', 'mlp+complex', 'mlp+transe'
    Optional kwargs:
      - mlp_hidden_dim: int (default: 512; used only by the MLP-only decoder)
      - transe_p: int in {1,2} (default: 1; used only by TransE)
      - gate_init: float (default: 0.0; initial relation-gate value for composite decoders)
    """
    n = name.lower().strip()
    mlp_hidden = int(kwargs.get("mlp_hidden_dim", 512))
    mlp_dropout = float(kwargs.get("mlp_dropout", 0.2))
    transe_p = int(kwargs.get("transe_p", 1))
    gate_init = float(kwargs.get("gate_init", 0.0))

    if n == "mlp":
        return MLPDecoder(combined_dim, num_rels, mlp_hidden_dim=mlp_hidden, mlp_dropout=mlp_dropout, return_logits=return_logits)
    if n == "distmult":
        return DistMultDecoder(combined_dim, num_rels, return_logits=return_logits)
    if n == "complex":
        return ComplExDecoder(combined_dim, num_rels, return_logits=return_logits)
    if n == "transe":
        return TransEDecoder(combined_dim, num_rels, p=transe_p, return_logits=return_logits)

    if n == "mlp+distmult":
        mlp = MLPDecoder(combined_dim, num_rels, mlp_hidden_dim=mlp_hidden, mlp_dropout=mlp_dropout, return_logits=True)
        dsm = DistMultDecoder(combined_dim, num_rels, return_logits=True)
        return DualChannelDecoderGate(mlp, dsm, num_rels, gate_init=gate_init)

    if n == "mlp+complex":
        mlp = MLPDecoder(combined_dim, num_rels, mlp_hidden_dim=mlp_hidden, mlp_dropout=mlp_dropout, return_logits=True)
        cpx = ComplExDecoder(combined_dim, num_rels, return_logits=True)
        return DualChannelDecoderGate(mlp, cpx, num_rels, gate_init=gate_init)

    if n == "mlp+transe":
        mlp = MLPDecoder(combined_dim, num_rels, mlp_hidden_dim=mlp_hidden, mlp_dropout=mlp_dropout, return_logits=True)
        tre = TransEDecoder(combined_dim, num_rels, p=transe_p, return_logits=True)
        return DualChannelDecoderGate(mlp, tre, num_rels, gate_init=gate_init)

    raise ValueError(f"Unknown decoder name: {name}")


class HGT(nn.Module):
    def __init__(
        self,
        hgt_emb_dim,
        metadata,
        num_heads,
        ent2type,
        feat_dims,
        num_layers=2,
        dropout=None,
        num_rels=None,
        return_logits=False,
        decoder: nn.Module | None = None,
    ):
        super().__init__()
        self.hgt_emb_dim = hgt_emb_dim
        self.ent2type = ent2type
        train_mode = os.environ.get("TRAIN_MODE", DEFAULT_TRAIN_MODE)
        mode_cfg = TRAIN_MODE_CONFIGS.get(train_mode, {})

        if dropout is None:
            dropout = mode_cfg.get("hgt_dropout", DEFAULT_HGT_DROPOUT)

        self.dropout = float(dropout)
        self.drop_layer = nn.Dropout(self.dropout)

        self.proj = nn.ModuleDict({t: nn.Linear(feat_dims[t], hgt_emb_dim) for t in feat_dims})

        self.node_types, self.edge_types = metadata
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    hgt_emb_dim,
                    hgt_emb_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_rels = num_rels
        self.decoder = decoder if decoder is not None else build_decoder(
            "mlp+complex", hgt_emb_dim, self.num_rels, return_logits=return_logits
        )

        self.type_norm = nn.ModuleDict({t: nn.LayerNorm(hgt_emb_dim) for t in self.proj.keys()})

        self.type_embed = nn.ParameterDict({t: nn.Parameter(torch.zeros(hgt_emb_dim)) for t in self.proj.keys()})
        for t in self.type_embed:
            nn.init.normal_(self.type_embed[t], mean=0.0, std=0.02)

        self.proj_bn = nn.ModuleDict({t: nn.BatchNorm1d(hgt_emb_dim) for t in self.proj.keys()})

        self.ffn_norm = nn.ModuleDict({t: nn.LayerNorm(hgt_emb_dim) for t in self.proj.keys()})
        self.ffn = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.Linear(hgt_emb_dim, 2 * hgt_emb_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(2 * hgt_emb_dim, hgt_emb_dim),
                )
                for t in self.proj.keys()
            }
        )

    def forward(self, data: HeteroData, features_list):
        """
        data.x_dict: {"compound":Tensor[N_c,?], ...}
        data.edge_index_dict: {("compound","CPI","protein"):LongTensor[2,E], ...}
        features_list: original list[Tensor[f_i]] stored in full-graph node order
        """
        device = next(self.parameters()).device

        x_dict = {t: [] for t in self.proj.keys()}
        idx_dict = {t: [] for t in self.proj.keys()}
        for idx, feat in enumerate(features_list):
            t = self.ent2type[idx]
            x_dict[t].append(feat)
            idx_dict[t].append(idx)

        for t in x_dict:
            X = torch.stack(x_dict[t], dim=0)
            X = X.to(device, non_blocking=True)
            X = self.proj[t](X)
            X = self.proj_bn[t](X)
            X = X + self.type_embed[t]
            X = self.drop_layer(X)
            x_dict[t] = X

        if not hasattr(data, "_g2l_cache") or data._g2l_cache is None:
            num_nodes = len(features_list)
            g2l = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            for t, ids in idx_dict.items():
                ids_t = torch.tensor(ids, dtype=torch.long, device=device)
                g2l[ids_t] = torch.arange(ids_t.numel(), dtype=torch.long, device=device)
            data._g2l_cache = g2l

        if not hasattr(data, "_local_eidx") or data._local_eidx is None:
            local_edge_index_dict = {}
            g2l = data._g2l_cache
            for key in data.edge_types:
                eidx = data[key].edge_index
                h_local = g2l[eidx[0]]
                t_local = g2l[eidx[1]]
                local_edge_index_dict[key] = torch.stack([h_local, t_local], 0)
            data._local_eidx = local_edge_index_dict

        local_edge_index_dict = data._local_eidx

        for t in self.node_types:
            assert t in x_dict, f"Missing node type in x_dict before conv: {t}"

        for conv in self.convs:
            if not local_edge_index_dict:
                break
            for t, emb in x_dict.items():
                logging.debug(f"[HGTConv] Before passing: {t} -> {emb.shape}")

            res_dict = {t: x_dict[t] for t in x_dict}
            out = conv(x_dict, local_edge_index_dict)

            new_x = {}
            for t in self.node_types:
                msg = out.get(t, None)
                if msg is None:
                    x = res_dict[t]
                else:
                    x = self.drop_layer(msg) + res_dict[t]
                    x = self.type_norm[t](x)
                    x = F.relu(x)
                    x = x + self.drop_layer(self.ffn[t](self.ffn_norm[t](x)))
                new_x[t] = x

            x_dict = new_x

            for t, emb in x_dict.items():
                logging.debug(f"[HGTConv] After passing: {t} -> {emb.shape}")

        num_nodes = len(features_list)
        h = torch.zeros((num_nodes, self.hgt_emb_dim), device=device)
        for t, emb in x_dict.items():
            ids = torch.tensor(idx_dict[t], device=device)
            h[ids] = emb

        return h

    def get_score(self, h, triplets):
        return self.decoder(h, triplets)
