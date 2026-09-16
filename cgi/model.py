from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from .graph_builder import get_edge_index_dict_safe


class MLPDecoder(nn.Module):
    """
    Standalone MLP-only decoder (without any bilinear terms).
    """
    def __init__(self, combined_dim, num_rels, mlp_hidden_dim=512, return_logits=True):
        super().__init__()
        self.return_logits = return_logits
        self.rel_embedding = nn.Embedding(num_rels, combined_dim)
        nn.init.xavier_uniform_(self.rel_embedding.weight, gain=nn.init.calculate_gain('relu'))
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim * 3, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, h, triplets):
        heads = h[triplets[:, 0]]
        tails = h[triplets[:, 2]]
        rels  = self.rel_embedding(triplets[:, 1])
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
        rels  = self.rel_diag(triplets[:, 1])
        logits = (heads * rels * tails).sum(dim=1)   # Use directly as logits
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
        return torch.chunk(x, 2, dim=1)  # (re, im)

    def forward(self, h, triplets):
        h_h = h[triplets[:, 0]]
        h_t = h[triplets[:, 2]]
        r_re = self.rel_re(triplets[:, 1])
        r_im = self.rel_im(triplets[:, 1])

        hh_re, hh_im = self._split_ri(h_h)
        tt_re, tt_im = self._split_ri(h_t)
        # Re(<h, r, conj(t)>)
        # = sum(h_re*r_re*t_re + h_im*r_re*t_im + h_re*r_im*t_im - h_im*r_im*t_re)
        logits = (hh_re * r_re * tt_re
                 +hh_im * r_re * tt_im
                 +hh_re * r_im * tt_im
                 -hh_im * r_im * tt_re).sum(dim=1)
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
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Learnable scaling to stabilize the logit scale
        nn.init.xavier_uniform_(self.rel.weight, gain=1.0)

    def forward(self, h, triplets):
        heads = h[triplets[:, 0]]
        tails = h[triplets[:, 2]]
        rels  = self.rel(triplets[:, 1])
        dist = torch.norm(heads + rels - tails, p=self.p, dim=1)  # >=0
        logits = - self.gamma * dist
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
        nn.init.constant_(self.rel_gate.weight, float(gate_init))  # init=0 => sigmoid(0)=0.5
        self.return_logits = True  # Always return logits

    def forward(self, h, triplets):
        # Both sub-decoders should return logits
        a = self.dec_a(h, triplets)
        b = self.dec_b(h, triplets)
        alpha = torch.sigmoid(self.rel_gate(triplets[:, 1]).squeeze(1))  # [B]
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
    transe_p   = int(kwargs.get("transe_p", 1))
    gate_init  = float(kwargs.get("gate_init", 0.0))

    if n == "mlp":
        return MLPDecoder(combined_dim, num_rels, mlp_hidden_dim=mlp_hidden, return_logits=return_logits)
    if n == "distmult":
        return DistMultDecoder(combined_dim, num_rels, return_logits=return_logits)
    if n == "complex":
        return ComplExDecoder(combined_dim, num_rels, return_logits=return_logits)
    if n == "transe":
        return TransEDecoder(combined_dim, num_rels, p=transe_p, return_logits=return_logits)

    if n == "mlp+distmult":
        mlp = MLPDecoder(combined_dim, num_rels, mlp_hidden_dim=mlp_hidden, return_logits=True)
        dsm = DistMultDecoder(combined_dim, num_rels, return_logits=True)
        return DualChannelDecoderGate(mlp, dsm, num_rels, gate_init=gate_init)

    if n == "mlp+complex":
        mlp = MLPDecoder(combined_dim, num_rels, mlp_hidden_dim=mlp_hidden, return_logits=True)
        cpx = ComplExDecoder(combined_dim, num_rels, return_logits=True)
        return DualChannelDecoderGate(mlp, cpx, num_rels, gate_init=gate_init)

    if n == "mlp+transe":
        mlp = MLPDecoder(combined_dim, num_rels, mlp_hidden_dim=mlp_hidden, return_logits=True)
        tre = TransEDecoder(combined_dim, num_rels, p=transe_p, return_logits=True)
        return DualChannelDecoderGate(mlp, tre, num_rels, gate_init=gate_init)

    raise ValueError(f"Unknown decoder name: {name}")


class HGT(nn.Module):
    def __init__(self, hgt_emb_dim, metadata, num_heads, ent2type,
                 feat_dims, num_layers=2, dropout=0.2, num_rels=None, return_logits=False,
                 decoder: nn.Module | None = None):

        super().__init__()
        self.hgt_emb_dim = hgt_emb_dim
        self.ent2type = ent2type
        # Manage dropout manually
        self.dropout = dropout
        self.drop_layer = nn.Dropout(dropout)

        # 1) Project features of different types to the same dimension
        self.proj = nn.ModuleDict({
            t: nn.Linear(feat_dims[t], hgt_emb_dim)
            for t in feat_dims
        })
        # 2) Stack HGTConv layers
        #    metadata: the first item is the node-type list; the second is the edge-type list
        # metadata = (node_types: List[str], edge_types: List[Tuple[src,rel,dst]])
        self.node_types, self.edge_types = metadata
        self.convs = nn.ModuleList([
            HGTConv(hgt_emb_dim, hgt_emb_dim, 
                    metadata=metadata,
                    #dropout=dropout,
                    heads=num_heads)
            for _ in range(num_layers)
        ])
        # 3) Reuse the original MLPDecoder as the decoder
        #self.decoder = MLPDecoder(hgt_emb_dim, len(self.edge_types))
        self.num_rels = num_rels
        #self.decoder = MLPDecoder(hgt_emb_dim, self.num_rels, return_logits=return_logits)
        self.decoder = decoder if decoder is not None else build_decoder(
            "mlp+complex", hgt_emb_dim, self.num_rels, return_logits=return_logits
        )

        # 4) Initialize LayerNorm
        # Create a LayerNorm for each node type
        self.type_norm = nn.ModuleDict({
            t: nn.LayerNorm(hgt_emb_dim) for t in self.proj.keys()
        })

        # --- Type embedding + post-projection BN + feed-forward network (FFN) ---
        self.type_embed = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(hgt_emb_dim)) for t in self.proj.keys()
        })
        for t in self.type_embed:
            nn.init.normal_(self.type_embed[t], mean=0.0, std=0.02)

        self.proj_bn = nn.ModuleDict({
            t: nn.BatchNorm1d(hgt_emb_dim) for t in self.proj.keys()
        })

        self.ffn_norm = nn.ModuleDict({
            t: nn.LayerNorm(hgt_emb_dim) for t in self.proj.keys()
        })
        self.ffn = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(hgt_emb_dim, 2*hgt_emb_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(2*hgt_emb_dim, hgt_emb_dim)
            ) for t in self.proj.keys()
        })

    def forward(self, data: HeteroData, features_list):
        """
        data.x_dict: {"compound":Tensor[N_c,?], ...}
        data.edge_index_dict: {("compound","CPI","protein"):LongTensor[2,E], ...}
        features_list: original list[Tensor[f_i]] stored in full-graph node order
        """
        device = next(self.parameters()).device

        # ——— Step A. Build x_dict by type ———
        # First project the raw features to hgt_emb_dim,
        # then group them into tensors of different types based on ent2type.
        x_dict = {t: [] for t in self.proj.keys()}
        idx_dict = {t: [] for t in self.proj.keys()}
        for idx, feat in enumerate(features_list):
            t = self.ent2type[idx]
            #x_dict[t].append(self.proj[t](feat))
            x_dict[t].append(feat)      # Collect raw features first; project them together later
            idx_dict[t].append(idx)
        # stack:
        for t in x_dict:
            #x_dict[t] = torch.stack(x_dict[t], dim=0)
            # [num_nodes_of_type, in_dim] -> proj -> BN -> + type_embed -> Dropout
            X = torch.stack(x_dict[t], dim=0)                 # [Nt, in_dim]
            X = X.to(device, non_blocking=True)
            X = self.proj[t](X)                               # [Nt, d]
            X = self.proj_bn[t](X)                            # BN
            X = X + self.type_embed[t]                        # Type embedding
            X = self.drop_layer(X)                            # Dropout
            x_dict[t] = X

        ## ——— Step A.5. Convert global IDs to type-specific local indices ———
        ## Build the global-to-local mapping table
        #global2local = {
        #    t: {ent_id: loc for loc, ent_id in enumerate(idx_dict[t])}
        #    for t in idx_dict
        #}
        ## Remap head/tail for each relation
        #local_edge_index_dict = {}
        #for (src_t, rel_type, dst_t), edge_index in data.edge_index_dict.items():
        #    head_glob, tail_glob = edge_index  # shape [E]
        #    # Use the mapping table to convert global IDs to local IDs
        #    head_local = torch.tensor(
        #        [global2local[src_t][int(h)] for h in head_glob],
        #        device=device
        #    )
        #    tail_local = torch.tensor(
        #        [global2local[dst_t][int(t)] for t in tail_glob],
        #        device=device
        #    )
        #    local_edge_index_dict[(src_t, rel_type, dst_t)] = torch.stack(
        #        [head_local, tail_local], dim=0
        #    )
        # ——— Step A.5: Vectorize and cache mappings ———
        if not hasattr(data, "_g2l_cache") or data._g2l_cache is None:
            num_nodes = len(features_list)
            g2l = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            for t, ids in idx_dict.items():
                ids_t = torch.tensor(ids, dtype=torch.long, device=device)
                g2l[ids_t] = torch.arange(ids_t.numel(), dtype=torch.long, device=device)
            data._g2l_cache = g2l  # Cache global-to-local indices

        if not hasattr(data, "_local_eidx") or data._local_eidx is None:
            local_edge_index_dict = {}
            g2l = data._g2l_cache
            edge_index_dict = get_edge_index_dict_safe(data)
            for key, eidx in edge_index_dict.items():
                h_local = g2l[eidx[0]]
                t_local = g2l[eidx[1]]
                local_edge_index_dict[key] = torch.stack([h_local, t_local], 0)
            data._local_eidx = local_edge_index_dict

        local_edge_index_dict = data._local_eidx

        # ——— Step B. Message passing through HGT layers ———
        for t in self.node_types:
            assert t in x_dict, f"Missing node type in x_dict before conv: {t}"

        for conv in self.convs:
            # Empty background graphs retain projected features without message passing.
            if not local_edge_index_dict:
                continue

            # Before message passing, log the size of the x_dict for each node type
            for t, emb in x_dict.items():
                logging.debug(f"[HGTConv] Before passing: {t} -> {emb.shape}")

            #x_dict = conv(x_dict, data.edge_index_dict)
            # Use the remapped local edge_index here
            # Save residuals based on the current x_dict, ensuring every type is present
            res_dict = {t: x_dict[t] for t in x_dict}

            # Call conv only once, using x_dict before entering this layer as input
            out = conv(x_dict, local_edge_index_dict)  # out may omit types that received no messages

            # Assemble the next layer input by type, ensuring all type keys are present
            new_x = {}
            for t in self.node_types:  # Using the type list from metadata is more robust
                msg = out.get(t, None)
                if msg is None:
                    x = res_dict[t]                      # No messages: residual only
                else:
                    x = self.drop_layer(msg) + res_dict[t]
                    x = self.type_norm[t](x)
                    x = F.relu(x)                
                    # Add a feed-forward residual FFN
                    x = x + self.drop_layer(self.ffn[t](self.ffn_norm[t](x)))
                new_x[t] = x

            x_dict = new_x

            # After message passing, log the size of the x_dict for each node type
            for t, emb in x_dict.items():
                logging.debug(f"[HGTConv] After passing: {t} -> {emb.shape}")

        # ——— Step C. Restore the unified h (num_nodes x hgt_emb_dim)———
        num_nodes = len(features_list)
        h = torch.zeros((num_nodes, self.hgt_emb_dim), device=device)
        for t, emb in x_dict.items():
            ids = torch.tensor(idx_dict[t], device=device)
            h[ids] = emb

        return h

    def get_score(self, h, triplets):
        return self.decoder(h, triplets)
