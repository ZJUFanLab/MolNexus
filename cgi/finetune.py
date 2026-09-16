from __future__ import annotations

import copy
import logging
import os
import pandas as pd
import torch
import torch.optim as optim

from checkpoint_io import torch_load_checkpoint as _torch_load_checkpoint

from .utils import DECODER_BASE_RELATIONS, build_relation_id_map


def build_cpi_pretrain_rel2id(cgi_df=None):
    """
    Reproduce the rel2id construction used in the CPI pretraining script.

    The first dimension of the decoder relation embedding/gate in the CPI pretraining checkpoint is the relation dimension,
    so transfer to CGI must reorder it by the relation-name order used during pretraining rather than loading it directly by tensor row number.

    By default, this reads the CGI background file used for CPI pretraining. If pretraining used a different CGI file,
    set the following in sbatch:
        export PRETRAIN_CGI_FILE=/path/to/pretrain_cgi.csv
    """
    if cgi_df is None:
        pretrain_cgi_file = os.environ.get(
            "PRETRAIN_CGI_FILE",
            "demo/BioNexKG_demo/cgi.csv"
        )
        if pretrain_cgi_file and os.path.exists(pretrain_cgi_file):
            cgi_df = pd.read_csv(pretrain_cgi_file, usecols=["relation"], dtype=str)
            logging.info(f"[TRANSFER] Build pretrain rel2id with PRETRAIN_CGI_FILE={pretrain_cgi_file}")
        else:
            logging.warning(
                f"[TRANSFER] PRETRAIN_CGI_FILE not found: {pretrain_cgi_file}. "
                "Only KG background relations + CPI will be used for pretrain rel2id."
            )

    cgi_rels = set(cgi_df["relation"].unique()) if cgi_df is not None and not cgi_df.empty else set()
    return build_relation_id_map(DECODER_BASE_RELATIONS, cgi_rels, ("CPI",))


def _is_decoder_relation_weight(key):
    """
    Dimension 0 of these parameters is the relation dimension and must be reordered by relation name.
    """
    relation_weight_names = (
        "decoder.rel_embedding.weight",
        "decoder.rel_diag.weight",
        "decoder.rel.weight",
        "decoder.rel_re.weight",
        "decoder.rel_im.weight",
        "decoder.rel_gate.weight",
        "decoder.dec_a.rel_embedding.weight",
        "decoder.dec_a.rel_diag.weight",
        "decoder.dec_a.rel.weight",
        "decoder.dec_a.rel_re.weight",
        "decoder.dec_a.rel_im.weight",
        "decoder.dec_b.rel_embedding.weight",
        "decoder.dec_b.rel_diag.weight",
        "decoder.dec_b.rel.weight",
        "decoder.dec_b.rel_re.weight",
        "decoder.dec_b.rel_im.weight",
    )
    return key.endswith(relation_weight_names)


def _remap_relation_weight(pre_w, cur_w, pre_rel2id, cur_rel2id):
    """
    Copy the relation embedding/gate from the CPI checkpoint to the current CMI model by relation name.
    Keep parameters for new or unmatched CGI-N/CGI-D/CGI-U relations randomly initialized.
    """
    new_w = cur_w.clone()
    copied = []

    for rel, pre_idx in pre_rel2id.items():
        cur_idx = cur_rel2id.get(rel, None)
        if cur_idx is None:
            continue
        if pre_idx >= pre_w.size(0) or cur_idx >= new_w.size(0):
            continue
        if pre_w[pre_idx].shape != new_w[cur_idx].shape:
            continue

        new_w[cur_idx].copy_(pre_w[pre_idx])
        copied.append(rel)

    return new_w, copied


def load_cpi_pretrained_for_cgi(model, ckpt_path, pre_rel2id, cur_rel2id, device):
    """
    Transfer from a CPI checkpoint to a CGI model:
    1) Load ordinary parameters with matching names and shapes directly;
    2) Reorder relation-related decoder parameters by relation name;
    3) Keep newly added CMI relation parameters randomly initialized;
    4) Skip parameters with mismatched shapes or no counterpart in the current model.
    """
    if ckpt_path is None or ckpt_path.strip() == "":
        logging.info("[TRANSFER] PRETRAIN_CKPT is empty. Train CGI from scratch.")
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"PRETRAIN_CKPT does not exist: {ckpt_path}")

    ckpt = _torch_load_checkpoint(ckpt_path, map_location=device)
    pre_sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cur_sd = model.state_dict()
    new_sd = copy.deepcopy(cur_sd)

    loaded = []
    remapped = []
    skipped = []

    for key, pre_v in pre_sd.items():
        if key not in cur_sd:
            skipped.append((key, "not_in_current_model"))
            continue

        cur_v = cur_sd[key]

        if _is_decoder_relation_weight(key):
            if pre_v.dim() >= 1 and cur_v.dim() >= 1:
                try:
                    new_w, copied_rels = _remap_relation_weight(
                        pre_v.detach().to(device),
                        cur_v.detach().to(device),
                        pre_rel2id,
                        cur_rel2id
                    )
                    new_sd[key] = new_w
                    remapped.append((key, copied_rels))
                except Exception as e:
                    skipped.append((key, f"relation_remap_failed: {repr(e)}"))
            else:
                skipped.append((key, "relation_weight_dim_error"))
            continue

        if pre_v.shape == cur_v.shape:
            new_sd[key] = pre_v.detach().to(device)
            loaded.append(key)
        else:
            skipped.append((key, f"shape_mismatch pre={tuple(pre_v.shape)} cur={tuple(cur_v.shape)}"))

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    logging.info(f"[TRANSFER] Loaded same-shape parameters: {len(loaded)}")
    logging.info(f"[TRANSFER] Remapped relation parameters: {len(remapped)}")
    for key, rels in remapped:
        logging.info(f"[TRANSFER] {key}: copied relations={rels}")

    logging.info(f"[TRANSFER] Skipped parameters: {len(skipped)}")
    for key, reason in skipped[:50]:
        logging.info(f"[TRANSFER][SKIP] {key}: {reason}")

    if missing or unexpected:
        logging.info(f"[TRANSFER] load_state_dict missing_keys={missing}, unexpected_keys={unexpected}")

    logging.info(f"[TRANSFER] CPI pretrained checkpoint loaded from: {ckpt_path}")


def set_encoder_trainable(model, trainable: bool):
    """
    Control whether the HGT encoder participates in training.
    The encoder includes:
    - proj
    - proj_bn
    - type_embed
    - HGTConv convs
    - type_norm
    - ffn_norm
    - ffn

    The decoder always remains trainable.
    """
    encoder_modules = [
        model.proj,
        model.proj_bn,
        model.convs,
        model.type_norm,
        model.ffn_norm,
        model.ffn,
    ]

    for module in encoder_modules:
        for p in module.parameters():
            p.requires_grad = trainable

    for p in model.type_embed.parameters():
        p.requires_grad = trainable

    for p in model.decoder.parameters():
        p.requires_grad = True


def set_frozen_encoder_eval_mode(model):
    """
    When the encoder is frozen, BatchNorm should not continue updating running_mean/running_var.
    Because the training loop calls model.train() every epoch, this must be called again each epoch.
    """
    model.proj.eval()
    model.proj_bn.eval()
    model.convs.eval()
    model.type_norm.eval()
    model.ffn_norm.eval()
    model.ffn.eval()


def build_finetune_optimizer(model, lr):
    """
    Pass only parameters with requires_grad=True to the optimizer.
    """
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
