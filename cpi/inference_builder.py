# inference_builder.py

import copy
import logging

import pandas as pd
import torch

from checkpoint_io import load_checkpoint_bundle

from .config import CGI_GLOBAL, normalize_ablation_mode
from .graph_builder import add_cpi_edges, build_bg_graph
from .model import HGT, build_decoder


def _infer_decoder_name(state_dict: dict) -> str:
    keys = set(state_dict.keys())
    if "decoder.dec_b.rel_re.weight" in keys:
        return "mlp+complex"
    if "decoder.dec_b.rel_diag.weight" in keys:
        return "mlp+distmult"
    if "decoder.dec_b.rel.weight" in keys:
        return "mlp+transe"
    if "decoder.rel_re.weight" in keys:
        return "complex"
    if "decoder.rel_diag.weight" in keys:
        return "distmult"
    if "decoder.rel.weight" in keys:
        return "transe"
    return "mlp"


def _infer_decoder_kwargs(state_dict: dict, decoder_name: str) -> dict:
    kwargs = {}
    if "mlp" in decoder_name:
        for key in ["decoder.dec_a.mlp.0.weight", "decoder.mlp.0.weight"]:
            if key in state_dict:
                kwargs["mlp_hidden_dim"] = int(state_dict[key].shape[0])
                break
        kwargs.setdefault("mlp_hidden_dim", 512)
    if "transe" in decoder_name:
        kwargs.setdefault("transe_p", 1)
    if "+" in decoder_name:
        kwargs.setdefault("gate_init", 0.0)
    return kwargs


def load_inference_ckpt(checkpoint_path: str, device):
    """
    Load model weights plus optional same-stem JSON inference config.
    The .pt file may be either a raw state_dict or the legacy full checkpoint dict.
    """
    return load_checkpoint_bundle(checkpoint_path, map_location=device)


def resolve_cpi_threshold(ckpt: dict, checkpoint_path: str | None = None) -> float:
    if "model_state_dict" in ckpt:
        keys = ("best_thr", "bester_thr", "threshold", "best_thr_acc", "best_thr_f1")
    else:
        keys = ("threshold",)
    for key in keys:
        if key in ckpt:
            return float(ckpt[key])
    loc = f" for {checkpoint_path}" if checkpoint_path else ""
    raise ValueError(
        "CPI binary prediction requires a threshold in the inference config" + loc +
        ". For weights-only checkpoints, provide a same-stem .json file such as ensemble_1.json."
    )


def resolve_inference_config(ckpt: dict, trainer, ablation_mode: str | None = None, cgi_mode: str | None = None):
    """
    Resolve the inference configuration using the same logic as the original predictor / trainer.

    Returns:
      {
        "cfg_ablation": ...,
        "included_rels": ...,
        "cfg_cgi_mode": ...,
        "use_train_cpi": ...,
        "ckpt_node_types": ...,
        "ckpt_edge_types": ...,
        "decoder_name": ...,
        "decoder_kwargs": ...,
      }
    """
    cfg_ablation = normalize_ablation_mode(ablation_mode or ckpt.get("ablation_mode", "FULL"))
    included_rels = set(trainer.get_included_bg_rels(cfg_ablation))

    cfg_cgi_mode = cgi_mode or ckpt.get("cgi_mode", CGI_GLOBAL)
    use_train_cpi = int(ckpt.get("val_use_train_cpi", 1))

    ckpt_node_types = ckpt.get("metadata_node_types", None)
    ckpt_edge_types = ckpt.get("metadata_edge_types", None)
    if ckpt_edge_types is not None:
        ckpt_edge_types = [tuple(edge_type) for edge_type in ckpt_edge_types]

    state_dict = ckpt.get("state_dict", {})
    decoder_name = ckpt.get("decoder_name") or _infer_decoder_name(state_dict)
    decoder_kwargs = _infer_decoder_kwargs(state_dict, decoder_name)
    decoder_kwargs.update(ckpt.get("decoder_kwargs", {}))

    return {
        "cfg_ablation": cfg_ablation,
        "included_rels": included_rels,
        "cfg_cgi_mode": cfg_cgi_mode,
        "use_train_cpi": use_train_cpi,
        "ckpt_node_types": ckpt_node_types,
        "ckpt_edge_types": ckpt_edge_types,
        "decoder_name": decoder_name,
        "decoder_kwargs": decoder_kwargs,
    }


def get_train_cpi_positive_edges(trainer):
    """
    Restore positive training CPI edges for optional inclusion in the inference graph as historical edges.
    The logic matches the original test_external / predict_unlabeled_*.
    """
    tr_cpi = trainer.map_triplets(trainer.train_df)
    tr_pos = tr_cpi[
        (tr_cpi["relation"] == "CPI") | (tr_cpi["relation_id"] == trainer.rel2id.get("CPI", -1))
    ].copy()

    if "label" in tr_pos.columns:
        tr_pos = tr_pos[tr_pos["label"].astype(int) == 1]

    tr_pos = tr_pos.drop_duplicates(subset=["head_id", "tail_id"])
    return tr_pos


def build_cgi_background_for_inference(
    trainer,
    included_rels: set,
    cfg_cgi_mode: str,
    ext_cgi_mapped=None,
    compounds: list[str] | None = None,
    external_cpi_mapped=None,
):
    """
    Construct the CGI background used for inference.

    Support two scenarios:
    1) labeled external test:
       - With CGI_SPLIT, split CGI by the compounds appearing in external_cpi_mapped
       - With CGI_GLOBAL, include all CGI in the background
    2) unlabeled prediction:
       - With CGI_SPLIT, split CGI by the compounds appearing in the compounds list
       - With CGI_GLOBAL, include all CGI in the background

    Returns:
      cgi_for_bg
    """
    if "CGI" not in included_rels:
        return None

    cgi_train = trainer.map_triplets(trainer.cgi_df) if not trainer.cgi_df.empty else None

    if ext_cgi_mapped is None or ext_cgi_mapped.empty:
        return cgi_train

    if cfg_cgi_mode == CGI_GLOBAL:
        if cgi_train is not None:
            return pd.concat([cgi_train, ext_cgi_mapped], ignore_index=True)
        return ext_cgi_mapped

    # CGI_SPLIT
    if external_cpi_mapped is not None:
        ext_comp_ids = set(
            external_cpi_mapped.loc[
                external_cpi_mapped["head_type"].eq("compound"),
                "head_id",
            ].astype(int).tolist()
        )
    elif compounds is not None:
        ext_comp_ids = set(
            trainer.ent2id[(c, "compound")]
            for c in compounds
            if (c, "compound") in trainer.ent2id
        )
    else:
        ext_comp_ids = set()

    ext_sub = ext_cgi_mapped[
        ext_cgi_mapped["head_type"].eq("compound")
        & ext_cgi_mapped["head_id"].isin(list(ext_comp_ids))
    ].copy()

    extra = ext_cgi_mapped[~ext_cgi_mapped["head_type"].eq("compound")].copy()
    if not extra.empty:
        ext_sub = pd.concat([ext_sub, extra], ignore_index=True)

    if cgi_train is not None:
        return pd.concat([cgi_train, ext_sub], ignore_index=True)
    return ext_sub


def build_inference_graph(
    trainer,
    included_rels: set,
    cgi_for_bg,
    use_train_cpi: int,
):
    """
    Construct the inference graph from background relations and historical training CPI edges.
    The logic matches the original predictor / trainer.
    """
    bg = build_bg_graph(
        included_rels,
        cgi_for_bg,
        trainer.ppi_df,
        trainer.mpi_df,
        trainer.gpi_df,
        trainer.ggi_df,
        trainer.device,
    )

    tr_pos = get_train_cpi_positive_edges(trainer) if use_train_cpi else None

    data_test = copy.deepcopy(bg)
    if use_train_cpi:
        data_test = add_cpi_edges(data_test, tr_pos, trainer.device)

    data_test = data_test.to(trainer.device)
    return data_test, tr_pos


def build_inference_model(
    trainer,
    ckpt: dict,
    ckpt_node_types=None,
    ckpt_edge_types=None,
    data_test=None,
):
    """
    Restore the inference model.
    The HGT + decoder + load_state_dict logic matches the original predictor / trainer.

    Returns:
      model, decoder, feat_dims, node_types, edge_types
    """
    feat_dims = {
        "compound": trainer.cpd_features.shape[1],
        "metabolite": trainer.metabolite_features.shape[1],
        "protein": trainer.protein_features.shape[1],
        "gene": trainer.gene_features.shape[1],
    }

    node_types = ckpt_node_types if ckpt_node_types is not None else list(feat_dims.keys())
    edge_types = ckpt_edge_types if ckpt_edge_types is not None else list(data_test.edge_types)
    cpi_edge_type = ("compound", "CPI", "protein")
    if ckpt_edge_types is None and cpi_edge_type not in edge_types:
        edge_types.append(cpi_edge_type)

    state_dict = ckpt["state_dict"]
    decoder_name = ckpt.get("decoder_name") or _infer_decoder_name(state_dict)
    decoder_kwargs = _infer_decoder_kwargs(state_dict, decoder_name)
    decoder_kwargs.update(ckpt.get("decoder_kwargs", {}))

    decoder = build_decoder(
        decoder_name,
        trainer.hgt_emb_dim,
        trainer.num_rels_total,
        return_logits=True,
        **decoder_kwargs,
    )

    model = HGT(
        hgt_emb_dim=trainer.hgt_emb_dim,
        metadata=(node_types, edge_types),
        num_heads=trainer.num_heads,
        ent2type=trainer.ent2type,
        feat_dims=feat_dims,
        num_layers=trainer.num_layers,
        num_rels=trainer.num_rels_total,
        return_logits=True,
        decoder=decoder,
    ).to(trainer.device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logging.warning(
            f"[build_inference_model] load_state_dict non-strict: missing={missing}, unexpected={unexpected}"
        )

    model.eval()
    return model, decoder, feat_dims, node_types, edge_types


def prepare_inference_runtime(
    trainer,
    checkpoint_path: str,
    ablation_mode: str | None = None,
    cgi_mode: str | None = None,
    ext_cgi_mapped=None,
    compounds: list[str] | None = None,
    external_cpi_mapped=None,
    use_train_cpi: bool | None = None,
):
    """
    Prepare all inference runtime objects in one operation.

    Inputs:
      - trainer
      - checkpoint_path
      - ablation_mode / cgi_mode from override
      - external CGI mapping result ext_cgi_mapped
      - compounds (unlabeled scenario)
      - external_cpi_mapped (labeled external-test scenario)

    Returns a dictionary:
      {
        "ckpt": ckpt,
        "cfg": cfg,
        "cgi_for_bg": cgi_for_bg,
        "data_test": data_test,
        "train_cpi_pos": train_cpi_pos,
        "model": model,
        "decoder": decoder,
        "feat_dims": feat_dims,
        "node_types": node_types,
        "edge_types": edge_types,
        "rid_cpi": rid_cpi,
      }
    """
    ckpt = load_inference_ckpt(checkpoint_path, trainer.device)
    cfg = resolve_inference_config(
        ckpt,
        trainer,
        ablation_mode=ablation_mode,
        cgi_mode=cgi_mode,
    )
    if use_train_cpi is not None:
        cfg["use_train_cpi"] = int(use_train_cpi)

    cgi_for_bg = build_cgi_background_for_inference(
        trainer,
        included_rels=cfg["included_rels"],
        cfg_cgi_mode=cfg["cfg_cgi_mode"],
        ext_cgi_mapped=ext_cgi_mapped,
        compounds=compounds,
        external_cpi_mapped=external_cpi_mapped,
    )

    data_test, train_cpi_pos = build_inference_graph(
        trainer,
        included_rels=cfg["included_rels"],
        cgi_for_bg=cgi_for_bg,
        use_train_cpi=cfg["use_train_cpi"],
    )

    model, decoder, feat_dims, node_types, edge_types = build_inference_model(
        trainer,
        ckpt,
        ckpt_node_types=cfg["ckpt_node_types"],
        ckpt_edge_types=cfg["ckpt_edge_types"],
        data_test=data_test,
    )

    rid_cpi = trainer.rel2id["CPI"]

    return {
        "ckpt": ckpt,
        "cfg": cfg,
        "cgi_for_bg": cgi_for_bg,
        "data_test": data_test,
        "train_cpi_pos": train_cpi_pos,
        "model": model,
        "decoder": decoder,
        "feat_dims": feat_dims,
        "node_types": node_types,
        "edge_types": edge_types,
        "rid_cpi": rid_cpi,
    }
