from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import random
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from checkpoint_io import load_checkpoint_bundle as load_checkpoint
from torch_geometric.data import HeteroData

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
from .config import TRAINING_SCRIPT
from .entity_expander import (
    build_entity_maps, create_entity_features, map_target_pairs,
    merge_external_feature_file, read_base_tables, upsert_external_features,
    validate_target_features,
)
from .inference_builder import build_background_graph, build_rel2id, map_background_edges
from .metrics import make_prediction_output, save_evaluation
from .utils import (
    detect_label_column, features_to_matrix, infer_compound_metabolite_ids,
    map_label_series, read_input_table, select_feature_columns,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_reproducible_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def import_training_module():
    return importlib.import_module('cmi.trainer')


def strip_module_prefix(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def checkpoint_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return strip_module_prefix(ckpt[key])
        if all(hasattr(v, "shape") for v in ckpt.values()):
            return strip_module_prefix(ckpt)
    raise ValueError("Unsupported checkpoint format; expected model_state_dict/state_dict or a raw state_dict.")


def sorted_model_paths(model_dir: str, model_glob: str, expected_folds: int) -> list[Path]:
    paths = sorted(Path(model_dir).glob(model_glob), key=model_sort_key)
    if not paths:
        raise FileNotFoundError(f"No model files matched {Path(model_dir) / model_glob}")
    if expected_folds and len(paths) != expected_folds:
        raise ValueError(
            f"Expected {expected_folds} model files, found {len(paths)}: "
            f"{[p.name for p in paths]}"
        )
    return paths


def model_sort_key(path: Path):
    name = path.name
    patterns = [
        r"ensemble[_-]?(\d+)",
        r"fold[_-]?(\d+)",
        r"fold(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return (0, int(match.group(1)), name)
    match = re.search(r"(\d+)", path.stem)
    if match:
        return (1, int(match.group(1)), name)
    return (2, name)


def infer_decoder_name(state_dict: dict) -> str:
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


def infer_decoder_kwargs(state_dict: dict, ckpt, decoder_name: str) -> dict:
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
    if isinstance(ckpt, dict) and isinstance(ckpt.get("decoder_kwargs"), dict):
        kwargs.update(ckpt["decoder_kwargs"])
    return kwargs


def infer_model_hparams(state_dict: dict, edge_type_count: int) -> dict:
    compound_weight = state_dict.get("proj.compound.weight")
    metabolite_weight = state_dict.get("proj.metabolite.weight")
    protein_weight = state_dict.get("proj.protein.weight")
    gene_weight = state_dict.get("proj.gene.weight")
    if compound_weight is None or metabolite_weight is None:
        raise ValueError("Checkpoint is missing proj.compound/proj.metabolite weights.")

    hgt_emb_dim = int(compound_weight.shape[0])
    feat_dims = {
        "compound": int(compound_weight.shape[1]),
        "metabolite": int(metabolite_weight.shape[1]),
        "protein": int(protein_weight.shape[1]) if protein_weight is not None else None,
        "gene": int(gene_weight.shape[1]) if gene_weight is not None else None,
    }

    conv_layers = set()
    for key in state_dict:
        match = re.match(r"convs\.(\d+)\.", key)
        if match:
            conv_layers.add(int(match.group(1)))
    num_layers = max(conv_layers) + 1 if conv_layers else 1

    k_rel = state_dict.get("convs.0.k_rel.weight")
    if k_rel is None:
        num_heads = 4
    else:
        rel_head_count = int(k_rel.shape[0])
        if rel_head_count % edge_type_count != 0:
            raise ValueError(
                "Metadata edge type count does not match checkpoint HGTConv "
                f"relation parameters: {edge_type_count} edge types vs {rel_head_count} relation-head weights."
            )
        num_heads = rel_head_count // edge_type_count

    decoder_rel_count = None
    for key in [
        "decoder.rel_gate.weight",
        "decoder.dec_a.rel_embedding.weight",
        "decoder.dec_b.rel_re.weight",
        "decoder.rel_embedding.weight",
        "decoder.rel_diag.weight",
        "decoder.rel.weight",
    ]:
        if key in state_dict:
            decoder_rel_count = int(state_dict[key].shape[0])
            break

    return {
        "hgt_emb_dim": hgt_emb_dim,
        "feat_dims": feat_dims,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "decoder_rel_count": decoder_rel_count,
    }


def choose_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(training_module, ckpt: dict, state_dict: dict, metadata, ent2type, feat_dims, rel_count, hparams, device):
    decoder_name = ckpt.get("decoder_name") if isinstance(ckpt, dict) else None
    if decoder_name is None:
        decoder_name = infer_decoder_name(state_dict)
    decoder_name = str(decoder_name).lower()
    decoder_kwargs = infer_decoder_kwargs(state_dict, ckpt, decoder_name)
    decoder = training_module.build_decoder(
        decoder_name,
        hparams["hgt_emb_dim"],
        rel_count,
        return_logits=True,
        **decoder_kwargs,
    )
    model = training_module.HGT(
        hgt_emb_dim=hparams["hgt_emb_dim"],
        metadata=metadata,
        num_heads=hparams["num_heads"],
        ent2type=ent2type,
        feat_dims=feat_dims,
        num_layers=hparams["num_layers"],
        num_rels=rel_count,
        return_logits=True,
        decoder=decoder,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, decoder_name, decoder_kwargs


def score_cmi_batches(model, h, target_pair_ids, rel2id, label_order, batch_size: int, device):
    logits_batches = []
    for start in range(0, target_pair_ids.shape[0], batch_size):
        stop = min(start + batch_size, target_pair_ids.shape[0])
        batch = target_pair_ids[start:stop]
        heads = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
        tails = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
        rel_logits = []
        for rel in label_order:
            rel_ids = torch.full_like(heads, int(rel2id[rel]))
            triplets = torch.stack([heads, rel_ids, tails], dim=1)
            rel_logits.append(model.get_score(h, triplets))
        logits_batches.append(torch.stack(rel_logits, dim=1).detach().cpu())
    return torch.cat(logits_batches, dim=0)


def run_ensemble_prediction(
    training_module,
    model_paths: list[Path],
    data,
    entity_features,
    target_pair_ids,
    metadata,
    ent2type,
    feat_dims,
    rel2id,
    label_order,
    batch_size: int,
    device,
):
    fold_probs = []
    fold_names = []
    first_state_hparams = None
    for fold_idx, model_path in enumerate(model_paths):
        logging.info("Loading ensemble model %s/%s: %s", fold_idx + 1, len(model_paths), model_path)
        ckpt = load_checkpoint(model_path, map_location=device)
        state_dict = checkpoint_state_dict(ckpt)
        hparams = infer_model_hparams(state_dict, edge_type_count=len(metadata[1]))
        if first_state_hparams is None:
            first_state_hparams = hparams
        elif hparams != first_state_hparams:
            raise ValueError(f"Checkpoint hparams differ from previous folds: {model_path}")

        if hparams["decoder_rel_count"] is not None and hparams["decoder_rel_count"] != len(rel2id):
            raise ValueError(
                f"Relation count mismatch for {model_path.name}: checkpoint decoder has "
                f"{hparams['decoder_rel_count']} relations, reconstructed rel2id has {len(rel2id)}."
            )
        for entity_type, expected_dim in hparams["feat_dims"].items():
            if expected_dim is None:
                continue
            if feat_dims[entity_type] != expected_dim:
                raise ValueError(
                    f"{entity_type} feature dimension mismatch: checkpoint expects "
                    f"{expected_dim}, current feature table has {feat_dims[entity_type]}."
                )

        model, decoder_name, decoder_kwargs = build_model(
            training_module,
            ckpt,
            state_dict,
            metadata,
            ent2type,
            feat_dims,
            len(rel2id),
            hparams,
            device,
        )
        logging.info("Fold decoder: %s %s", decoder_name, decoder_kwargs)
        with torch.no_grad():
            h = model(data, entity_features)
            logits = score_cmi_batches(
                model,
                h,
                target_pair_ids,
                rel2id,
                label_order,
                batch_size,
                device,
            )
            probs = torch.softmax(logits, dim=1).numpy()
        fold_probs.append(probs)
        fold_names.append(model_path.stem)
        del model, h, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stacked = np.stack(fold_probs, axis=0)
    return stacked.mean(axis=0), fold_probs, fold_names


def run(args):
    setup_logging()

    if args.allow_random_missing_external_features:
        logging.warning('DANGER: random fallback features explicitly enabled; real KPGT features are preferred.')

    global np, pd, torch, HeteroData
    global accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
    global classification_report, confusion_matrix, matthews_corrcoef, log_loss
    global roc_auc_score, average_precision_score
    import numpy as np
    import pandas as pd
    import torch
    set_reproducible_seed(42)
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
        matthews_corrcoef,
        log_loss,
        roc_auc_score,
        average_precision_score,
    )
    from torch_geometric.data import HeteroData

    training_module = import_training_module()
    set_reproducible_seed(42)
    label_order = list(training_module.CMI_LABEL_ORDER)
    rel2class = dict(training_module.CMI_REL2CLASS)
    class2rel = dict(training_module.CMI_CLASS2REL)

    if args.num_workers:
        logging.info("--num_workers is accepted for CLI compatibility; HGT full-graph inference does not use DataLoader workers.")

    input_df = read_input_table(args.input_csv, args.delimiter, args.max_rows)
    model_paths = sorted_model_paths(args.model_dir, args.model_glob, args.expected_folds)
    logging.info("Using model files in order: %s", [p.name for p in model_paths])

    first_ckpt = load_checkpoint(model_paths[0], map_location="cpu")
    first_state = checkpoint_state_dict(first_ckpt)

    tables, features = read_base_tables(args)
    compound_dim = features["compound"].shape[1]
    metabolite_dim = features["metabolite"].shape[1]
    merge_external_feature_file(
        features,
        "compound",
        args.external_compound_features_file,
        args.external_compound_feature_id_col,
        compound_dim,
        args.external_feature_delimiter,
    )
    merge_external_feature_file(
        features,
        "metabolite",
        args.external_metabolite_features_file,
        args.external_metabolite_feature_id_col,
        metabolite_dim,
        args.external_feature_delimiter,
    )

    label_col = detect_label_column(input_df, args.label_col, rel2class)
    if label_col:
        logging.info("Detected label column: %s (used only for evaluation)", label_col)
        true_label_ids, true_label_names = map_label_series(input_df[label_col], rel2class, class2rel)
    else:
        logging.info("No label column detected; evaluation files will be skipped.")
        true_label_ids = None
        true_label_names = None

    compound_ids, metabolite_ids, compound_id_col, metabolite_id_col = infer_compound_metabolite_ids(
        input_df,
        args,
        args.input_format,
    )
    compound_cols, metabolite_cols, selected_by_prefix = select_feature_columns(
        input_df,
        args,
        label_col,
        compound_id_col,
        metabolite_id_col,
        compound_dim,
        metabolite_dim,
    )

    inferred_format = args.input_format
    if args.input_format == "auto":
        inferred_format = "kpgt_features" if compound_cols and metabolite_cols else "triplet"
    if inferred_format == "kpgt_features" and not (compound_cols and metabolite_cols):
        raise ValueError(
            "input_format=kpgt_features requires compound and metabolite feature columns. "
            "Use --compound_feature_prefix/--metabolite_feature_prefix or explicit column lists."
        )
    logging.info("Input format resolved as: %s", inferred_format)

    compound_matrix = features_to_matrix(input_df, compound_cols, "compound")
    metabolite_matrix = features_to_matrix(input_df, metabolite_cols, "metabolite")
    upsert_external_features(features, "compound", compound_ids, compound_matrix)
    upsert_external_features(features, "metabolite", metabolite_ids, metabolite_matrix)
    validate_target_features(
        features,
        compound_ids,
        metabolite_ids,
        args.allow_random_missing_external_features,
    )

    ent2id, id2ent, ent2type = build_entity_maps(tables, features, compound_ids, metabolite_ids)
    mapped_tables = {name: map_background_edges(df, ent2id) for name, df in tables.items()}

    device = choose_device(args.device)
    logging.info("Using device: %s", device)
    data, edge_types = build_background_graph(mapped_tables, device)
    data = data.to(device)

    rel2id = build_rel2id(tables["cgi"], label_order)
    missing_labels = [rel for rel in label_order if rel not in rel2id]
    if missing_labels:
        raise ValueError(f"Reconstructed rel2id is missing CMI labels: {missing_labels}")

    hparams = infer_model_hparams(first_state, edge_type_count=len(edge_types))
    feat_dims = {entity_type: features[entity_type].shape[1] for entity_type in ["compound", "metabolite", "protein", "gene"]}
    for entity_type, expected_dim in hparams["feat_dims"].items():
        if expected_dim is not None and feat_dims[entity_type] != expected_dim:
            raise ValueError(
                f"{entity_type} feature dimension mismatch before inference: "
                f"checkpoint expects {expected_dim}, table has {feat_dims[entity_type]}."
            )
    if hparams["decoder_rel_count"] is not None and hparams["decoder_rel_count"] != len(rel2id):
        raise ValueError(
            f"Relation count mismatch: checkpoint decoder has {hparams['decoder_rel_count']} "
            f"relations, reconstructed rel2id has {len(rel2id)}."
        )

    metadata = (["compound", "metabolite", "protein", "gene"], edge_types)
    entity_features = create_entity_features(ent2type, id2ent, features)
    target_pair_ids = map_target_pairs(compound_ids, metabolite_ids, ent2id)

    mean_probs, fold_probs, fold_names = run_ensemble_prediction(
        training_module,
        model_paths,
        data,
        entity_features,
        target_pair_ids,
        metadata,
        ent2type,
        feat_dims,
        rel2id,
        label_order,
        args.batch_size,
        device,
    )

    feature_cols = set((compound_cols or []) + (metabolite_cols or []))
    predictions, pred_ids = make_prediction_output(
        input_df,
        feature_cols,
        compound_ids,
        metabolite_ids,
        label_col,
        true_label_ids,
        true_label_names,
        mean_probs,
        fold_probs,
        fold_names,
        label_order,
        args.save_fold_probs,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    logging.info("Saved predictions: %s", output_dir / "predictions.csv")

    if label_col:
        save_evaluation(output_dir, true_label_ids.to_numpy(), pred_ids, label_order, mean_probs)
        logging.info("Saved evaluation files: classification_report.csv, confusion_matrix.csv")
    else:
        logging.info("No true labels provided; skipped classification_report.csv and confusion_matrix.csv.")
