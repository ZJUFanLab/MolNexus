from __future__ import annotations

import importlib
import importlib.util
import glob
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch

from checkpoint_io import load_checkpoint_bundle, torch_load_checkpoint
from .config import CLASS2REL as LOCAL_CLASS2REL
from .config import LABEL_ORDER as LOCAL_LABEL_ORDER
from .config import NODE_TYPES, REL2CLASS as LOCAL_REL2CLASS, TRAINING_SCRIPT
from .entity_expander import (
    build_entity_maps, make_feature_list, prepare_external_input,
    read_base_features, read_edge_table,
)
from .inference_builder import build_prediction_graph, build_rel2id, map_prediction_pairs
from .metrics import write_evaluation, write_predictions
from .utils import read_input_table


def set_global_determinism(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def natural_key(path: Path) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def find_checkpoints(model_dir: str) -> List[Path]:
    root = Path(model_dir)

    patterns = [
        "*.pt",
        "fold*/best_checkpoint_fold*.pt",
        "**/best_checkpoint_fold*.pt",
    ]

    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))

    candidates = sorted(set(candidates), key=natural_key)

    if not candidates:
        raise FileNotFoundError(
            f"No .pt checkpoint files found under model_dir: {model_dir}. "
            "Expected files like fold1/best_checkpoint_fold1_*.pt."
        )

    def _fold_id(path: Path) -> Optional[int]:
        text = str(path)
        m = re.search(r"fold(\d+)", text)
        if m:
            return int(m.group(1))
        return None

    def _metric_of(path: Path) -> float:
        try:
            ckpt = torch_load_checkpoint(path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                return float(ckpt.get("metric", float("-inf")))
        except Exception:
            pass
        return float("-inf")

    by_fold: Dict[int, List[Path]] = {}
    no_fold: List[Path] = []

    for path in candidates:
        fid = _fold_id(path)
        if fid is None:
            no_fold.append(path)
        else:
            by_fold.setdefault(fid, []).append(path)

    if by_fold:
        selected = []
        for fid in sorted(by_fold):
            fold_paths = by_fold[fid]
            best_path = max(fold_paths, key=lambda p: (_metric_of(p), p.stat().st_mtime))
            selected.append(best_path)

        if len(selected) != 5:
            logging.warning(
                "Expected 5 fold checkpoints, selected %d: %s",
                len(selected),
                [p.name for p in selected],
            )

        return selected

    paths = sorted(no_fold, key=natural_key)
    if len(paths) != 5:
        logging.warning(
            "Expected 5 fold checkpoints, found %d: %s",
            len(paths),
            [p.name for p in paths],
        )
    return paths


def load_training_module(path: str):
    local_path = Path(__file__).resolve().with_name('trainer.py')
    if not path or Path(path).resolve() == local_path:
        return importlib.import_module('cgi.trainer')
    script_path = Path(path)
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    import logging as py_logging

    original_basic_config = py_logging.basicConfig

    def _basic_config_noop(*args, **kwargs):
        return None

    py_logging.basicConfig = _basic_config_noop
    try:
        spec = importlib.util.spec_from_file_location("hgt_cgi_training_module", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load import spec for {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency while loading {script_path}: {exc}. "
            "Run this script in the same conda environment used by train.sh "
            "(for example py310_cuda)."
        ) from exc
    finally:
        py_logging.basicConfig = original_basic_config

    required = ["HGT", "build_decoder", "HeteroData"]
    for name in required:
        if not hasattr(module, name):
            raise AttributeError(f"Training module does not define required symbol: {name}")
    return module


def get_training_label_constants(training_module) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    label_order = list(getattr(training_module, "CGI_LABEL_ORDER", LOCAL_LABEL_ORDER))
    rel2class = dict(getattr(training_module, "CGI_REL2CLASS", {r: i for i, r in enumerate(label_order)}))
    class2rel = dict(getattr(training_module, "CGI_CLASS2REL", {i: r for r, i in rel2class.items()}))
    if label_order != LOCAL_LABEL_ORDER:
        logging.warning("Training label order differs from local fallback: %s", label_order)
    return label_order, rel2class, class2rel


def extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint state_dict is not a dict: {type(state_dict)}")

    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def infer_decoder_name(state_dict: Dict[str, Any]) -> str:
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


def infer_decoder_kwargs(
    state_dict: Dict[str, Any],
    ckpt: Any,
    decoder_name: str,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("decoder_kwargs"), dict):
        kwargs.update(ckpt["decoder_kwargs"])

    if "mlp" in decoder_name and "mlp_hidden_dim" not in kwargs:
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


def infer_model_config(state_dict: Dict[str, Any], ckpt: Any, args) -> Dict[str, Any]:
    feat_dims: Dict[str, int] = {}
    hgt_emb_dim: Optional[int] = None
    for node_type in NODE_TYPES:
        key = f"proj.{node_type}.weight"
        if key not in state_dict:
            raise KeyError(f"Checkpoint is missing {key}; cannot infer feature dimensions.")
        weight = state_dict[key]
        if hgt_emb_dim is None:
            hgt_emb_dim = int(weight.shape[0])
        elif hgt_emb_dim != int(weight.shape[0]):
            raise ValueError("Inconsistent HGT embedding dimensions in projection layers.")
        feat_dims[node_type] = int(weight.shape[1])

    layer_ids = []
    for key in state_dict:
        match = re.match(r"convs\.(\d+)\.", key)
        if match:
            layer_ids.append(int(match.group(1)))
    num_layers = max(layer_ids) + 1 if layer_ids else 1

    num_rels = None
    relation_weight_keys = [
        "decoder.rel_gate.weight",
        "decoder.rel_embedding.weight",
        "decoder.rel_diag.weight",
        "decoder.rel.weight",
        "decoder.rel_re.weight",
        "decoder.dec_a.rel_embedding.weight",
        "decoder.dec_b.rel_re.weight",
        "decoder.dec_b.rel_diag.weight",
        "decoder.dec_b.rel.weight",
    ]
    for key in relation_weight_keys:
        if key in state_dict:
            num_rels = int(state_dict[key].shape[0])
            break
    if num_rels is None:
        raise KeyError("Could not infer num_rels from decoder relation weights.")

    decoder_name = args.decoder
    if decoder_name is None and isinstance(ckpt, dict):
        decoder_name = ckpt.get("decoder_name")
    if decoder_name is None:
        decoder_name = infer_decoder_name(state_dict)
    decoder_name = str(decoder_name).lower()
    decoder_kwargs = infer_decoder_kwargs(state_dict, ckpt, decoder_name)

    return {
        "hgt_emb_dim": int(hgt_emb_dim),
        "feat_dims": feat_dims,
        "num_layers": int(num_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "num_rels": int(num_rels),
        "decoder_name": decoder_name,
        "decoder_kwargs": decoder_kwargs,
    }


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[Tuple[int, int, pd.DataFrame]]:
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end, df.iloc[start:end]


def score_multiclass_batched(
    model,
    h,
    pair_df: pd.DataFrame,
    rel2id: Dict[str, int],
    label_order: Sequence[str],
    batch_size: int,
    device,
    torch_module,
) -> Any:
    logits_batches = []
    model.eval()
    with torch_module.no_grad():
        for _, _, sub_df in iter_batches(pair_df, batch_size):
            per_rel_logits = []
            for rel in label_order:
                arr = np.column_stack(
                    [
                        sub_df["head_id"].values.astype(np.int64),
                        np.full(len(sub_df), rel2id[rel], dtype=np.int64),
                        sub_df["tail_id"].values.astype(np.int64),
                    ]
                )
                triplets = torch_module.from_numpy(arr).long().to(device)
                per_rel_logits.append(model.get_score(h, triplets))
            logits_batch = torch_module.stack(per_rel_logits, dim=1)
            logits_batches.append(logits_batch.detach().cpu())
    return torch_module.cat(logits_batches, dim=0)


def build_model(
    training_module,
    model_config: Dict[str, Any],
    metadata: Tuple[List[str], List[Tuple[str, str, str]]],
    ent2type: Dict[int, str],
    state_dict: Dict[str, Any],
    device,
    allow_partial_load: bool = False,
):
    decoder = training_module.build_decoder(
        model_config["decoder_name"],
        model_config["hgt_emb_dim"],
        model_config["num_rels"],
        return_logits=True,
        **model_config["decoder_kwargs"],
    )
    model = training_module.HGT(
        hgt_emb_dim=model_config["hgt_emb_dim"],
        metadata=metadata,
        num_heads=model_config["num_heads"],
        ent2type=ent2type,
        feat_dims=model_config["feat_dims"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        num_rels=model_config["num_rels"],
        return_logits=True,
        decoder=decoder,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if (missing or unexpected) and not allow_partial_load:
        raise ValueError(
            "Checkpoint did not load cleanly. "
            f"Missing keys examples: {missing[:20]}; unexpected keys examples: {unexpected[:20]}. "
            "This usually means metadata, decoder, or feature dimensions do not match training."
        )
    if missing or unexpected:
        logging.warning("Partial load: missing=%s unexpected=%s", missing[:20], unexpected[:20])
    return model


def choose_device(torch_module, requested: str):
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


set_global_determinism(42)

def run(args):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.allow_missing_features_random:
        logging.warning('DANGER: random fallback features explicitly enabled; real KPGT features are preferred.')

    if args.num_workers != 0:
        logging.info("--num_workers is accepted for interface compatibility; inference uses direct batching.")

    #set_global_determinism(args.random_seed)

    training_module = load_training_module(args.training_script)

    device = choose_device(torch, args.device)
    logging.info("Using device: %s", device)

    label_order, rel2class, class2rel = get_training_label_constants(training_module)
    ckpt_paths = find_checkpoints(args.model_dir)

    first_ckpt = load_checkpoint_bundle(ckpt_paths[0], map_location="cpu")
    first_state_dict = extract_state_dict(first_ckpt)
    model_config = infer_model_config(first_state_dict, first_ckpt, args)
    logging.info("Model config inferred from checkpoint: %s", model_config)

    rel2id = build_rel2id(label_order, model_config["num_rels"])
    logging.info("CGI relation ids: %s", {rel: rel2id[rel] for rel in label_order})

    raw_df = read_input_table(args.input_csv, args.delimiter)
    if raw_df.empty:
        raise ValueError(f"Input file is empty: {args.input_csv}")

    prepared = prepare_external_input(raw_df, args, model_config["feat_dims"], rel2class)

    logging.info("Reading base feature tables.")
    base_features = read_base_features(args, model_config["feat_dims"])

    logging.info("Reading background graph edge tables.")
    edge_tables = {
        "ppi": read_edge_table(args.ppi_file, "PPI"),
        "mpi": read_edge_table(args.mpi_file, "MPI"),
        "gpi": read_edge_table(args.gpi_file, "GPI"),
        "ggi": read_edge_table(args.ggi_file, "GGI"),
        "cpi": read_edge_table(args.cpi_pos_file, "CPI"),
    }

    ent2id, id2ent, ent2type = build_entity_maps(
        edge_tables=edge_tables,
        base_features=base_features,
        pair_df=prepared["pair_df"],
        compound_override=prepared["compound_override"],
        gene_override=prepared["gene_override"],
    )
    logging.info("Entity count: %d", len(ent2id))

    pair_df = map_prediction_pairs(prepared["pair_df"], ent2id)
    required_external = set(zip(pair_df["__compound_entity"].astype(str), ["compound"] * len(pair_df)))
    required_external.update(zip(pair_df["__gene_entity"].astype(str), ["gene"] * len(pair_df)))

    features_list = make_feature_list(
        ent2id=ent2id,
        id2ent=id2ent,
        ent2type=ent2type,
        base_features=base_features,
        feat_dims=model_config["feat_dims"],
        required_external=required_external,
        compound_override=prepared["compound_override"],
        gene_override=prepared["gene_override"],
        allow_missing_features_random=args.allow_missing_features_random,
        random_seed=args.random_seed,
        torch_module=torch,
    )

    data, edge_types = build_prediction_graph(
        training_module=training_module,
        edge_tables=edge_tables,
        ent2id=ent2id,
        ablation_mode=args.ablation_mode,
        torch_module=torch,
    )
    metadata = (NODE_TYPES, edge_types)
    logging.info("Metadata node_types=%s", NODE_TYPES)
    logging.info("Metadata edge_types=%s", edge_types)

    data = data.to(device)

    fold_probs: List[np.ndarray] = []
    for fold_idx, ckpt_path in enumerate(ckpt_paths):
        logging.info("Loading fold %d checkpoint: %s", fold_idx, ckpt_path)
        ckpt = load_checkpoint_bundle(ckpt_path, map_location="cpu")
        state_dict = extract_state_dict(ckpt)

        fold_config = infer_model_config(state_dict, ckpt, args)
        comparable_keys = ["hgt_emb_dim", "feat_dims", "num_layers", "num_rels", "decoder_name"]
        for key in comparable_keys:
            if fold_config[key] != model_config[key]:
                raise ValueError(
                    f"Fold config mismatch for {ckpt_path}: {key}={fold_config[key]} "
                    f"differs from first checkpoint {model_config[key]}"
                )
        model = build_model(
            training_module=training_module,
            model_config=fold_config,
            metadata=metadata,
            ent2type=ent2type,
            state_dict=state_dict,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            h = model(data, features_list)
            logits_cpu = score_multiclass_batched(
                model=model,
                h=h,
                pair_df=pair_df,
                rel2id=rel2id,
                label_order=label_order,
                batch_size=args.batch_size,
                device=device,
                torch_module=torch,
            )
            probs = torch.softmax(logits_cpu, dim=1).numpy()
            fold_probs.append(probs)

        del model
        del h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
    pred_ids = avg_probs.argmax(axis=1).astype(np.int64)

    output_dir = Path(args.output_dir)
    pred_path = write_predictions(
        output_dir=output_dir,
        meta_df=prepared["meta_df"],
        avg_probs=avg_probs,
        pred_ids=pred_ids,
        class2rel=class2rel,
        fold_probs=fold_probs,
        save_fold_probs=args.save_fold_probs,
    )
    logging.info("Saved predictions to %s", pred_path)

    if prepared["y_true"] is not None:
        report_path, cm_path, summary_path = write_evaluation(
            output_dir=output_dir,
            y_true=prepared["y_true"],
            y_pred=pred_ids,
            y_prob=avg_probs,
            label_order=label_order,
        )
        logging.info("Saved classification report to %s", report_path)
        logging.info("Saved confusion matrix to %s", cm_path)
        logging.info("Saved evaluation summary to %s", summary_path)
    else:
        logging.info(
            "No label column was detected/provided. Skipping classification_report.csv and confusion_matrix.csv."
        )

    return 0
