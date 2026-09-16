from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def torch_load_checkpoint(path, map_location):
    """Load new weights-only and legacy full checkpoints across PyTorch versions."""
    try:
        # weights_only does not discard plain dict/list/scalar metadata. If a
        # legacy checkpoint contains an unsupported object, retry with the
        # legacy unpickler so its inference metadata remains available.
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def is_wrapped_checkpoint(checkpoint: Any) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    return any(
        isinstance(checkpoint.get(key), dict)
        for key in ("model_state_dict", "state_dict")
    )


def checkpoint_state_dict(checkpoint: Any) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def checkpoint_config_path(checkpoint_path) -> Path:
    return Path(checkpoint_path).with_suffix(".json")


def load_inference_config(checkpoint_path) -> dict:
    config_path = checkpoint_config_path(checkpoint_path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint inference config must be a JSON object: {config_path}")
    return config


def save_inference_config(checkpoint_path, payload: dict) -> None:
    with checkpoint_config_path(checkpoint_path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_checkpoint_bundle(path, map_location) -> dict:
    """Return state_dict plus metadata from exactly one checkpoint format.

    Wrapped legacy checkpoints are self-contained and never consult a JSON
    sidecar. Raw state_dict checkpoints obtain non-weight configuration from
    the optional same-stem JSON file.
    """
    checkpoint = torch_load_checkpoint(path, map_location=map_location)
    state_dict = checkpoint_state_dict(checkpoint)

    if is_wrapped_checkpoint(checkpoint):
        bundle = dict(checkpoint)
    else:
        bundle = load_inference_config(path)

    bundle["state_dict"] = state_dict
    return bundle


def remove_inference_config(checkpoint_path) -> None:
    try:
        checkpoint_config_path(checkpoint_path).unlink()
    except OSError:
        pass


def remove_checkpoint_bundle(checkpoint_path) -> None:
    for path in (Path(checkpoint_path), checkpoint_config_path(checkpoint_path)):
        try:
            path.unlink()
        except OSError:
            pass
