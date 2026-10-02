from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from models.state import communication_state, load_communication_state


CHECKPOINT_METADATA_KEYS = {
    "round", "seed", "method", "dataset", "partition", "alpha", "num_clients",
    "local_epochs", "learning_rate", "metrics", "client_sample_counts", "round_time", "local_training_time",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_path(root: str | Path, dataset: str, method: str, partition: str, seed: int, round_number: int) -> Path:
    return Path(root) / dataset / method / partition / f"seed_{seed}" / f"round_{round_number:03d}.pt"


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Checkpointing requires torch.") from exc
    return torch


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_round_checkpoint(path: str | Path, model: Any, metadata: Mapping[str, Any], mode: str) -> Path:
    metadata = dict(metadata)
    missing = CHECKPOINT_METADATA_KEYS - set(metadata)
    if missing:
        raise ValueError(f"Checkpoint metadata missing keys: {sorted(missing)}")
    state = communication_state(model, mode)
    if set(state) != ({"adapter"} if mode == "adapter_only" else {"adapter", "classifier"}):
        raise ValueError("Checkpoint contains an invalid communication state")
    path = Path(path)
    _atomic_torch_save({"state": state, "metadata": metadata}, path)
    return path


def load_round_checkpoint(path: str | Path, model: Any, mode: str, expected_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    torch = _torch()
    path = Path(path)
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"state", "metadata"}:
        raise ValueError(f"Checkpoint {path} is missing state or metadata")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or CHECKPOINT_METADATA_KEYS - set(metadata):
        raise ValueError(f"Checkpoint {path} has incomplete metadata")
    if expected_metadata:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Checkpoint {path} metadata mismatch for {key}")
    state = payload["state"]
    load_communication_state(model, state, mode)
    return {"state": state, "metadata": metadata}


def latest_checkpoint(root: str | Path, dataset: str, method: str, partition: str, seed: int, model: Any, mode: str, expected_metadata: Mapping[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    directory = Path(root) / dataset / method / partition / f"seed_{seed}"
    checkpoints = sorted(directory.glob("round_*.pt"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    return latest, load_round_checkpoint(latest, model, mode, expected_metadata)


def next_round_from_checkpoint(payload: Mapping[str, Any], rounds_requested: int) -> int | None:
    try:
        completed_round = int(payload["metadata"]["round"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint does not contain a valid completed round") from exc
    if completed_round < 0:
        raise ValueError("Checkpoint round cannot be negative")
    next_round = completed_round + 1
    return None if next_round > rounds_requested else next_round


def write_status(path: str | Path, status: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(status), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def new_status(experiment_name: str, metadata: Mapping[str, Any], rounds_requested: int) -> dict[str, Any]:
    now = _utc_now()
    return {"experiment_name": experiment_name, **dict(metadata), "rounds_requested": rounds_requested, "rounds_completed": 0, "started_at": now, "updated_at": now, "completed_at": None, "total_time_seconds": 0.0, "status": "running"}