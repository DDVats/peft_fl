from __future__ import annotations

import pickle
import sys
from types import SimpleNamespace

import pytest

from federated.checkpointing import (
    checkpoint_path,
    latest_checkpoint,
    load_round_checkpoint,
    new_status,
    next_round_from_checkpoint,
    save_round_checkpoint,
    write_status,
)


class Value:
    shape = ()
    dtype = "float32"

    def __init__(self, value=0):
        self.value = value


class Adapter:
    def __init__(self):
        self.value = Value()

    def adapter_state_dict(self):
        return {"weight": self.value}

    def load_adapter_state_dict(self, state):
        self.value = state["weight"]


class Classifier:
    def __init__(self):
        self.value = Value()

    def state_dict(self):
        return {"weight": self.value}

    def load_state_dict(self, state):
        self.value = state["weight"]


class Model:
    def __init__(self):
        self.adapter = Adapter()
        self.classifier = Classifier()

    def load_communication_state(self, state, mode):
        self.adapter.load_adapter_state_dict(state["adapter"])
        if mode == "adapter_plus_classifier":
            self.classifier.load_state_dict(state["classifier"])


def fake_torch(monkeypatch):
    def save(payload, path):
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    def load(path, **kwargs):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(save=save, load=load))


def metadata(round_number=1):
    return {
        "round": round_number, "seed": 42, "method": "lora", "dataset": "cifar100",
        "partition": "noniid", "alpha": 0.5, "num_clients": 4, "local_epochs": 1,
        "learning_rate": 0.001, "metrics": {"global_val_loss": 1.0},
        "client_sample_counts": [10, 11, 12, 13], "round_time": 1.0, "local_training_time": 0.5,
    }


def test_checkpoint_path_and_payload_exclude_backbone_optimizer(tmp_path, monkeypatch):
    fake_torch(monkeypatch)
    model = Model()
    path = checkpoint_path(tmp_path, "cifar100", "lora", "noniid", 42, 1)
    save_round_checkpoint(path, model, metadata(), "adapter_plus_classifier")
    payload = pickle.loads(path.read_bytes())
    assert path.name == "round_001.pt"
    assert set(payload) == {"state", "metadata"}
    assert set(payload["state"]) == {"adapter", "classifier"}
    assert "backbone" not in payload["state"]
    assert "optimizer" not in payload["state"]
    assert "scheduler" not in payload["state"]


def test_latest_checkpoint_loads_and_corrupt_checkpoint_is_rejected(tmp_path, monkeypatch):
    fake_torch(monkeypatch)
    model = Model()
    path_one = checkpoint_path(tmp_path, "cifar100", "lora", "iid", 42, 1)
    path_two = checkpoint_path(tmp_path, "cifar100", "lora", "iid", 42, 2)
    expected = {key: metadata()[key] for key in ("dataset", "method", "partition", "alpha", "seed", "num_clients", "local_epochs", "learning_rate")}
    save_round_checkpoint(path_one, model, {**metadata(), "partition": "iid", "alpha": None}, "adapter_plus_classifier")
    save_round_checkpoint(path_two, model, {**metadata(2), "partition": "iid", "alpha": None}, "adapter_plus_classifier")
    latest = latest_checkpoint(tmp_path, "cifar100", "lora", "iid", 42, model, "adapter_plus_classifier", expected | {"partition": "iid", "alpha": None})
    assert latest is not None
    assert latest[0] == path_two
    assert next_round_from_checkpoint(latest[1], 10) == 3
    assert next_round_from_checkpoint(latest[1], 2) is None
    path_two.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Could not load checkpoint"):
        load_round_checkpoint(path_two, model, "adapter_plus_classifier", expected | {"partition": "iid", "alpha": None})


def test_status_updates_are_persisted_atomically(tmp_path):
    path = tmp_path / "fl_runs" / "experiment" / "status.json"
    status = new_status("experiment", {"dataset": "cifar100", "method": "lora", "partition": "iid", "alpha": None, "seed": 42, "num_clients": 4}, 10)
    status["rounds_completed"] = 3
    write_status(path, status)
    assert path.exists()
    assert '"rounds_completed": 3' in path.read_text()