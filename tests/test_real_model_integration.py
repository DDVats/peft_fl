from __future__ import annotations

import importlib.util
import os
import pytest

from federated.aggregation import fedavg
from federated.client import Client
from models.classifier import classifier_parameter_count
from models.factory import build_model
from models.state import communication_size_bytes, communication_state, load_communication_state


REQUIRED_PACKAGES = ("torch", "transformers", "peft")
RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) for name in REQUIRED_PACKAGES)


@pytest.fixture(scope="module")
def real_runtime():
    if not RUNTIME_AVAILABLE:
        pytest.skip("NOT RUNTIME VERIFIED: torch, transformers, and peft are unavailable")
    import torch
    return torch


def test_real_factory_parameter_contracts(real_runtime):
    expected_classes = {"cifar100": 76900, "pathmnist": 6921}
    for dataset, expected_classifier_count in expected_classes.items():
        for method in ("lora", "hres"):
            model = build_model(dataset=dataset, method=method)
            assert classifier_parameter_count(model.classifier) == expected_classifier_count
            assert model.adapter.adapter_parameter_count() == 589824
            assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


@pytest.mark.parametrize("method", ["lora", "hres"])
def test_real_communication_state_and_round_trip(real_runtime, method):
    model = build_model(dataset="cifar100", method=method)
    state = communication_state(model, "adapter_plus_classifier")
    assert set(state) == {"adapter", "classifier"}
    assert state["adapter"]
    assert state["classifier"]
    assert "backbone" not in state
    assert communication_size_bytes(state) == sum(value.numel() * value.element_size() for group in state.values() for value in group.values())
    if method == "lora":
        assert model.adapter.target_modules == ("q_proj", "v_proj")
        assert state["adapter"]
        assert all("lora_" in key.lower() for key in state["adapter"])
    else:
        assert "layer_0.down.weight" in state["adapter"]
        assert "layer_11.up.weight" in state["adapter"]
        assert all(value == 0 for value in model.adapter.module.layer_0.up.weight.detach().flatten())

    client_model = build_model(dataset="cifar100", method=method)
    load_communication_state(client_model, state, "adapter_plus_classifier")
    client_state = communication_state(client_model, "adapter_plus_classifier")
    aggregated = fedavg([state, client_state], [1, 1])
    load_communication_state(model, aggregated, "adapter_plus_classifier")
    for group in state:
        for key in state[group]:
            assert torch_equal(state[group][key], aggregated[group][key])


@pytest.mark.parametrize("method", ["lora", "hres"])
def test_one_tiny_real_client_update_preserves_backbone(real_runtime, method):
    learning_rate = os.getenv("FL_VERIFY_LEARNING_RATE")
    if learning_rate is None:
        pytest.skip("NOT RUNTIME VERIFIED: set FL_VERIFY_LEARNING_RATE for the verification-only update")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = build_model(dataset="cifar100", method=method)
    pixel_values = torch.randn(1, 3, 224, 224)
    labels = torch.tensor([0])
    dataloader = DataLoader(TensorDataset(pixel_values, labels), batch_size=1)
    before_adapter = {key: value.detach().clone() for key, value in model.adapter.adapter_state_dict().items()}
    before_classifier = {key: value.detach().clone() for key, value in model.classifier.state_dict().items()}
    before_backbone = {key: value.detach().clone() for key, value in model.backbone.state_dict().items()}

    client = Client(0, model, dataloader, 1, local_epochs=1, learning_rate=float(learning_rate))
    client.fit(communication_state(model, "adapter_plus_classifier"))

    assert any(not torch.equal(before, after) for key, before in before_adapter.items() for after in [model.adapter.adapter_state_dict()[key]])
    assert any(not torch.equal(before, after) for key, before in before_classifier.items() for after in [model.classifier.state_dict()[key]])
    assert all(torch.equal(before, model.backbone.state_dict()[key]) for key, before in before_backbone.items())
    assert client.fit


def torch_equal(left, right):
    import torch
    return torch.equal(left, right)