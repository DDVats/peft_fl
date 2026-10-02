from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def adapter_state(model: Any) -> Mapping[str, Any]:
    return {key: model.adapter.adapter_state_dict()[key] for key in sorted(model.adapter.adapter_state_dict())}


def classifier_state(model: Any) -> Mapping[str, Any]:
    state = model.classifier.state_dict()
    return {key: state[key] for key in sorted(state)}


def adapter_plus_classifier_state(model: Any) -> dict[str, Any]:
    return {"adapter": dict(adapter_state(model)), "classifier": dict(classifier_state(model))}


def full_model_state(model: Any) -> Mapping[str, Any]:
    return {
        "backbone": model.backbone.state_dict(),
        "adapter": dict(adapter_state(model)),
        "classifier": dict(classifier_state(model)),
    }

def communication_state(model: Any, mode: str = "adapter_only") -> Mapping[str, Any]:
    if mode == "adapter_only":
        state = {"adapter": dict(adapter_state(model))}
    elif mode == "adapter_plus_classifier":
        state = adapter_plus_classifier_state(model)
    else:
        raise ValueError(f"Unsupported communication mode: {mode}")
    return copy.deepcopy(state)


def load_communication_state(model: Any, state: Mapping[str, Any], mode: str) -> None:
    expected = {"adapter"} if mode == "adapter_only" else {"adapter", "classifier"}
    if set(state) != expected:
        raise ValueError(f"Communication state keys {set(state)} do not match {expected}")
    current = communication_state(model, mode)
    for group in expected:
        if set(state[group]) != set(current[group]):
            raise ValueError(f"Communication keys for {group} do not match the model")
        for key in current[group]:
            expected_value, received_value = current[group][key], state[group][key]
            if tuple(expected_value.shape) != tuple(received_value.shape):
                raise ValueError(f"Shape mismatch for {group}.{key}")
            if expected_value.dtype != received_value.dtype:
                raise TypeError(f"Dtype mismatch for {group}.{key}")
    model.load_communication_state(state, mode)


def communication_parameter_count(state: Mapping[str, Any]) -> int:
    return sum(value.numel() for group in state.values() for value in group.values())


def communication_size_bytes(state: Mapping[str, Any]) -> int:
    return sum(value.numel() * value.element_size() for group in state.values() for value in group.values())


def state_to_ndarrays(state: Mapping[str, Any]) -> list[Any]:
    arrays = []
    for group in sorted(state):
        for key in sorted(state[group]):
            arrays.append(state[group][key].detach().cpu().numpy())
    return arrays


def ndarrays_to_state(arrays: list[Any], template: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("State array conversion requires torch.") from exc
    expected = [(group, key, template[group][key]) for group in sorted(template) for key in sorted(template[group])]
    if len(arrays) != len(expected):
        raise ValueError(f"Expected {len(expected)} state arrays, received {len(arrays)}")
    state: dict[str, dict[str, Any]] = {group: {} for group in sorted(template)}
    for array, (group, key, reference) in zip(arrays, expected):
        value = torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(f"Shape mismatch for {group}.{key}")
        state[group][key] = value
    return state


def assert_backbone_excluded(model: Any, state: Mapping[str, Any]) -> None:
    if "backbone" in state:
        raise AssertionError("Frozen backbone group found in FL state")
    backbone_names = {name for name, _ in model.backbone.named_parameters()}
    communicated_names = {key for group in state.values() for key in group}
    if backbone_names.intersection(communicated_names):
        raise AssertionError("Frozen backbone parameter found in FL state")