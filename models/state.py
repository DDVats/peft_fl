from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def adapter_state(model: Any) -> Mapping[str, Any]:
    return model.adapter.adapter_state_dict()


def classifier_state(model: Any) -> Mapping[str, Any]:
    return model.classifier.state_dict()


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
    model.load_communication_state(state, mode)


def communication_parameter_count(state: Mapping[str, Any]) -> int:
    return sum(value.numel() for group in state.values() for value in group.values())


def communication_size_bytes(state: Mapping[str, Any]) -> int:
    return sum(value.numel() * value.element_size() for group in state.values() for value in group.values())


def assert_backbone_excluded(model: Any, state: Mapping[str, Any]) -> None:
    if "backbone" in state:
        raise AssertionError("Frozen backbone group found in FL state")
    backbone_names = {name for name, _ in model.backbone.named_parameters()}
    if backbone_names.intersection(state):
        raise AssertionError("Frozen backbone parameter found in FL state")