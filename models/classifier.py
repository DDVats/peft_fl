from __future__ import annotations

import importlib
from typing import Any


def build_classifier(num_classes: int) -> Any:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("Classifier construction requires torch.") from exc
    return torch.nn.Linear(768, num_classes)


def classifier_parameter_count(classifier: Any) -> int:
    return sum(parameter.numel() for parameter in classifier.parameters())