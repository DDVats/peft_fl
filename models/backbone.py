from __future__ import annotations

import importlib
from typing import Any


BACKBONE_CHECKPOINT = "google/vit-base-patch16-224-in21k"


def require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("Model construction requires torch.") from exc


def build_frozen_backbone(checkpoint: str = BACKBONE_CHECKPOINT) -> Any:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError("Model construction requires transformers.") from exc
    backbone = transformers.ViTModel.from_pretrained(checkpoint)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


def assert_backbone_frozen(backbone: Any) -> None:
    unfrozen = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    if unfrozen:
        raise AssertionError(f"Frozen backbone has trainable parameters: {unfrozen}")


def assert_vit_contract(backbone: Any) -> None:
    config = backbone.config
    actual = (config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.image_size)
    expected = (768, 12, 12, 224)
    if actual != expected:
        raise AssertionError(f"Unexpected ViT architecture: actual={actual}, expected={expected}")