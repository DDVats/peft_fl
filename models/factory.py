from __future__ import annotations

from typing import Any

from .backbone import assert_backbone_frozen, assert_vit_contract, build_frozen_backbone
from .classifier import build_classifier
from .hres import HResAdapter
from .lora import LoraAdapter


DATASET_CLASSES = {"cifar100": 100, "pathmnist": 9}


class FoundationModel:
    def __init__(self, backbone: Any, classifier: Any, adapter: Any, method: str) -> None:
        self.backbone = backbone
        self.classifier = classifier
        self.adapter = adapter
        self.method = method

    def named_parameters(self):
        yield from ((f"classifier.{name}", value) for name, value in self.classifier.named_parameters())
        if self.method == "hres":
            yield from ((f"adapter.{name}", value) for name, value in self.adapter.module.named_parameters())
        else:
            yield from ((name, value) for name, value in self.adapter.model.named_parameters() if value.requires_grad)

    def trainable_parameters(self):
        yield from self.classifier.parameters()
        yield from self.adapter.trainable_parameters()

    def load_communication_state(self, state, mode: str = "adapter_plus_classifier") -> None:
        if mode not in {"adapter_only", "adapter_plus_classifier"}:
            raise ValueError(f"Unsupported communication mode: {mode}")
        self.adapter.load_adapter_state_dict(state["adapter"])
        if mode == "adapter_plus_classifier":
            self.classifier.load_state_dict(state["classifier"])

    def train(self):
        self.classifier.train()
        self.backbone.train()
        (self.adapter.module if self.method == "hres" else self.adapter.model).train()
        return self

    def eval(self):
        self.classifier.eval()
        self.backbone.eval()
        (self.adapter.module if self.method == "hres" else self.adapter.model).eval()
        return self

    def forward(self, pixel_values: Any) -> Any:
        backbone = self.backbone if self.method == "hres" else self.adapter.model
        outputs = backbone(pixel_values=pixel_values)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

    def remove_hooks(self) -> None:
        if self.method == "hres":
            self.adapter.remove_hooks()


def build_model(dataset: str, method: str, checkpoint: str = "google/vit-base-patch16-224-in21k") -> FoundationModel:
    dataset, method = dataset.lower(), method.lower()
    if dataset not in DATASET_CLASSES or method not in {"lora", "hres"}:
        raise ValueError(f"Unsupported dataset/method: {dataset}/{method}")
    backbone = build_frozen_backbone(checkpoint)
    assert_vit_contract(backbone)
    classifier = build_classifier(DATASET_CLASSES[dataset])
    adapter = LoraAdapter(backbone) if method == "lora" else HResAdapter(backbone)
    assert_backbone_frozen(backbone)
    return FoundationModel(backbone, classifier, adapter, method)