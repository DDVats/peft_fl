from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import AdapterInterface


def resolve_attention_projections(model: Any) -> tuple[str, str]:
    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name}
    for query_name, value_name in (("q_proj", "v_proj"), ("query", "value")):
        if {query_name, value_name} <= leaf_names:
            return query_name, value_name
    candidates = sorted(name for name in leaf_names if "query" in name or "value" in name or "proj" in name)
    raise AssertionError(f"Could not resolve ViT query/value projections; candidates={candidates}")


class LoraAdapter(AdapterInterface):
    def __init__(self, model: Any) -> None:
        try:
            peft = importlib.import_module("peft")
        except ImportError as exc:
            raise RuntimeError("LoRA construction requires peft.") from exc
        query_name, value_name = resolve_attention_projections(model)
        config = peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules=[query_name, value_name])
        self.model = peft.get_peft_model(model, config)
        self.peft = peft
        self.target_modules = (query_name, value_name)
        self.adapter_parameter_names = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and ("lora_" in name.lower() or "lora" in name.lower())
        }

    def adapter_state_dict(self) -> Mapping[str, Any]:
        getter = getattr(self.peft, "get_peft_model_state_dict", None)
        return getter(self.model) if getter else self.model.get_peft_model_state_dict()

    def load_adapter_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        setter = getattr(self.peft, "set_peft_model_state_dict", None)
        if setter:
            setter(self.model, state_dict)
        else:
            self.model.load_state_dict(state_dict, strict=False)

    def trainable_parameters(self) -> Iterable[Any]:
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())