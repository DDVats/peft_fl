from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import AdapterInterface


class LoraAdapter(AdapterInterface):
    def __init__(self, model: Any) -> None:
        try:
            peft = importlib.import_module("peft")
        except ImportError as exc:
            raise RuntimeError("LoRA construction requires peft.") from exc
        config = peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules=["q_proj", "v_proj"])
        self.model = peft.get_peft_model(model, config)
        self.peft = peft
        self.target_modules = ("q_proj", "v_proj")

    def adapter_state_dict(self) -> Mapping[str, Any]:
        return self.peft.get_peft_model_state_dict(self.model)

    def load_adapter_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.peft.set_peft_model_state_dict(self.model, state_dict)

    def trainable_parameters(self) -> Iterable[Any]:
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())