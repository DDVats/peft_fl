from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import AdapterInterface


class HResBlock:
    def __init__(self, torch: Any, bottleneck: int, dropout: float) -> None:
        self.module = torch.nn.Module()
        self.module.down = torch.nn.Linear(768, bottleneck, bias=False)
        self.module.activation = torch.nn.GELU()
        self.module.dropout = torch.nn.Dropout(dropout)
        self.module.up = torch.nn.Linear(bottleneck, 768, bias=False)
        torch.nn.init.zeros_(self.module.up.weight)

    def __call__(self, hidden: Any) -> Any:
        return self.module.up(self.module.dropout(self.module.activation(self.module.down(hidden))))


class HResAdapter(AdapterInterface):
    def __init__(self, backbone: Any, bottleneck: int = 32, dropout: float = 0.1, scale: float = 1.0) -> None:
        torch = importlib.import_module("torch")
        self.module = torch.nn.Module()
        self.blocks = []
        for index in range(12):
            block = HResBlock(torch, bottleneck, dropout)
            setattr(self.module, f"layer_{index}", block.module)
            self.blocks.append(block)
        self.scale = scale
        self._attach(backbone)

    def _attach(self, backbone: Any) -> None:
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise AssertionError("Centralized H-Res contract requires backbone.layers")
        if len(layers) != 12:
            raise AssertionError(f"Expected 12 backbone layers, found {len(layers)}")
        for index, layer in enumerate(layers):
            original_forward = layer.forward
            adapter = self.blocks[index]
            def forward_with_adapter(*args: Any, _original=original_forward, _adapter=adapter, **kwargs: Any) -> Any:
                output = _original(*args, **kwargs)
                hidden = output[0] if isinstance(output, tuple) else output
                updated = hidden + self.scale * _adapter(hidden)
                return (updated, *output[1:]) if isinstance(output, tuple) else updated
            layer.forward = forward_with_adapter

    def adapter_state_dict(self) -> Mapping[str, Any]:
        return self.module.state_dict()

    def load_adapter_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state_dict)

    def trainable_parameters(self) -> Iterable[Any]:
        return self.module.parameters()

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())