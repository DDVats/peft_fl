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


def find_transformer_layers(backbone: Any) -> tuple[Any, ...]:
    candidates = []
    for name, module in backbone.named_modules():
        if name.endswith("encoder.layer") and len(module) == 12:
            candidates.append((name, tuple(module)))
    if len(candidates) != 1:
        raise AssertionError(f"Expected one 12-layer ViT encoder, found {[name for name, _ in candidates]}")
    return candidates[0][1]


class HResAdapter(AdapterInterface):
    def __init__(self, backbone: Any, bottleneck: int = 32, dropout: float = 0.1, scale: float = 1.0) -> None:
        torch = importlib.import_module("torch")
        self.module = torch.nn.ModuleDict()
        self.blocks: list[HResBlock] = []
        for index in range(12):
            block = HResBlock(torch, bottleneck, dropout)
            self.module[f"layer_{index}"] = block.module
            self.blocks.append(block)
        self.scale = scale
        self._hook_handles: list[Any] = []
        self._attach(backbone)

    def _attach(self, backbone: Any) -> None:
        if self._hook_handles:
            return
        for index, layer in enumerate(find_transformer_layers(backbone)):
            adapter = self.blocks[index]

            def forward_hook(_module: Any, inputs: tuple[Any, ...], output: Any, _adapter=adapter) -> Any:
                if not inputs:
                    raise RuntimeError("H-Res hook received no transformer-layer input")
                correction = self.scale * _adapter(inputs[0])
                if isinstance(output, tuple):
                    return (output[0] + correction, *output[1:])
                return output + correction

            self._hook_handles.append(layer.register_forward_hook(forward_hook))

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def adapter_state_dict(self) -> Mapping[str, Any]:
        return self.module.state_dict()

    def load_adapter_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state_dict, strict=True)

    def trainable_parameters(self) -> Iterable[Any]:
        return self.module.parameters()

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())