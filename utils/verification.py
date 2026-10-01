from __future__ import annotations

from typing import Any


def verification_report(model: Any) -> dict[str, Any]:
    adapter_keys = list(model.adapter.adapter_state_dict())
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    module_names = [name for name, _ in model.backbone.named_modules()]
    return {
        "vit_module_names": module_names,
        "lora_targeted_modules": getattr(model.adapter, "target_modules", []),
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_count": sum(count for _, count in trainable),
        "adapter_parameter_count": model.adapter.adapter_parameter_count(),
        "adapter_state_dict_keys": adapter_keys,
        "state_dict_keys": list(model.backbone.state_dict()) + list(model.classifier.state_dict()),
    }


def print_verification_report(model: Any) -> None:
    for key, value in verification_report(model).items():
        print(f"{key}: {value}")