from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any


class AdapterInterface(ABC):
    """Common serialization and accounting contract for PEFT adapters."""

    @abstractmethod
    def adapter_state_dict(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def load_adapter_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def trainable_parameters(self) -> Iterable[Any]: ...

    @abstractmethod
    def adapter_parameter_count(self) -> int: ...

    def communication_size_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.adapter_state_dict().values())