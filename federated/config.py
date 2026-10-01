from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FederatedConfig:
    dataset: str
    method: str
    communication_state: str = "adapter_plus_classifier"
    number_of_clients: int | None = None
    client_fraction: float | None = None
    local_epochs: int | None = None
    federated_rounds: int | None = None
    batch_size: int = 32
    learning_rate: float | None = None
    seed: int = 42
    partition_strategy: str = "iid"

    def __post_init__(self) -> None:
        if self.communication_state not in {"adapter_only", "adapter_plus_classifier"}:
            raise ValueError(f"Unsupported communication state: {self.communication_state}")
        if self.partition_strategy != "iid":
            raise NotImplementedError("Only deterministic IID partitioning is implemented currently.")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "FederatedConfig":
        return cls(**values)