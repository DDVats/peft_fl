from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FederatedConfig:
    dataset: str
    method: str
    communication_state: str = "adapter_plus_classifier"
    number_of_clients: int = 4
    client_fraction: float = 1.0
    local_epochs: int = 1
    federated_rounds: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    partition_strategy: str = "iid"
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.communication_state not in {"adapter_only", "adapter_plus_classifier"}:
            raise ValueError(f"Unsupported communication state: {self.communication_state}")
        if self.partition_strategy not in {"iid", "noniid"}:
            raise ValueError(f"Unsupported partition strategy: {self.partition_strategy}")
        if self.partition_strategy == "noniid" and (self.alpha is None or self.alpha <= 0):
            raise ValueError("alpha must be explicitly provided and greater than zero for noniid partitioning")
        if self.number_of_clients <= 0 or not 0 < self.client_fraction <= 1 or self.local_epochs <= 0 or self.federated_rounds <= 0:
            raise ValueError("Invalid federated protocol values")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "FederatedConfig":
        return cls(**values)