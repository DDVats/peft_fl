from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from models.state import communication_size_bytes, communication_state, load_communication_state
from training.local import train_local


@dataclass
class ClientResult:
    client_id: int
    state: dict[str, Any]
    num_samples: int
    metrics: dict[str, float]
    training_time: float
    parameter_count: int
    communication_size_bytes: int
    local_epochs: int | None


@dataclass
class Client:
    client_id: int
    model: Any
    dataloader: Any
    num_samples: int
    communication_mode: str = "adapter_plus_classifier"
    local_epochs: int | None = None
    learning_rate: float | None = None
    training_function: Callable[[Any], dict[str, float]] | None = None
    last_received_state: dict[str, Any] | None = field(default=None, init=False)

    def fit(self, global_state: dict[str, Any]) -> ClientResult:
        if self.local_epochs is None and self.training_function is None:
            raise ValueError("local_epochs is protocol-pending and must be configured before training")
        if self.training_function is None and self.learning_rate is None:
            raise ValueError("learning_rate is protocol-pending and must be configured before training")
        load_communication_state(self.model, global_state, self.communication_mode)
        self.last_received_state = global_state
        started = time.perf_counter()
        metrics = self.training_function(self.model) if self.training_function else train_local(self.model, self.dataloader, self.local_epochs, self.learning_rate)
        elapsed = time.perf_counter() - started
        state = dict(communication_state(self.model, self.communication_mode))
        parameters = sum(parameter.numel() for parameter in self.model.trainable_parameters() if parameter.requires_grad)
        return ClientResult(self.client_id, state, self.num_samples, metrics, elapsed, parameters, communication_size_bytes(state), self.local_epochs)