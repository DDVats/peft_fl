from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from evaluation.metrics import evaluate_model
from models.state import communication_state, load_communication_state

from .aggregation import FedAvg


@dataclass
class Server:
    model: Any
    dataset: str
    method: str
    communication_mode: str = "adapter_plus_classifier"
    validation_loader: Any = None
    evaluator: Callable[[Any, Any], dict[str, float]] | None = None
    aggregator: FedAvg = field(default_factory=FedAvg)
    round_results: list[dict[str, Any]] = field(default_factory=list)

    def global_state(self) -> dict[str, Any]:
        return dict(communication_state(self.model, self.communication_mode))

    def distribute(self) -> dict[str, Any]:
        return self.global_state()

    def run_round(self, clients: list[Any], round_number: int) -> dict[str, Any]:
        if not clients:
            raise ValueError("Cannot run a round with no clients")
        started = time.perf_counter()
        results = [client.fit(self.distribute()) for client in clients]
        aggregated = self.aggregator.aggregate([result.state for result in results], [result.num_samples for result in results])
        load_communication_state(self.model, aggregated, self.communication_mode)
        validation = self.evaluator(self.model, self.validation_loader) if self.evaluator else evaluate_model(self.model, self.validation_loader)
        record = {
            "round": round_number, "dataset": self.dataset, "method": self.method,
            "communication_state": self.communication_mode, "participating_clients": [result.client_id for result in results],
            "total_client_samples": sum(result.num_samples for result in results),
            "global_val_loss": validation["loss"], "global_val_accuracy": validation["accuracy"], "global_val_macro_f1": validation["macro_f1"],
            "total_communication_bytes": sum(result.communication_size_bytes for result in results),
            "round_time": time.perf_counter() - started,
            "clients": [{"round": round_number, "client_id": result.client_id, "num_samples": result.num_samples, "local_epochs": result.local_epochs, "local_loss": result.metrics.get("loss"), "local_accuracy": result.metrics.get("accuracy"), "local_macro_f1": result.metrics.get("macro_f1"), "local_training_time": result.training_time, "uploaded_bytes": result.communication_size_bytes} for result in results],
        }
        self.round_results.append(record)
        return record