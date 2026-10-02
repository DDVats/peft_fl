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
    total_rounds: int | None = None
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
        global_state = self.distribute()
        results = [client.fit(global_state) for client in clients]
        local_training_time = sum(result.training_time for result in results)
        for result in results:
            print(f"[CLIENT {result.client_id}] train_time={result.training_time:.6f} samples={result.num_samples}", flush=True)
        aggregation_started = time.perf_counter()
        aggregated = self.aggregator.aggregate([result.state for result in results], [result.num_samples for result in results])
        load_communication_state(self.model, aggregated, self.communication_mode)
        aggregation_time = time.perf_counter() - aggregation_started
        evaluation_started = time.perf_counter()
        validation = self.evaluator(self.model, self.validation_loader) if self.evaluator else evaluate_model(self.model, self.validation_loader)
        evaluation_time = time.perf_counter() - evaluation_started
        round_time = time.perf_counter() - started
        denominator = self.total_rounds or round_number
        print(f"[ROUND {round_number}/{denominator}]", flush=True)
        print(f"local_training_time={local_training_time:.6f}", flush=True)
        print(f"aggregation_time={aggregation_time:.6f}", flush=True)
        print(f"evaluation_time={evaluation_time:.6f}", flush=True)
        print(f"round_time={round_time:.6f}", flush=True)
        print(f"global_val_loss={validation.get('loss')}", flush=True)
        print(f"global_accuracy={validation.get('accuracy')}", flush=True)
        print(f"global_macro_f1={validation.get('macro_f1')}", flush=True)
        print(f"global_roc_auc={validation.get('roc_auc')}", flush=True)
        print(f"global_pr_auc={validation.get('pr_auc')}", flush=True)
        record = {
            "round": round_number, "dataset": self.dataset, "method": self.method,
            "communication_state": self.communication_mode, "participating_clients": [result.client_id for result in results],
            "total_client_samples": sum(result.num_samples for result in results),
            "global_val_loss": validation["loss"], "global_val_accuracy": validation["accuracy"], "global_val_macro_f1": validation["macro_f1"],
            "global_roc_auc": validation.get("roc_auc"), "global_pr_auc": validation.get("pr_auc"),
            "total_communication_bytes": sum(result.communication_size_bytes for result in results),
            "local_training_time": local_training_time, "aggregation_time": aggregation_time,
            "evaluation_time": evaluation_time, "round_time": round_time,
            "clients": [{"round": round_number, "client_id": result.client_id, "num_samples": result.num_samples, "local_epochs": result.local_epochs, "local_loss": result.metrics.get("loss"), "local_accuracy": result.metrics.get("accuracy"), "local_macro_f1": result.metrics.get("macro_f1"), "local_training_time": result.training_time, "uploaded_bytes": result.communication_size_bytes} for result in results],
        }
        self.round_results.append(record)
        return record