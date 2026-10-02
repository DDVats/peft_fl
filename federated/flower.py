from __future__ import annotations

from typing import Any, Callable

from models.state import communication_state, load_communication_state, ndarrays_to_state, state_to_ndarrays


def require_flower():
    try:
        import flwr
    except ImportError as exc:
        raise RuntimeError("Flower integration requires flwr; install requirements.txt.") from exc
    return flwr


class FlowerClient:
    def __init__(self, client: Any, evaluator: Callable[[Any, Any], dict[str, float]] | None = None, validation_loader: Any = None) -> None:
        self.client = client
        self.evaluator = evaluator
        self.validation_loader = validation_loader
        self.template = communication_state(client.model, client.communication_mode)

    def to_client(self):
        flwr = require_flower()
        parent = self

        class NumPyClient(flwr.client.NumPyClient):
            def get_parameters(self, config):
                return parent.get_parameters(config)

            def fit(self, parameters, config):
                return parent.fit(parameters, config)

            def evaluate(self, parameters, config):
                return parent.evaluate(parameters, config)

        return NumPyClient().to_client()

    def get_parameters(self, config: dict[str, Any]):
        return state_to_ndarrays(self.template)

    def fit(self, parameters, config: dict[str, Any]):
        state = ndarrays_to_state(parameters, self.template)
        result = self.client.fit(state)
        self.template = result.state
        return state_to_ndarrays(result.state), result.num_samples, {"loss": result.metrics.get("loss", 0.0), "accuracy": result.metrics.get("accuracy", 0.0)}

    def evaluate(self, parameters, config: dict[str, Any]):
        state = ndarrays_to_state(parameters, self.template)
        load_communication_state(self.client.model, state, self.client.communication_mode)
        if self.evaluator is None:
            raise RuntimeError("Flower client evaluation requires an evaluator")
        metrics = self.evaluator(self.client.model, self.validation_loader)
        return float(metrics["loss"]), self.client.num_samples, {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}


def start_simulation(server_model: Any, clients: list[Any], validation_loader: Any, num_rounds: int, client_fraction: float = 1.0) -> Any:
    flwr = require_flower()
    if not clients:
        raise ValueError("Flower simulation requires at least one client")
    try:
        from evaluation.metrics import evaluate_model
        from flwr.common import ndarrays_to_parameters
        from flwr.server import ServerConfig
        from flwr.server.strategy import FedAvg as FlowerFedAvg
    except ImportError as exc:
        raise RuntimeError("Installed Flower does not expose the required server API") from exc
    template = communication_state(server_model, clients[0].communication_mode)
    initial_parameters = ndarrays_to_parameters(state_to_ndarrays(template))
    wrappers = [FlowerClient(client, evaluate_model, validation_loader) for client in clients]
    def client_fn(cid: str):
        wrapper = wrappers[int(cid)]
        return wrapper.to_client() if hasattr(wrapper, "to_client") else wrapper
    strategy = FlowerFedAvg(
        fraction_fit=client_fraction,
        fraction_evaluate=client_fraction,
        min_fit_clients=max(1, round(len(clients) * client_fraction)),
        min_evaluate_clients=max(1, round(len(clients) * client_fraction)),
        min_available_clients=len(clients),
        initial_parameters=initial_parameters,
    )
    return flwr.simulation.start_simulation(client_fn=client_fn, num_clients=len(clients), config=ServerConfig(num_rounds=num_rounds), strategy=strategy)