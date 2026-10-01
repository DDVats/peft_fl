from __future__ import annotations

from tests.test_federated_core import FakeModel, make_client
from federated.server import Server


def main() -> None:
    server_model = FakeModel()
    clients = [make_client(0, 2, 1), make_client(1, 3, 2)]
    server = Server(server_model, "smoke", "fake", validation_loader=[], evaluator=lambda model, loader: {"loss": model.adapter.value.value, "accuracy": 0.5, "macro_f1": 0.25})
    record = server.run_round(clients, 1)
    print({"round": record["round"], "global_state_value": server_model.adapter.value.value, "validation": record["global_val_accuracy"], "uploaded_bytes": record["total_communication_bytes"]})


if __name__ == "__main__":
    main()