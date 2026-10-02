from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import load_dataset
from data.partitioning import partition_dataset
from federated.client import Client
from federated.config import FederatedConfig
from federated.logging import ResultLogger
from federated.server import Server
from models.factory import build_model
from models.state import communication_parameter_count, communication_size_bytes, communication_state
from evaluation.metrics import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the initial CIFAR-100 PEFT FedAvg experiment")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--method", choices=("lora", "hres"), default="lora")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--partition", choices=("iid",), default="iid")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--communication-state", choices=("adapter_only", "adapter_plus_classifier"), default="adapter_plus_classifier")
    return parser.parse_args()


def save_checkpoint(path: Path, model, metadata: dict[str, object], mode: str) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": communication_state(model, mode), "metadata": metadata}, path)


def main() -> int:
    args = parse_args()
    config = FederatedConfig(
        dataset=args.dataset,
        method=args.method,
        communication_state=args.communication_state,
        number_of_clients=args.num_clients,
        client_fraction=1.0,
        local_epochs=args.local_epochs,
        federated_rounds=args.rounds,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        partition_strategy=args.partition,
    )
    random.seed(config.seed)
    dataset = load_dataset(config.dataset, root=args.data_root, seed=config.seed)
    import torch
    from torch.utils.data import DataLoader
    partitions = partition_dataset(dataset.train, config.number_of_clients, config.partition_strategy, config.batch_size, config.seed)
    global_model = build_model(config.dataset, config.method)
    clients = [
        Client(
            partition.client_id,
            build_model(config.dataset, config.method),
            partition.client_dataloader,
            partition.num_samples,
            communication_mode=config.communication_state,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
        )
        for partition in partitions
    ]
    validation_loader = DataLoader(dataset.validation, batch_size=config.batch_size, shuffle=False)
    server = Server(global_model, config.dataset, config.method, config.communication_state, validation_loader)
    logger = ResultLogger(args.results_dir)
    initial_state = communication_state(global_model, config.communication_state)
    metadata = {
        "dataset": config.dataset, "method": config.method, "model_name": "google/vit-base-patch16-224-in21k",
        "seed": config.seed, "num_clients": config.number_of_clients, "partition": config.partition_strategy,
        "local_epochs": config.local_epochs, "batch_size": config.batch_size, "learning_rate": config.learning_rate,
        "communication_state": config.communication_state, "communication_parameters": communication_parameter_count(initial_state),
        "communication_bytes": communication_size_bytes(initial_state),
    }
    for round_number in range(1, config.federated_rounds + 1):
        record = server.run_round(clients, round_number)
        record.update(metadata)
        logger.write_round(record)
        save_checkpoint(Path(args.results_dir) / f"{config.method}_round_{round_number}.pt", global_model, {**metadata, "round": round_number}, config.communication_state)
        print(json.dumps(record, allow_nan=False))
    final_test = evaluate_model(global_model, DataLoader(dataset.test, batch_size=config.batch_size, shuffle=False))
    final_record = {**metadata, "final_test": final_test}
    (Path(args.results_dir) / "final_test.json").write_text(json.dumps(final_record, allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps(final_record, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())