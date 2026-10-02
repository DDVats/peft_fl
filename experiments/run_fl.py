from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import load_dataset
from data.partitioning import partition_dataset
from federated.client import Client
from federated.checkpointing import latest_checkpoint, new_status, next_round_from_checkpoint, save_round_checkpoint, write_status
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
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--partition", choices=("iid", "noniid"), default="iid")
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet concentration parameter for non-IID partitioning; required when --partition noniid")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="results/fl_checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest completed checkpoint")
    parser.add_argument("--communication-state", choices=("adapter_only", "adapter_plus_classifier"), default="adapter_plus_classifier")
    return parser.parse_args()


def _status_path(results_dir: str, experiment_name: str) -> Path:
    return Path(results_dir) / "fl_runs" / experiment_name / "status.json"


def _experiment_name(config: FederatedConfig) -> str:
    alpha = f"_alpha_{config.alpha}" if config.partition_strategy == "noniid" else ""
    return f"{config.dataset}_{config.method}_{config.partition_strategy}{alpha}_seed_{config.seed}"


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
        alpha=args.alpha,
    )
    random.seed(config.seed)
    dataset = load_dataset(config.dataset, root=args.data_root, seed=config.seed)
    import torch
    from torch.utils.data import DataLoader
    partitions = partition_dataset(dataset.train, config.number_of_clients, config.partition_strategy, config.batch_size, config.seed, config.alpha)
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
    server = Server(global_model, config.dataset, config.method, config.communication_state, validation_loader, config.federated_rounds)
    logger = ResultLogger(args.results_dir)
    initial_state = communication_state(global_model, config.communication_state)
    metadata = {
        "dataset": config.dataset, "method": config.method, "model_name": "google/vit-base-patch16-224-in21k",
        "seed": config.seed, "num_clients": config.number_of_clients, "partition": config.partition_strategy, "alpha": config.alpha,
        "local_epochs": config.local_epochs, "batch_size": config.batch_size, "learning_rate": config.learning_rate,
        "communication_state": config.communication_state, "communication_parameters": communication_parameter_count(initial_state),
        "communication_bytes": communication_size_bytes(initial_state),
    }
    experiment_name = _experiment_name(config)
    status_path = _status_path(args.results_dir, experiment_name)
    status = new_status(experiment_name, metadata, config.federated_rounds)
    start_round = 1
    if args.resume:
        checkpoint = latest_checkpoint(
            args.checkpoint_dir,
            config.dataset,
            config.method,
            config.partition_strategy,
            config.seed,
            global_model,
            config.communication_state,
            {key: metadata[key] for key in ("dataset", "method", "partition", "alpha", "seed", "num_clients", "local_epochs", "learning_rate")},
        )
        if checkpoint is not None:
            _, payload = checkpoint
            start_round = next_round_from_checkpoint(payload, config.federated_rounds)
            if start_round is None:
                print("Experiment already complete.", flush=True)
                return 0
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
            status["rounds_completed"] = start_round - 1
            status["status"] = "running"
            status["completed_at"] = None
            status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_status(status_path, status)
    started = time.perf_counter()
    try:
        for round_number in range(start_round, config.federated_rounds + 1):
            record = server.run_round(clients, round_number)
            record.update(metadata)
            checkpoint_metadata = {
                **metadata,
                "round": round_number,
                "metrics": record,
                "client_sample_counts": [client_record["num_samples"] for client_record in record["clients"]],
                "round_time": record["round_time"],
                "local_training_time": record.get("local_training_time"),
            }
            checkpoint_path = Path(args.checkpoint_dir) / config.dataset / config.method / config.partition_strategy / f"seed_{config.seed}" / f"round_{round_number:03d}.pt"
            save_round_checkpoint(checkpoint_path, global_model, checkpoint_metadata, config.communication_state)
            logger.write_round(record)
            status["rounds_completed"] = round_number
            status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status["total_time_seconds"] = time.perf_counter() - started
            write_status(status_path, status)
            print(json.dumps(record, allow_nan=False), flush=True)
        final_test = evaluate_model(global_model, DataLoader(dataset.test, batch_size=config.batch_size, shuffle=False))
        final_record = {**metadata, "final_test": final_test}
        final_path = Path(args.results_dir) / "fl_runs" / experiment_name / "final_test.json"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(json.dumps(final_record, allow_nan=False, indent=2), encoding="utf-8")
        status["status"] = "completed"
        status["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["updated_at"] = status["completed_at"]
        status["total_time_seconds"] = time.perf_counter() - started
        write_status(status_path, status)
        print(json.dumps(final_record, allow_nan=False), flush=True)
        return 0
    except KeyboardInterrupt:
        status["status"] = "interrupted"
        status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["total_time_seconds"] = time.perf_counter() - started
        write_status(status_path, status)
        print("Experiment interrupted; completed checkpoints preserved.", flush=True)
        return 130
    except Exception:
        status["status"] = "failed"
        status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["total_time_seconds"] = time.perf_counter() - started
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())