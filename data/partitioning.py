from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClientPartition:
    client_id: int
    client_indices: tuple[int, ...]
    client_dataset: Any
    client_dataloader: Any
    strategy: str = "iid"
    alpha: float | None = None

    @property
    def num_samples(self) -> int:
        return len(self.client_indices)


def iid_partitions(dataset: Any, number_of_clients: int, batch_size: int = 32, seed: int = 42) -> list[ClientPartition]:
    chunks = iid_partition_indices(dataset, number_of_clients, seed)
    return _build_partitions(dataset, chunks, batch_size, seed, "iid", None)


def _dataset_labels(dataset: Any) -> list[int]:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_labels = _dataset_labels(dataset.dataset)
        return [base_labels[index] for index in dataset.indices]
    for attribute in ("targets", "labels"):
        if hasattr(dataset, attribute):
            values = getattr(dataset, attribute)
            if hasattr(values, "tolist"):
                values = values.tolist()
            return [int(value[0] if isinstance(value, (list, tuple)) else value) for value in values]
    labels = []
    for item in dataset:
        if isinstance(item, dict):
            labels.append(int(item["labels"]))
        else:
            labels.append(int(item[1]))
    return labels


def iid_partition_indices(dataset: Any, number_of_clients: int, seed: int = 42) -> list[list[int]]:
    if number_of_clients <= 0:
        raise ValueError("number_of_clients must be positive")
    indices = list(range(len(dataset)))
    import random
    random.Random(seed).shuffle(indices)
    return [indices[index::number_of_clients] for index in range(number_of_clients)]


def dirichlet_partition_indices(dataset: Any, number_of_clients: int, alpha: float, seed: int = 42) -> list[list[int]]:
    if number_of_clients <= 0:
        raise ValueError("number_of_clients must be positive")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be greater than zero")
    labels = _dataset_labels(dataset)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    import random
    generator = random.Random(seed)
    for attempt in range(100):
        partitions = [[] for _ in range(number_of_clients)]
        for label in sorted(by_class):
            class_indices = by_class[label][:]
            generator.shuffle(class_indices)
            weights = [generator.gammavariate(alpha, 1.0) for _ in range(number_of_clients)]
            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            for index in class_indices:
                client_id = generator.choices(range(number_of_clients), weights=probabilities, k=1)[0]
                partitions[client_id].append(index)
        if all(partitions):
            for partition in partitions:
                generator.shuffle(partition)
            return partitions
        generator.seed(seed + attempt + 1)
    raise RuntimeError("Could not produce non-empty Dirichlet client partitions")


def _build_partitions(dataset: Any, chunks: list[list[int]], batch_size: int, seed: int, strategy: str, alpha: float | None) -> list[ClientPartition]:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        raise RuntimeError("Creating dataloaders requires torch.") from exc
    partitions = []
    for client_id, chunk in enumerate(chunks):
        client_dataset = Subset(dataset, chunk)
        generator = torch.Generator().manual_seed(seed + client_id)
        partitions.append(ClientPartition(client_id, tuple(chunk), client_dataset, DataLoader(client_dataset, batch_size=batch_size, shuffle=True, generator=generator), strategy, alpha))
    return partitions


def partition_dataset(dataset: Any, number_of_clients: int, strategy: str = "iid", batch_size: int = 32, seed: int = 42, alpha: float | None = None) -> list[ClientPartition]:
    if strategy == "iid":
        return iid_partitions(dataset, number_of_clients, batch_size, seed)
    if strategy == "noniid":
        if alpha is None:
            raise ValueError("--alpha is required when --partition noniid is selected")
        return _build_partitions(dataset, dirichlet_partition_indices(dataset, number_of_clients, alpha, seed), batch_size, seed, strategy, alpha)
    raise ValueError(f"Unsupported partition strategy: {strategy}")