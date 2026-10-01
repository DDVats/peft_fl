from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClientPartition:
    client_id: int
    client_indices: tuple[int, ...]
    client_dataset: Any
    client_dataloader: Any

    @property
    def num_samples(self) -> int:
        return len(self.client_indices)


def iid_partitions(dataset: Any, number_of_clients: int, batch_size: int = 32, seed: int = 42) -> list[ClientPartition]:
    if number_of_clients <= 0:
        raise ValueError("number_of_clients must be positive")
    indices = list(range(len(dataset)))
    import random
    random.Random(seed).shuffle(indices)
    chunks = [indices[index::number_of_clients] for index in range(number_of_clients)]
    try:
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        raise RuntimeError("Creating dataloaders requires torch.") from exc
    partitions = []
    for client_id, chunk in enumerate(chunks):
        client_dataset = Subset(dataset, chunk)
        partitions.append(ClientPartition(client_id, tuple(chunk), client_dataset, DataLoader(client_dataset, batch_size=batch_size, shuffle=True)))
    return partitions


def partition_dataset(dataset: Any, number_of_clients: int, strategy: str = "iid", batch_size: int = 32, seed: int = 42) -> list[ClientPartition]:
    if strategy != "iid":
        raise NotImplementedError(f"Partition strategy is not implemented: {strategy}")
    return iid_partitions(dataset, number_of_clients, batch_size, seed)