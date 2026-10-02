import pytest

from data.datasets import stratified_split_indices
import data.partitioning as partitioning
from data.partitioning import dirichlet_partition_indices, iid_partition_indices, partition_dataset
from federated.config import FederatedConfig


def test_cifar_stratified_split_is_deterministic_and_complete():
    labels = [label for label in range(100) for _ in range(500)]
    train_a, validation_a = stratified_split_indices(labels, seed=42)
    train_b, validation_b = stratified_split_indices(labels, seed=42)
    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 45000
    assert len(validation_a) == 5000
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a) | set(validation_a) == set(range(50000))


def _labeled_dataset(class_count=10, samples_per_class=100):
    return [(0, label) for label in range(class_count) for _ in range(samples_per_class)]


def test_iid_works_without_alpha_and_ignores_alpha():
    dataset = _labeled_dataset()
    assert iid_partition_indices(dataset, 2, seed=42) == iid_partition_indices(dataset, 2, seed=42)
    assert FederatedConfig(dataset="cifar100", method="lora", partition_strategy="iid").alpha is None


def test_noniid_requires_positive_explicit_alpha():
    dataset = _labeled_dataset()
    with pytest.raises(ValueError, match="alpha"):
        partition_dataset(dataset, 2, strategy="noniid", seed=42)
    with pytest.raises(ValueError, match="alpha"):
        FederatedConfig(dataset="cifar100", method="lora", partition_strategy="noniid")
    with pytest.raises(ValueError, match="greater than zero"):
        FederatedConfig(dataset="cifar100", method="lora", partition_strategy="noniid", alpha=0.0)
    with pytest.raises(ValueError, match="greater than zero"):
        dirichlet_partition_indices(dataset, 2, alpha=-0.5, seed=42)


def test_noniid_alpha_is_propagated_and_reproducible():
    dataset = _labeled_dataset()
    first = dirichlet_partition_indices(dataset, 2, alpha=0.5, seed=42)
    second = dirichlet_partition_indices(dataset, 2, alpha=0.5, seed=42)
    other_alpha = dirichlet_partition_indices(dataset, 2, alpha=10.0, seed=42)
    assert first == second
    assert first != other_alpha
    assert sorted(index for partition in first for index in partition) == list(range(len(dataset)))
    assert all(partition for partition in first)


def test_partition_dataset_forwards_alpha(monkeypatch):
    captured = {}

    def fake_build(dataset, chunks, batch_size, seed, strategy, alpha):
        captured["alpha"] = alpha
        return chunks

    monkeypatch.setattr(partitioning, "_build_partitions", fake_build)
    result = partition_dataset(_labeled_dataset(), 2, strategy="noniid", batch_size=4, seed=42, alpha=0.5)
    assert captured["alpha"] == 0.5
    assert len(result) == 2