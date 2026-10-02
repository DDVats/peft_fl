from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .preprocessing import build_transforms

DATASET_SPEC = {
    "cifar100": {"num_classes": 100, "train": 45000, "validation": 5000, "test": 10000, "split": "existing centralized stratified split"},
    "pathmnist": {"num_classes": 9, "split": "official train/validation/test splits"},
}


def dataset_spec(name: str) -> dict[str, object]:
    try:
        return dict(DATASET_SPEC[name.lower()])
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc


@dataclass(frozen=True)
class DatasetBundle:
    train: Any
    validation: Any
    test: Any
    num_classes: int


def stratified_split_indices(labels: list[int], validation_size: int = 5000, seed: int = 42) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(labels) != 50000 or validation_size != 5000:
        raise ValueError("The CIFAR-100 research split requires 50,000 samples and 5,000 validation samples")
    import random
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)
    if len(by_class) != 100 or any(len(indices) != 500 for indices in by_class.values()):
        raise ValueError("Unexpected CIFAR-100 class distribution")
    generator = random.Random(seed)
    train_indices, validation_indices = [], []
    for label in sorted(by_class):
        indices = by_class[label][:]
        generator.shuffle(indices)
        validation_indices.extend(indices[:50])
        train_indices.extend(indices[50:])
    generator.shuffle(train_indices)
    generator.shuffle(validation_indices)
    return tuple(train_indices), tuple(validation_indices)


def load_cifar100(root: str = "data", download: bool = True, seed: int = 42) -> DatasetBundle:
    try:
        from torch.utils.data import Subset
        from torchvision.datasets import CIFAR100
    except ImportError as exc:
        raise RuntimeError("CIFAR-100 loading requires torch and torchvision.") from exc
    raw_train = CIFAR100(root=root, train=True, transform=None, download=download)
    train_indices, validation_indices = stratified_split_indices(list(raw_train.targets), seed=seed)
    train_source = CIFAR100(root=root, train=True, transform=build_transforms(training=True), download=False)
    validation_source = CIFAR100(root=root, train=True, transform=build_transforms(training=False), download=False)
    test_source = CIFAR100(root=root, train=False, transform=build_transforms(training=False), download=download)
    return DatasetBundle(Subset(train_source, train_indices), Subset(validation_source, validation_indices), test_source, 100)


def load_dataset(name: str, root: str = "data", download: bool = True, seed: int = 42) -> DatasetBundle:
    if name.lower() == "cifar100":
        return load_cifar100(root=root, download=download, seed=seed)
    raise NotImplementedError("PathMNIST loading is reserved for its official-split integration.")