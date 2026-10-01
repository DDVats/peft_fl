from __future__ import annotations

DATASET_SPEC = {
    "cifar100": {"num_classes": 100, "train": 45000, "validation": 5000, "test": 10000, "split": "existing centralized stratified split"},
    "pathmnist": {"num_classes": 9, "split": "official train/validation/test splits"},
}


def dataset_spec(name: str) -> dict[str, object]:
    try:
        return dict(DATASET_SPEC[name.lower()])
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {name}") from exc


def load_dataset(name: str):
    raise NotImplementedError("Dataset download/loading is pending the federated protocol implementation.")