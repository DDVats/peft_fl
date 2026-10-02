from __future__ import annotations

PREPROCESSING_SPEC = {
    "resize": (224, 224), "interpolation": "bicubic", "crop_padding": 16,
    "padding_mode": "reflect", "horizontal_flip_probability": 0.5,
    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
}


def preprocessing_spec() -> dict[str, object]:
    return dict(PREPROCESSING_SPEC)


def build_transforms(training: bool):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("CIFAR-100 transforms require torchvision.") from exc
    operations = [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)]
    if training:
        operations.extend([
            transforms.RandomCrop(224, padding=16, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=PREPROCESSING_SPEC["mean"], std=PREPROCESSING_SPEC["std"]),
    ])
    return transforms.Compose(operations)