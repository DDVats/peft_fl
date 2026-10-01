from __future__ import annotations

PREPROCESSING_SPEC = {
    "resize": (224, 224), "interpolation": "bicubic", "crop_padding": 16,
    "padding_mode": "reflect", "horizontal_flip_probability": 0.5,
    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
}


def preprocessing_spec() -> dict[str, object]:
    return dict(PREPROCESSING_SPEC)