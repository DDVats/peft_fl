from __future__ import annotations

TRAINING_SPEC = {"loss": "CrossEntropyLoss", "optimizer": "AdamW", "weight_decay": 0.01, "scheduler": "cosine", "warmup_ratio": 0.1, "max_epochs": 30, "gradient_accumulation": 1, "amp": False, "early_stopping_patience": 5, "early_stopping_metric": "validation macro-F1", "batch_size": 32, "seeds": [42, 123, 2024]}


def build_optimizer(*args, **kwargs):
    raise NotImplementedError("Local training is intentionally not implemented in the foundation step.")