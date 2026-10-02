from __future__ import annotations

import time
from typing import Any, Callable

from models.backbone import is_peft_adapter_parameter


def _batch_inputs(batch: Any):
    if isinstance(batch, dict):
        return batch["pixel_values"], batch["labels"]
    return batch[0], batch[1]


def train_local(model: Any, dataloader: Any, local_epochs: int, learning_rate: float) -> dict[str, float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Local training requires torch.") from exc
    if local_epochs <= 0:
        raise ValueError("local_epochs must be positive")
    started = time.perf_counter()
    parameters = list(model.trainable_parameters())
    if any(not parameter.requires_grad for parameter in parameters):
        raise AssertionError("Local optimizer received a frozen parameter")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    total_steps = max(1, local_epochs * len(dataloader))
    warmup_steps = int(total_steps * 0.1)
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    predictions = []
    targets = []
    for _ in range(local_epochs):
        for batch in dataloader:
            inputs, labels = _batch_inputs(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward(inputs)
            batch_loss = criterion(logits, labels)
            batch_loss.backward()
            if any(
                parameter.grad is not None
                for name, parameter in model.backbone.named_parameters()
                if not is_peft_adapter_parameter(name)
            ):
                raise AssertionError("Frozen ViT backbone received a gradient")
            if any(parameter.grad is not None and not parameter.requires_grad for _, parameter in model.named_parameters()):
                raise AssertionError("A non-trainable parameter received a gradient")
            optimizer.step()
            scheduler.step()
            total_loss += float(batch_loss.detach()) * labels.numel()
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_items += labels.numel()
            predictions.extend(logits.argmax(dim=1).tolist())
            targets.extend(labels.tolist())
    classes = set(targets) | set(predictions)
    f1_values = []
    for class_id in classes:
            true_positive = sum(prediction == class_id and target == class_id for prediction, target in zip(predictions, targets))
            false_positive = sum(prediction == class_id and target != class_id for prediction, target in zip(predictions, targets))
            false_negative = sum(prediction != class_id and target == class_id for prediction, target in zip(predictions, targets))
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    if total_items == 0:
        raise ValueError("Cannot train on an empty dataloader")
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items, "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0, "training_time": time.perf_counter() - started}


def make_training_function(dataloader: Any, local_epochs: int, learning_rate: float) -> Callable[[Any], dict[str, float]]:
    return lambda model: train_local(model, dataloader, local_epochs, learning_rate)