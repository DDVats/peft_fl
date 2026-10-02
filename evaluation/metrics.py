from __future__ import annotations


def accuracy(*args, **kwargs):
    predictions, targets = args[:2]
    return float((predictions == targets).sum()) / targets.numel()


def macro_f1(*args, **kwargs):
    predictions, targets = args[:2]
    classes = set(targets.tolist()) | set(predictions.tolist())
    values = []
    for class_id in classes:
        true_positive = sum(prediction == class_id and target == class_id for prediction, target in zip(predictions.tolist(), targets.tolist()))
        false_positive = sum(prediction == class_id and target != class_id for prediction, target in zip(predictions.tolist(), targets.tolist()))
        false_negative = sum(prediction != class_id and target == class_id for prediction, target in zip(predictions.tolist(), targets.tolist()))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values) if values else 0.0


def loss(*args, **kwargs):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Loss calculation requires torch.") from exc
    return float(torch.nn.functional.cross_entropy(*args, **kwargs))
    
def evaluate_model(model, dataloader):
    if dataloader is None:
        raise ValueError("A validation dataloader is required for global evaluation")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Evaluation requires torch.") from exc
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = (batch["pixel_values"], batch["labels"]) if isinstance(batch, dict) else (batch[0], batch[1])
            logits = model.forward(inputs)
            total_loss += float(criterion(logits, labels)) * labels.numel()
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
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items, "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0}