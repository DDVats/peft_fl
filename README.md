# Federated PEFT Foundation

This repository contains the model, data, state-management, and evaluation foundation for future federated experiments. FedAvg, client training, federation protocols, and final experiments are intentionally not implemented.

## Setup

Install the scientific runtime required for model construction in the target environment:

```text
pip install torch transformers peft torchvision medmnist pytest
```

Then run the foundation checks:

```text
pytest -q
```

Model construction downloads `google/vit-base-patch16-224-in21k` when the Hugging Face cache does not already contain it. The implementation inspects the loaded model at runtime and fails loudly when required module paths or PEFT behavior do not match the centralized source contract.

## Scope

The classifier is always part of the trainable model. Future FL code can choose `adapter_only` or `adapter_plus_classifier` through the state-management interfaces. No client counts, rounds, local epochs, partition parameters, or aggregation rules are specified here.