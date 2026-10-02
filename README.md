# Federated PEFT Foundation

This repository contains the reusable ViT PEFT model, CIFAR-100 data pipeline, local training, and FedAvg implementation for the initial federated experiment.

## Setup

Install the scientific runtime required for model construction in the target environment:

```text
pip install -r requirements.txt
```

Then run the foundation checks:

```text
pytest -q

python experiments/verify_real_models.py

python experiments/run_fl.py \
	--dataset cifar100 \
	--method lora \
	--num-clients 2 \
	--partition iid \
	--rounds 3 \
	--local-epochs 1 \
	--seed 42
```

Model construction downloads `google/vit-base-patch16-224-in21k` when the Hugging Face cache does not already contain it. The implementation inspects the loaded model at runtime and fails loudly when required module paths or PEFT behavior do not match the centralized source contract.

## Model and FL scope

The frozen checkpoint is `google/vit-base-patch16-224-in21k`. Communication contains only PEFT adapter state and classifier state; the frozen ViT and optimizer state are never communicated. The initial protocol is two deterministic IID clients, one local epoch, three FedAvg rounds, seed 42, batch size 32, and learning rate 0.001. The test set is evaluated only after the final round.

The H-Res adapter discovers the actual 12-layer encoder module list at runtime and removes its forward hooks through `remove_hooks()`. LoRA discovers the installed Transformers projection leaf names and uses PEFT serialization APIs.