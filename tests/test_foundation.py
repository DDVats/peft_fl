import pytest

from models.classifier import classifier_parameter_count
from models.factory import DATASET_CLASSES, build_model
from models.state import adapter_plus_classifier_state, assert_backbone_excluded


def test_classifier_counts_without_torch():
    assert 768 * 100 + 100 == 76900
    assert 768 * 9 + 9 == 6921


def test_factory_combinations(require_scientific_runtime):
    for dataset in ("cifar100", "pathmnist"):
        for method in ("lora", "hres"):
            model = build_model(dataset, method)
            assert model.classifier.out_features == DATASET_CLASSES[dataset]


def test_runtime_contracts(require_scientific_runtime):
    for dataset in ("cifar100", "pathmnist"):
        for method in ("lora", "hres"):
            model = build_model(dataset, method)
            expected = 589824
            assert model.adapter.adapter_parameter_count() == expected
            assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
            state = adapter_plus_classifier_state(model)
            assert state["adapter"]
            assert state["classifier"]
            assert_backbone_excluded(model, state)


def test_hres_key_contract(require_scientific_runtime):
    model = build_model("cifar100", "hres")
    keys = set(model.adapter.adapter_state_dict())
    assert "layer_0.down.weight" in keys
    assert "layer_0.up.weight" in keys
    assert "layer_11.down.weight" in keys
    assert "layer_11.up.weight" in keys