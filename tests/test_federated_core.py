from copy import deepcopy

import pytest

from federated.aggregation import fedavg
from federated.client import Client
from federated.server import Server
from models.state import communication_size_bytes


class Scalar:
    shape = ()
    dtype = "float32"
    requires_grad = True

    def __init__(self, value):
        self.value = float(value)

    def __mul__(self, other):
        return Scalar(self.value * other)

    def __add__(self, other):
        return Scalar(self.value + other.value)

    def numel(self):
        return 1

    def element_size(self):
        return 4


class ShapedScalar(Scalar):
    shape = (2,)


class DoubleScalar(Scalar):
    dtype = "float64"


class FakeParameter(Scalar):
    pass


class FakeAdapter:
    def __init__(self):
        self.value = Scalar(0)

    def adapter_state_dict(self):
        return {"weight": self.value}

    def load_adapter_state_dict(self, state):
        self.value = deepcopy(state["weight"])

    def trainable_parameters(self):
        return iter([self.value])


class FakeClassifier:
    def __init__(self):
        self.value = Scalar(0)

    def state_dict(self):
        return {"weight": self.value}

    def load_state_dict(self, state):
        self.value = deepcopy(state["weight"])

    def parameters(self):
        return iter([self.value])


class FakeBackbone:
    def named_parameters(self):
        return [("frozen", FakeParameter(99))]


class FakeModel:
    def __init__(self):
        self.adapter = FakeAdapter()
        self.classifier = FakeClassifier()
        self.backbone = FakeBackbone()

    def trainable_parameters(self):
        yield from self.classifier.parameters()
        yield from self.adapter.trainable_parameters()

    def load_communication_state(self, state, mode):
        self.adapter.load_adapter_state_dict(state["adapter"])
        if mode == "adapter_plus_classifier":
            self.classifier.load_state_dict(state["classifier"])


def make_client(client_id, samples, increment):
    model = FakeModel()

    def train(local_model):
        local_model.adapter.value = Scalar(local_model.adapter.value.value + increment)
        local_model.classifier.value = Scalar(local_model.classifier.value.value + increment)
        return {"loss": 1.0, "accuracy": 0.5, "macro_f1": 0.4}

    return Client(client_id, model, None, samples, training_function=train, local_epochs=1)


def test_weighted_fedavg_and_sample_weighting():
    result = fedavg(
        [{"adapter": {"weight": Scalar(1)}, "classifier": {"weight": Scalar(1)}}, {"adapter": {"weight": Scalar(3)}, "classifier": {"weight": Scalar(3)}}],
        [1, 3],
    )
    assert result["adapter"]["weight"].value == pytest.approx(2.5)
    assert result["classifier"]["weight"].value == pytest.approx(2.5)


def test_state_compatibility_failures():
    with pytest.raises(ValueError, match="keys"):
        fedavg([{"adapter": {"weight": Scalar(1)}}, {"classifier": {"weight": Scalar(1)}}], [1, 1])
    with pytest.raises(ValueError, match="Shape"):
        fedavg([{"adapter": {"weight": Scalar(1)}}, {"adapter": {"weight": ShapedScalar(1)}}], [1, 1])
    with pytest.raises(TypeError, match="dtype"):
        fedavg([{"adapter": {"weight": Scalar(1)}}, {"adapter": {"weight": DoubleScalar(1)}}], [1, 1])
    with pytest.raises(ValueError, match="empty"):
        fedavg([], [])


def test_client_receives_and_returns_only_communication_state():
    client = make_client(0, 2, 1)
    global_state = {"adapter": {"weight": Scalar(5)}, "classifier": {"weight": Scalar(7)}}
    result = client.fit(global_state)
    assert client.last_received_state == global_state
    assert set(result.state) == {"adapter", "classifier"}
    assert "backbone" not in result.state
    assert result.communication_size_bytes == communication_size_bytes(result.state)


def test_one_complete_smoke_round_and_validation():
    server_model = FakeModel()
    clients = [make_client(0, 2, 1), make_client(1, 3, 2)]
    server = Server(server_model, "smoke", "fake", validation_loader=[], evaluator=lambda model, loader: {"loss": model.adapter.value.value, "accuracy": 0.5, "macro_f1": 0.25})
    record = server.run_round(clients, 1)
    assert server_model.adapter.value.value == pytest.approx(1.6)
    assert server_model.classifier.value.value == pytest.approx(1.6)
    assert record["global_val_loss"] == pytest.approx(1.6)
    assert record["total_client_samples"] == 5
    assert len(server.round_results) == 1