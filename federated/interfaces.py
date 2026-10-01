from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class AggregationInterface(ABC):
    @abstractmethod
    def aggregate(self, states: list[Mapping[str, Any]], sample_counts: list[int]) -> Mapping[str, Any]: ...


class ClientInterface(ABC):
    @abstractmethod
    def fit(self, state: Mapping[str, Any]): ...


class ServerInterface(ABC):
    @abstractmethod
    def distribute(self, state: Mapping[str, Any]): ...