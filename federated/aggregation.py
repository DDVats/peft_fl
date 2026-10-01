from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _shape(value: Any):
    return tuple(value.shape) if hasattr(value, "shape") else None


def _dtype(value: Any):
    return getattr(value, "dtype", None)


def _validate_states(states: list[Mapping[str, Any]], sample_counts: list[int]) -> None:
    if not states:
        raise ValueError("Cannot aggregate an empty client set")
    if len(states) != len(sample_counts):
        raise ValueError("Each client state must have a corresponding sample count")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("Client sample counts must be positive")
    expected_groups = set(states[0])
    for index, state in enumerate(states):
        if set(state) != expected_groups:
            raise ValueError(f"Client {index} communication groups/keys do not match")
        for group in expected_groups:
            expected_keys = set(states[0][group])
            if set(state[group]) != expected_keys:
                raise ValueError(f"Client {index} keys for {group} do not match")
            for key in expected_keys:
                reference = states[0][group][key]
                value = state[group][key]
                if _shape(value) != _shape(reference):
                    raise ValueError(f"Shape mismatch for {group}.{key}")
                if _dtype(value) != _dtype(reference):
                    raise TypeError(f"Unsafe dtype mismatch for {group}.{key}")


def fedavg(states: list[Mapping[str, Any]], sample_counts: list[int]) -> dict[str, dict[str, Any]]:
    _validate_states(states, sample_counts)
    total = sum(sample_counts)
    result = copy.deepcopy(states[0])
    for group in result:
        for key in result[group]:
            value = None
            for state, count in zip(states, sample_counts):
                contribution = state[group][key]
                value = contribution * (count / total) if value is None else value + contribution * (count / total)
            result[group][key] = value
    return result


class FedAvg:
    def aggregate(self, states: list[Mapping[str, Any]], sample_counts: list[int]) -> Mapping[str, Any]:
        return fedavg(states, sample_counts)