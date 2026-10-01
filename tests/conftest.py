import importlib.util

import pytest


SCIENTIFIC_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft"))


@pytest.fixture
def require_scientific_runtime():
    if not SCIENTIFIC_RUNTIME:
        pytest.skip("NOT VERIFIED: torch, transformers, and peft are required for runtime model checks")