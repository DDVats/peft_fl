from __future__ import annotations

import importlib.util
import os


REQUIRED_PACKAGES = ("torch", "transformers", "peft")


def main() -> int:
    missing = [name for name in REQUIRED_PACKAGES if not importlib.util.find_spec(name)]
    if missing:
        print(f"NOT RUNTIME VERIFIED: missing packages: {', '.join(missing)}")
        print("Install requirements.txt, then set FL_VERIFY_LEARNING_RATE and rerun this focused verifier.")
        return 0
    if os.getenv("FL_VERIFY_LEARNING_RATE") is None:
        print("NOT RUNTIME VERIFIED: FL_VERIFY_LEARNING_RATE is required for the one-step check.")
        return 0
    import pytest
    return pytest.main(["-q", "tests/test_real_model_integration.py"])


if __name__ == "__main__":
    raise SystemExit(main())