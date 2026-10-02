from __future__ import annotations

import importlib.util


REQUIRED_PACKAGES = ("torch", "transformers", "peft")


def main() -> int:
    missing = [name for name in REQUIRED_PACKAGES if not importlib.util.find_spec(name)]
    if missing:
        print(f"NOT RUNTIME VERIFIED: missing packages: {', '.join(missing)}")
        print("Install requirements.txt and rerun this focused verifier.")
        return 0
    import pytest
    return pytest.main(["-q", "tests/test_real_model_integration.py"])


if __name__ == "__main__":
    raise SystemExit(main())