from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultLogger:
    def __init__(self, directory: str = "results") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_round(self, record: dict[str, Any]) -> None:
        with (self.directory / "rounds.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()