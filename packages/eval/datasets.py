from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetItem:
    paper_id: str
    pipeline: str
    input_path: str
    ground_truth_path: str


@dataclass
class DatasetManifest:
    name: str
    description: str
    items: list[DatasetItem]



def load_manifest(path: str | Path) -> DatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [DatasetItem(**item) for item in payload.get("items", [])]
    return DatasetManifest(
        name=payload.get("name", "unknown_suite"),
        description=payload.get("description", ""),
        items=items,
    )


def load_ground_truth(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in payload.items()}
