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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    if base_dir is not None:
        base_candidate = (base_dir / candidate).resolve()
        if base_candidate.exists():
            return base_candidate

    repo_candidate = (_repo_root() / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    raw_text = candidate.as_posix()
    if raw_text.startswith("data/fixtures/"):
        remapped = Path("data") / raw_text[len("data/fixtures/") :]
        if remapped.exists():
            return remapped
        repo_remapped = (_repo_root() / remapped).resolve()
        if repo_remapped.exists():
            return repo_remapped

    return candidate



def load_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = _resolve_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent

    items = []
    for item in payload.get("items", []):
        items.append(
            DatasetItem(
                paper_id=item["paper_id"],
                pipeline=item["pipeline"],
                input_path=str(_resolve_path(item["input_path"], base_dir=manifest_dir)),
                ground_truth_path=str(_resolve_path(item["ground_truth_path"], base_dir=manifest_dir)),
            )
        )
    return DatasetManifest(
        name=payload.get("name", "unknown_suite"),
        description=payload.get("description", ""),
        items=items,
    )


def load_ground_truth(path: str | Path) -> dict[str, str]:
    payload = json.loads(_resolve_path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in payload.items()}
