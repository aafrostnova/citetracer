from __future__ import annotations

import re
from pathlib import Path

CITE_PATTERN = re.compile(r"\\cite\w*\*?(?:\[[^\]]*\])?\{([^}]+)\}")


def discover_tex_files(input_dir: str | Path) -> list[Path]:
    return sorted(Path(input_dir).rglob("*.tex"))


def extract_citation_keys(tex_content: str) -> list[str]:
    keys: list[str] = []
    for match in CITE_PATTERN.finditer(tex_content):
        key_group = match.group(1)
        key_candidates = [candidate.strip() for candidate in key_group.split(",")]
        keys.extend([candidate for candidate in key_candidates if candidate])
    return keys


def parse_tex_directory(input_dir: str | Path) -> dict[str, list[str]]:
    key_to_locations: dict[str, list[str]] = {}
    for tex_file in discover_tex_files(input_dir):
        content = tex_file.read_text(encoding="utf-8", errors="ignore")
        keys = extract_citation_keys(content)
        for key in keys:
            key_to_locations.setdefault(key, []).append(str(tex_file))
    return key_to_locations
