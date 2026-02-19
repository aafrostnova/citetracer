from __future__ import annotations

import re
from pathlib import Path

ENTRY_START_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
FIELD_RE = re.compile(r"(\w+)\s*=\s*(\{[^{}]*\}|\"[^\"]*\"|[^,\n]+)", re.MULTILINE)



def _strip_wrapping(value: str) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    elif value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.strip()


def _extract_entry_block(content: str, start: int) -> tuple[str, int]:
    depth = 0
    i = start
    while i < len(content):
        char = content[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : i + 1], i + 1
        i += 1
    return content[start:], len(content)


def parse_bib_file(path: str | Path) -> dict[str, dict[str, str]]:
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    entries: dict[str, dict[str, str]] = {}
    position = 0
    while True:
        match = ENTRY_START_RE.search(content, pos=position)
        if not match:
            break
        entry_type = match.group(1).strip().lower()
        citation_key = match.group(2).strip()
        block, new_position = _extract_entry_block(content, match.start())
        fields: dict[str, str] = {"entry_type": entry_type}
        for field_match in FIELD_RE.finditer(block):
            field_name = field_match.group(1).strip().lower()
            field_value = _strip_wrapping(field_match.group(2))
            fields[field_name] = field_value
        entries[citation_key] = fields
        position = new_position
    return entries


def parse_bib_directory(input_dir: str | Path) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for bib_file in sorted(Path(input_dir).rglob("*.bib")):
        merged.update(parse_bib_file(bib_file))
    return merged
