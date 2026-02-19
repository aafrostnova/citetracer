from __future__ import annotations

import re
from pathlib import Path

ENTRY_START_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


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


def _parse_fields(block: str) -> dict[str, str]:
    """Parse BibTeX field key=value pairs from an entry block using a state machine.

    Handles nested braces (e.g. ``{An {NLP} Paper}``) and quoted strings.
    """
    fields: dict[str, str] = {}
    # Skip the entry header (@type{key,) by advancing past the first comma.
    start = block.find(",")
    if start == -1:
        return fields
    i = start + 1
    n = len(block)
    while i < n:
        # Skip whitespace.
        while i < n and block[i] in " \t\n\r":
            i += 1
        # End of block or closing brace.
        if i >= n or block[i] == "}":
            break
        # Read field name (alphanumeric + underscore).
        name_start = i
        while i < n and (block[i].isalnum() or block[i] == "_"):
            i += 1
        field_name = block[name_start:i].strip().lower()
        if not field_name:
            i += 1
            continue
        # Skip whitespace, then expect '='.
        while i < n and block[i] in " \t\n\r":
            i += 1
        if i >= n or block[i] != "=":
            continue
        i += 1  # consume '='
        # Skip whitespace before value.
        while i < n and block[i] in " \t\n\r":
            i += 1
        if i >= n:
            break
        # Read value according to its delimiter.
        if block[i] == "{":
            # Brace-wrapped value: track nesting depth to support {A {B} C}.
            depth = 0
            val_start = i
            while i < n:
                if block[i] == "{":
                    depth += 1
                elif block[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            field_value = _strip_wrapping(block[val_start:i])
        elif block[i] == '"':
            # Quoted value: read until the closing unescaped '"'.
            i += 1  # consume opening quote
            val_start = i
            while i < n:
                if block[i] == "\\" and i + 1 < n:
                    i += 2  # skip escape sequence
                elif block[i] == '"':
                    break
                else:
                    i += 1
            field_value = block[val_start:i].strip()
            if i < n:
                i += 1  # consume closing quote
        else:
            # Bare value: read until ',' or closing '}'.
            val_start = i
            while i < n and block[i] not in (",", "}"):
                i += 1
            field_value = block[val_start:i].strip()
        fields[field_name] = field_value
        # Consume optional trailing comma.
        while i < n and block[i] in " \t\n\r":
            i += 1
        if i < n and block[i] == ",":
            i += 1
    return fields


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
        fields: dict[str, str] = _parse_fields(block)
        fields["entry_type"] = entry_type
        entries[citation_key] = fields
        position = new_position
    return entries


def parse_bib_directory(input_dir: str | Path) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for bib_file in sorted(Path(input_dir).rglob("*.bib")):
        merged.update(parse_bib_file(bib_file))
    return merged
