from __future__ import annotations

import re
from pathlib import Path

ENTRY_START_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
# Field name + opening `=`. The value itself is consumed by `_consume_value`
# below, which handles nested braces (the old single-level regex broke on
# values like `author={K{\"u}ttler}` where the LaTeX accent uses nested braces).
FIELD_HEAD_RE = re.compile(r"(\w+)\s*=\s*", re.MULTILINE)


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


def _consume_value(text: str, start: int) -> tuple[str, int]:
    """Consume one BibTeX field value starting at `text[start]`.

    Returns ``(value, end_index)`` where ``value`` is the raw string including
    any wrapping delimiters and ``end_index`` is the position just past the
    value's terminator (a comma or closing brace of the entry).

    Supports three value forms with nested-brace awareness:
      - ``{ ... }``  (matched balanced; nested braces allowed)
      - ``" ... "``  (matched up to the closing quote)
      - bare token   (digits / single identifier, terminated by ``,`` or ``}``)
    """
    i = start
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        return "", n

    ch = text[i]
    if ch == "{":
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[i : j + 1], j + 1
            j += 1
        # unterminated — return what we have
        return text[i:n], n

    if ch == '"':
        j = i + 1
        while j < n:
            if text[j] == '"' and text[j - 1] != "\\":
                return text[i : j + 1], j + 1
            j += 1
        return text[i:n], n

    # bare token: stop at the next comma or closing brace
    j = i
    while j < n and text[j] not in ",}\n":
        j += 1
    return text[i:j].strip(), j


def _parse_entry_fields(block: str) -> dict[str, str]:
    """Parse all `name = value` pairs from one entry block."""
    fields: dict[str, str] = {}
    # Skip past `@type{citekey,` ─ everything before the first comma is metadata.
    start = block.find(",")
    if start < 0:
        return fields
    pos = start + 1
    n = len(block)
    while pos < n:
        m = FIELD_HEAD_RE.match(block, pos)
        if not m:
            # Skip whitespace / separators between fields.
            pos += 1
            continue
        name = m.group(1).strip().lower()
        value_raw, end = _consume_value(block, m.end())
        fields[name] = _strip_wrapping(value_raw)
        # Skip the separator (comma) so the next iteration matches the next field.
        while end < n and block[end] in " ,\n\t\r":
            end += 1
        pos = end
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
        fields = _parse_entry_fields(block)
        fields["entry_type"] = entry_type
        entries[citation_key] = fields
        position = new_position
    return entries


def parse_bib_directory(input_dir: str | Path) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for bib_file in sorted(Path(input_dir).rglob("*.bib")):
        merged.update(parse_bib_file(bib_file))
    return merged
