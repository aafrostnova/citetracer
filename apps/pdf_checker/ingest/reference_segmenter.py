from __future__ import annotations

import re

from packages.core.models import ExtractionQuality


HEADING_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
STOP_RE = re.compile(r"^\s*(appendix|supplementary|acknowledg(?:e)?ments?)\s*$", re.IGNORECASE)
NUMERIC_START_RE = re.compile(r"^\s*(\[\d+\]|\d+\.)\s+")



def _references_block(lines: list[str]) -> list[str]:
    start = None
    for idx, line in enumerate(lines):
        if HEADING_RE.match(line):
            start = idx + 1
            break
    if start is None:
        return []

    block = []
    for line in lines[start:]:
        if STOP_RE.match(line):
            break
        block.append(line.rstrip())
    return block


def segment_references(document_text: str) -> tuple[list[str], ExtractionQuality]:
    lines = document_text.splitlines()
    block = _references_block(lines)
    if not block:
        return [], ExtractionQuality.LOW

    entries: list[str] = []
    current: list[str] = []
    numeric_mode = False

    for line in block:
        if not line.strip():
            if current:
                entries.append(" ".join(current).strip())
                current = []
            continue

        if NUMERIC_START_RE.match(line):
            numeric_mode = True
            if current:
                entries.append(" ".join(current).strip())
            current = [NUMERIC_START_RE.sub("", line).strip()]
            continue

        if current:
            current.append(line.strip())
        else:
            current = [line.strip()]

    if current:
        entries.append(" ".join(current).strip())

    if not entries:
        return [], ExtractionQuality.LOW

    if numeric_mode and len(entries) >= 8:
        quality = ExtractionQuality.HIGH
    elif len(entries) >= 4:
        quality = ExtractionQuality.MEDIUM
    else:
        quality = ExtractionQuality.LOW

    return entries, quality
