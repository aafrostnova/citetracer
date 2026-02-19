from __future__ import annotations

import re

NUMERIC_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
AUTHORYEAR_MARKER_RE = re.compile(r"\(([A-Z][^)]*?\b(19|20)\d{2}[a-z]?)\)")



def extract_inline_markers(text: str) -> list[str]:
    markers: list[str] = []
    for match in NUMERIC_MARKER_RE.finditer(text):
        markers.append(match.group(1))
    for match in AUTHORYEAR_MARKER_RE.finditer(text):
        markers.append(match.group(1))
    return markers


def estimate_link_quality(markers: list[str], references_count: int) -> float:
    if references_count <= 0:
        return 0.0
    if not markers:
        return 0.25
    ratio = min(1.0, len(markers) / references_count)
    return max(0.25, ratio)
