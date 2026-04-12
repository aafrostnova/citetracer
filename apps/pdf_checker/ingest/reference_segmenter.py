from __future__ import annotations

import json
import importlib
import os
import re
import sys
import tempfile
import traceback
import types
from datetime import datetime
from pathlib import Path
from typing import Any

from packages.core.models import ExtractionQuality


START_HEADING_RE = re.compile(
    # Match a "References" / "Bibliography" heading. The trailing junk allows
    # for: pure end-of-line, or non-alphabetic stuff (line numbers like "364",
    # bracketed page refs, punctuation). We REJECT continuation by a letter,
    # which would mean the word is part of "Referenced" / "Bibliographer" etc.
    r"^\s*(?:\d+(?:\.\d+)*[\)\.]?\s*)?(references|bibliography|reference list|参考文献)(?:[\s\d\W]*)?$",
    re.IGNORECASE,
)
END_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[\)\.]?\s*)?(appendix|appendices|acknowledg(?:e)?ments?|"
    r"supplement(?:ary(?: material)?)?|about the authors|biographies?)\s*$",
    re.IGNORECASE,
)
APPENDIX_STYLE_HEADING_RE = re.compile(r"^\s*[A-Z](?:\.\d+)?\.\s+[A-Z][A-Za-z].{0,140}$")
NUMBERED_REF_RE = re.compile(r"(?m)^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.)\s+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
UNNUMBERED_START_PATTERNS = [
    re.compile(r"(?<=\.\s)([A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*){1,4})"),
    re.compile(r"(?<=\.\s)(\d{2,6}\.\s+[A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*){1,4})"),
    re.compile(r"(?<=\.\s)([A-Z][A-Za-z&\-]{2,30}\.\s+[A-Z])"),
]
ENTRY_PREFIX_RE = re.compile(r"^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.)\s+")
OCR_TAGGED_BLOCK_RE = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>\s*(?P<content>.*?)(?=(?:\n?\s*<\|ref\|>|$))",
    re.DOTALL,
)
APPENDIX_HEADING_KEYWORDS = (
    "appendix",
    "supplement",
    "proof",
    "proofs",
    "related work",
    "experiment",
    "experimental",
    "implementation",
    "details",
    "additional",
    "ablation",
    "theorem",
    "lemma",
    "notation",
)
NARRATIVE_REF_PREFIXES = (
    "exact verification approaches",
    "scalable approximation methods",
    "abstract interpretation methods",
    "specialized and geometric approaches",
    "relaxation-based training",
    "advanced training strategies",
)

_LOCAL_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_LOCAL_VLLM_MODEL_CACHE: dict[str, tuple[Any, Any, Any, Any, Any]] = {}


def normalize_space(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _local_ocr_tmp_root() -> Path:
    explicit = (os.getenv("CITATION_CHECKER_LOCAL_OCR_TMP_ROOT", "") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path.cwd() / ".tmp" / "pdf_checker_local_ocr_pages"


def _normalize_local_inference_backend(backend: str | None) -> str:
    value = (backend or "").strip().lower()
    if value in {"hf", "vllm"}:
        return value
    return "hf"


def _resolve_deepseek_ocr2_vllm_code_dir(model_path: str) -> Path:
    explicit = (os.getenv("CITATION_CHECKER_DEEPSEEK_OCR2_VLLM_CODE_PATH", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path.resolve()

    candidates: list[Path] = []
    model_as_path = Path(model_path).expanduser()
    candidates.append(model_as_path / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm")
    candidates.append(model_as_path / "DeepSeek-OCR2-vllm")
    candidates.append(Path.cwd() / "DeepSeek-OCR-2" / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise RuntimeError(
        "Cannot locate DeepSeek-OCR-2 vLLM code directory. "
        "Set CITATION_CHECKER_DEEPSEEK_OCR2_VLLM_CODE_PATH to "
        ".../DeepSeek-OCR2-master/DeepSeek-OCR2-vllm."
    )


def _safe_write_text(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception:
        pass


def _normalize_heading_text(text: str) -> str:
    heading = text.replace("\r\n", "\n").replace("\r", "\n")
    heading = re.sub(r"(?m)^\s*#+\s*", "", heading)
    heading = normalize_space(heading)
    return heading


def _strip_reference_artifacts(text: str, preserve_line_breaks: bool = False) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00ad", "")
    # Page separators inserted by our wrapper should not become part of entries.
    cleaned = re.sub(r"\s*\[\[PAGE\s+\d+\]\]\s*", "\n", cleaned, flags=re.IGNORECASE)
    # DeepSeek OCR tagged markdown fallback can leak label/detection markers into text.
    cleaned = re.sub(
        r"(?im)\btext\s*\[\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\]",
        "\n",
        cleaned,
    )
    cleaned = re.sub(r"\[\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\]", "\n", cleaned)
    cleaned = re.sub(r"<\|[^|]+\|>", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    if preserve_line_breaks:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    return normalize_space(cleaned)


def _find_heading_line_index(lines: list[str], heading_re: re.Pattern[str]) -> int | None:
    for idx, line in enumerate(lines):
        if heading_re.match(line.strip()):
            return idx
    return None


def _is_appendix_style_heading(line: str) -> bool:
    stripped = normalize_space(line)
    if not stripped or YEAR_RE.search(stripped):
        return False
    if "," in stripped or ";" in stripped:
        return False
    lowered = stripped.lower()
    if "http://" in lowered or "https://" in lowered or "doi:" in lowered:
        return False
    if "et al" in lowered:
        return False

    if END_HEADING_RE.match(stripped):
        return True

    if APPENDIX_STYLE_HEADING_RE.match(stripped):
        return any(keyword in lowered for keyword in APPENDIX_HEADING_KEYWORDS)

    if re.match(r"^\s*[A-Z](?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9\- ]{2,100}$", stripped):
        return any(keyword in lowered for keyword in APPENDIX_HEADING_KEYWORDS)

    return False


def _find_appendix_style_line_index(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if _is_appendix_style_heading(line):
            return idx
    return None


def _looks_like_reference_entry(text: str) -> bool:
    cleaned = normalize_space(text)
    if len(cleaned) < 20:
        return False
    lowered = cleaned.lower()
    if lowered.startswith(NARRATIVE_REF_PREFIXES):
        return False

    has_signal = bool(
        YEAR_RE.search(cleaned)
        or re.search(r"https?://|doi:\s*|10\.\d{4,9}/|arxiv:", cleaned, flags=re.IGNORECASE)
    )
    if not has_signal:
        return False

    if re.match(r"^(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.)\s+", cleaned):
        return True

    head = cleaned[:140]
    if re.search(r",[ ](?:[A-Z]\.|[A-Z][a-z])", head):
        return True
    if re.search(r"\b(?:Proceedings|Journal|arXiv|PMLR|NeurIPS|ICLR|ICML)\b", cleaned):
        return cleaned[0].isupper()
    return cleaned[0].isupper() and len(cleaned.split()) <= 70


def _has_reference_signal(text: str) -> bool:
    return bool(
        YEAR_RE.search(text)
        or re.search(r"https?://|doi:\s*|10\.\d{4,9}/|arxiv:", text, flags=re.IGNORECASE)
    )


def _looks_like_reference_prefix_fragment(text: str) -> bool:
    cleaned = normalize_space(text)
    if len(cleaned) < 25:
        return False
    lowered = cleaned.lower()
    if lowered.startswith(("in proceedings", "proceedings of", "pp.", "volume ", "vol.", "association for", "pmlr", "springer")):
        return False
    # Typical author-prefix start: "Surname, A." / "Surname, A., Surname, B."
    if re.match(r"^[A-Z][A-Za-z'`\-]+,\s+(?:[A-Z]\.\s*){1,4}", cleaned):
        return True
    if re.match(r"^[A-Z][A-Za-z'`\-]+,\s+[A-Z][a-z]{1,20}", cleaned):
        return True
    return False


def _looks_like_reference_continuation_fragment(text: str) -> bool:
    cleaned = normalize_space(text)
    if len(cleaned) < 15:
        return False
    lowered = cleaned.lower()
    continuation_prefixes = (
        "in proceedings",
        "proceedings of",
        "findings of",
        "in advances in",
        "in international conference",
        "pp.",
        "pp ",
        "volume ",
        "vol.",
        "association for",
        "new york, ny",
        "florence, italy",
        "hong kong, china",
        "abu dhabi",
        "pmlr",
    )
    if lowered.startswith(continuation_prefixes):
        return True
    if re.match(r"^[A-Z][a-z]+,\s+[A-Z][a-z]+\s+\d{4}\.", cleaned):
        return True
    if re.match(r"^\d{2,6}\s*[-–]\s*\d{2,6}(?:,\s*\d{4})?$", cleaned):
        return True
    return False


def _looks_like_reference_fragment(text: str) -> bool:
    cleaned = normalize_space(text)
    return _looks_like_reference_prefix_fragment(cleaned) or _looks_like_reference_continuation_fragment(cleaned)


def _merge_fragmented_entries(entries: list[str]) -> list[str]:
    merged: list[str] = []
    for raw in entries:
        current = normalize_space(raw)
        if not current:
            continue
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        prev_has_signal = _has_reference_signal(previous)
        curr_has_signal = _has_reference_signal(current)
        prev_incomplete_tail = bool(
            re.search(r"(?:,\s*|:\s*|\bpp\.?\s*|\bvolume\s+\d+\s*)$", previous, flags=re.IGNORECASE)
            or previous.endswith(("In", "in", "In:", "in:"))
        )

        should_merge = False
        if _looks_like_reference_continuation_fragment(current) and (_looks_like_reference_prefix_fragment(previous) or prev_incomplete_tail):
            should_merge = True
        elif _looks_like_reference_continuation_fragment(current) and not prev_has_signal:
            should_merge = True
        elif _looks_like_reference_prefix_fragment(previous) and not prev_has_signal and curr_has_signal:
            should_merge = True

        if should_merge:
            merged[-1] = normalize_space(f"{previous} {current}")
        else:
            merged.append(current)
    return merged


def _split_tagged_block_on_reference_boundaries(content: str) -> tuple[str, bool, bool]:
    """
    Returns (trimmed_content, has_reference_start, has_reference_end).
    Handles cases where one OCR `text` block contains preface text + REFERENCES +
    reference entries (or reference entries + next section heading) together.
    """
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start_idx = _find_heading_line_index(lines, START_HEADING_RE)

    has_start = start_idx is not None
    if start_idx is not None:
        lines = lines[start_idx + 1 :]

    end_idx = _find_heading_line_index(lines, END_HEADING_RE)
    if end_idx is None:
        end_idx = _find_appendix_style_line_index(lines)

    has_end = end_idx is not None
    if end_idx is not None:
        lines = lines[:end_idx]

    return "\n".join(lines).strip(), has_start, has_end


def _extract_reference_entries_from_tagged_markdown(markdown_text: str) -> list[str]:
    blocks: list[tuple[str, str]] = []
    for match in OCR_TAGGED_BLOCK_RE.finditer(markdown_text):
        label = normalize_space(match.group("label").lower())
        content = match.group("content").strip()
        if not content:
            continue
        blocks.append((label, content))

    has_explicit_start = False
    prepared_blocks: list[tuple[str, str, str, bool, bool]] = []
    for label, content in blocks:
        content_clean = _strip_reference_artifacts(content, preserve_line_breaks=True)
        if not content_clean:
            continue
        heading = _normalize_heading_text(content_clean)
        trimmed_content, has_start_inside, has_end_inside = _split_tagged_block_on_reference_boundaries(content_clean)
        if has_start_inside:
            has_explicit_start = True
        if label in {"sub_title", "title", "section_title", "figure_title"} and START_HEADING_RE.match(heading):
            has_explicit_start = True
        prepared_blocks.append((label, heading, trimmed_content, has_start_inside, has_end_inside))

    # Some papers start directly from references (no visible REFERENCES heading in OCR blocks).
    auto_start = False
    if not has_explicit_start:
        probe = [item for item in prepared_blocks if item[0] == "text"][:10]
        if probe:
            ref_like = sum(1 for _, _, content, _, _ in probe if _looks_like_reference_entry(content))
            auto_start = ref_like >= 3

    in_references = auto_start
    extracted: list[str] = []
    for label, heading, trimmed_content, has_start_inside, has_end_inside in prepared_blocks:
        if has_start_inside:
            in_references = True
            if not trimmed_content:
                continue
        elif label in {"sub_title", "title", "section_title", "figure_title"}:
            if START_HEADING_RE.match(heading):
                in_references = True
                continue
            if in_references and (_is_appendix_style_heading(heading) or heading):
                break

        if in_references and label == "text":
            if trimmed_content and (_looks_like_reference_entry(trimmed_content) or _looks_like_reference_fragment(trimmed_content)):
                extracted.append(trimmed_content)
            if has_end_inside:
                break
        elif in_references and has_end_inside:
            break

    return extracted


def _page_has_heading(page_text: str, heading_re: re.Pattern[str]) -> bool:
    for line in page_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if heading_re.match(line.strip()):
            return True
    return False


def _page_has_appendix_style_heading(page_text: str) -> bool:
    lines = page_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line in lines[:24]:
        if _is_appendix_style_heading(line):
            return True
    return False


def find_reference_page_range(page_texts: list[str]) -> tuple[int, int, bool]:
    if not page_texts:
        return 0, 0, False

    start_candidates = [i for i, text in enumerate(page_texts) if _page_has_heading(text, START_HEADING_RE)]
    n_pages = len(page_texts)
    half = n_pages // 2
    heading_detected = bool(start_candidates)

    if start_candidates:
        start_page = next((idx for idx in start_candidates if idx >= half), start_candidates[0])
    else:
        start_page = max(0, int(n_pages * 0.65))

    end_page = n_pages - 1
    for idx in range(start_page + 1, n_pages):
        text = page_texts[idx]
        if _page_has_heading(text, END_HEADING_RE) or _page_has_appendix_style_heading(text):
            # Keep this page included; line-level block trimming will cut at the end heading.
            end_page = idx
            break

    if end_page < start_page:
        end_page = start_page

    return start_page, end_page, heading_detected


def build_page_range_text(page_texts: list[str], start_page: int, end_page: int) -> str:
    if not page_texts:
        return ""
    first = max(0, start_page)
    last = min(end_page, len(page_texts) - 1)
    parts = [f"[[PAGE {i + 1}]]\n{page_texts[i]}" for i in range(first, last + 1)]
    return "\n\n".join(parts)


def find_reference_block(full_text: str) -> tuple[str, bool]:
    lines = full_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start_candidates = [i for i, line in enumerate(lines) if START_HEADING_RE.match(line.strip())]

    if not start_candidates:
        start = max(0, int(len(lines) * 0.65))
        return "\n".join(lines[start:]).strip(), False

    half = len(lines) // 2
    start_heading = next((idx for idx in start_candidates if idx >= half), start_candidates[0])
    start = start_heading + 1
    end = len(lines)

    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if END_HEADING_RE.match(stripped):
            end = idx
            break
        if _is_appendix_style_heading(stripped):
            end = idx
            break

    return "\n".join(lines[start:end]).strip(), True


def _split_unnumbered_references(text: str) -> list[str]:
    flat = normalize_space(text.replace("\n", " "))
    if not flat:
        return []

    starts = {0}
    for pattern in UNNUMBERED_START_PATTERNS:
        for match in pattern.finditer(flat):
            starts.add(match.start(1))

    ordered = sorted(starts)
    if len(ordered) <= 1:
        return [flat]

    chunks: list[str] = []
    for idx, start in enumerate(ordered):
        end = ordered[idx + 1] if idx + 1 < len(ordered) else len(flat)
        piece = normalize_space(flat[start:end])
        if piece:
            chunks.append(piece)
    return chunks


def _filter_reference_like_entries(entries: list[str]) -> list[str]:
    filtered: list[str] = []
    for entry in entries:
        cleaned = normalize_space(entry)
        if len(cleaned) < 20:
            continue
        if cleaned.lower().startswith(NARRATIVE_REF_PREFIXES):
            continue

        has_signal = bool(
            YEAR_RE.search(cleaned)
            or re.search(r"https?://|doi:\s*|10\.\d{4,9}/|arxiv:", cleaned, flags=re.IGNORECASE)
        )
        if not has_signal:
            continue

        if len(cleaned) > 2200:
            pieces = _split_after_year_boundary(cleaned)
            for piece in pieces:
                normalized_piece = normalize_space(piece)
                if normalized_piece and len(normalized_piece) >= 20 and YEAR_RE.search(normalized_piece):
                    filtered.append(normalized_piece)
            continue

        filtered.append(cleaned)
    return filtered


def _split_text_chunks_for_model(text: str, max_chars: int, overlap_lines: int = 8) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunk_text = "\n".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            tail = current[-overlap_lines:] if overlap_lines > 0 else []
            current = list(tail)
            current_len = sum(len(value) + 1 for value in current)

        current.append(line)
        current_len += line_len

    final_text = "\n".join(current).strip()
    if final_text:
        chunks.append(final_text)
    return chunks


def _split_after_year_boundary(text: str) -> list[str]:
    # Python 3.8-compatible boundary split:
    # 1) "<year>. <Surname, A. ...>" (BibTeX-like)
    # 2) "<year>. <Firstname Lastname ...>" (plain author list)
    boundary_patterns = [
        re.compile(r"\b(?:19|20)\d{2}[a-z]?\.\s+(?=[A-Z][A-Za-z'`\-]+,\s)"),
        re.compile(
            r"\b(?:19|20)\d{2}[a-z]?\.\s+(?=(?:\[\d{1,3}\]\s*)?"
            r"[A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+){1,4}(?:\s*,|\s+and\s+|\s+[A-Z]))"
        ),
    ]
    matches = []
    for pattern in boundary_patterns:
        matches.extend(pattern.finditer(text))
    matches = sorted(matches, key=lambda m: m.start())
    # Deduplicate overlapping matches at the same location.
    unique_matches = []
    seen_starts: set[int] = set()
    for match in matches:
        if match.start() in seen_starts:
            continue
        seen_starts.add(match.start())
        unique_matches.append(match)
    matches = unique_matches
    if not matches:
        return [text]

    pieces: list[str] = []
    start = 0
    for match in matches:
        end = match.end()
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end

    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _build_bedrock_client(region: str, bearer_token: str | None = None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed. Install boto3 to enable model entry extraction.") from exc

    if bearer_token:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = str(bearer_token)

    return boto3.client(service_name="bedrock-runtime", region_name=region)


def _extract_text_from_converse_response(response: dict[str, Any]) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    parts = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
    return "\n".join(parts).strip()


def _parse_json_from_text(text: str) -> Any:
    candidates: list[str] = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(fenced)

    bracket_match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if bracket_match:
        candidates.append(bracket_match.group(1).strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Failed to parse JSON from entry extraction response.")


def _append_raw_entries_from_json(parsed: Any, target: list[str]) -> None:
    if isinstance(parsed, dict):
        refs_obj = parsed.get("references", [])
    elif isinstance(parsed, list):
        refs_obj = parsed
    else:
        refs_obj = []

    for item in refs_obj:
        if isinstance(item, dict):
            raw = item.get("raw_reference")
        else:
            raw = item
        if raw is None:
            continue
        normalized = _strip_reference_artifacts(str(raw), preserve_line_breaks=False)
        normalized = re.sub(r"\s+\d+(?:\s*,\s*\d+){1,8}\s*$", "", normalized).strip()
        if normalized:
            target.append(normalized)


def _finalize_raw_entries(raw_entries: list[str]) -> list[str]:
    expanded: list[str] = []
    for entry in raw_entries:
        cleaned = _strip_reference_artifacts(entry, preserve_line_breaks=True)
        if not cleaned:
            continue

        # If OCR leaked box markers or merged many references into one huge string,
        # try a second-pass split before dedupe/filter.
        marker_leak = bool(re.search(r"(?i)\btext\s*\[\[|\[\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\]", entry))
        many_years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", cleaned)) >= 3
        if marker_leak or (many_years and len(cleaned) > 900):
            split_entries, _ = split_reference_entries(cleaned)
            if split_entries:
                for split_entry in split_entries:
                    flat_split = _strip_reference_artifacts(split_entry, preserve_line_breaks=False)
                    split_years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", flat_split))
                    if split_years >= 3 and len(flat_split) > 500:
                        finer = [normalize_space(piece) for piece in _split_after_year_boundary(flat_split) if normalize_space(piece)]
                        if len(finer) > 1:
                            expanded.extend(finer)
                            continue
                    expanded.append(split_entry)
                continue
            if many_years:
                flat_cleaned = _strip_reference_artifacts(cleaned, preserve_line_breaks=False)
                finer = [normalize_space(piece) for piece in _split_after_year_boundary(flat_cleaned) if normalize_space(piece)]
                if len(finer) > 1:
                    expanded.extend(finer)
                    continue
        expanded.append(cleaned)

    expanded = _merge_fragmented_entries(expanded)

    deduped: list[str] = []
    seen: set[str] = set()
    for entry in expanded:
        normalized_entry = _strip_reference_artifacts(entry, preserve_line_breaks=False)
        key = re.sub(r"\s+", " ", normalized_entry).strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(normalized_entry)

    filtered = _filter_reference_like_entries(deduped)
    return filtered if filtered else deduped


def _extract_entries_with_bedrock(
    ref_block: str,
    model_id: str,
    region: str,
    chunk_chars: int,
    bearer_token: str | None = None,
) -> list[str]:
    client = _build_bedrock_client(region=region, bearer_token=bearer_token)
    cleaned = ref_block
    cleaned = re.sub(r"(?m)^\s*\d+\s+BibAgent:.*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*BibAgent:.*$", "", cleaned)
    cleaned = re.sub(r"\u00ad", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    chunks = _split_text_chunks_for_model(cleaned, max_chars=max(2000, chunk_chars))
    raw_entries: list[str] = []

    n_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"      [segment {idx}/{n_chunks}] extracting references via bedrock LLM...", end="\r", flush=True)
        prompt = (
            "You extract bibliography entries from noisy PDF text.\n"
            "Keep only reference list entries and preserve original order.\n"
            "Return ONLY valid JSON in the exact shape:\n"
            '{"references":[{"raw_reference":"..."}]}\n\n'
            f"Chunk {idx}/{n_chunks}:\n{chunk}"
        )
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0},
        )
        text = _extract_text_from_converse_response(response)
        if not text:
            continue

        parsed = _parse_json_from_text(text)
        _append_raw_entries_from_json(parsed, raw_entries)
    if n_chunks:
        print(f"      [segment] bedrock extraction done ({n_chunks} chunks → {len(raw_entries)} entries)       ", flush=True)

    return _finalize_raw_entries(raw_entries)


def _load_local_model_bundle(model_path: str) -> tuple[Any, Any]:
    model_path = str(Path(model_path).expanduser().resolve())
    cached = _LOCAL_MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is not installed. Install torch to enable local model extraction.") from exc

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is not installed. Install transformers to enable local model extraction.") from exc

    if not Path(model_path).exists():
        raise RuntimeError(f"Local model path does not exist: {model_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("Local DeepSeek-OCR-2 extraction currently requires CUDA.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    load_kwargs = {"trust_remote_code": True, "use_safetensors": True}
    try:
        model = AutoModel.from_pretrained(model_path, _attn_implementation="flash_attention_2", **load_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(model_path, **load_kwargs)

    model = model.eval().cuda()
    try:
        model = model.to(torch.bfloat16)
    except Exception:
        pass

    bundle = (tokenizer, model)
    _LOCAL_MODEL_CACHE[model_path] = bundle
    return bundle


def _load_local_vllm_bundle(model_path: str) -> tuple[Any, Any, Any, Any, Any]:
    model_path = str(Path(model_path).expanduser().resolve())
    if not Path(model_path).exists():
        raise RuntimeError(f"Local model path does not exist: {model_path}")

    code_dir = _resolve_deepseek_ocr2_vllm_code_dir(model_path)
    cache_key = f"{model_path}::{code_dir}"
    cached = _LOCAL_VLLM_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is not installed. Install torch to enable local vLLM extraction.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Local DeepSeek-OCR-2 vLLM extraction currently requires CUDA.")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is not installed. Install transformers to enable local vLLM extraction.") from exc

    try:
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models.registry import ModelRegistry
    except ImportError as exc:
        raise RuntimeError("vllm is not installed. Install vllm to enable local vLLM extraction.") from exc

    os.environ.setdefault("VLLM_USE_V1", "0")
    if getattr(torch.version, "cuda", "") == "11.8":
        os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-11.8/bin/ptxas")

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config_module = types.ModuleType("config")
    config_module.BASE_SIZE = 1024
    config_module.IMAGE_SIZE = 768
    config_module.CROP_MODE = True
    config_module.MIN_CROPS = 2
    config_module.MAX_CROPS = 6
    config_module.PRINT_NUM_VIS_TOKENS = False
    config_module.PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
    config_module.MODEL_PATH = model_path
    config_module.INPUT_PATH = ""
    config_module.OUTPUT_PATH = ""
    config_module.MAX_CONCURRENCY = 32
    config_module.NUM_WORKERS = 8
    config_module.SKIP_REPEAT = True
    config_module.TOKENIZER = tokenizer
    sys.modules["config"] = config_module

    deepseek_ocr2_module = importlib.import_module("deepseek_ocr2")
    image_process_module = importlib.import_module("process.image_process")
    ngram_module = importlib.import_module("process.ngram_norepeat")

    try:
        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", deepseek_ocr2_module.DeepseekOCR2ForCausalLM)
    except Exception:
        pass

    max_num_seqs = max(1, int(os.getenv("CITATION_CHECKER_LOCAL_OCR_VLLM_MAX_NUM_SEQS", "32")))
    tensor_parallel_size = max(1, int(os.getenv("CITATION_CHECKER_LOCAL_OCR_VLLM_TP_SIZE", "1")))
    max_tokens = max(256, int(os.getenv("CITATION_CHECKER_LOCAL_OCR_VLLM_MAX_TOKENS", "8192")))
    gpu_memory_utilization = float(os.getenv("CITATION_CHECKER_LOCAL_OCR_VLLM_GPU_MEMORY_UTILIZATION", "0.8"))
    gpu_memory_utilization = min(0.95, max(0.1, gpu_memory_utilization))

    llm_kwargs = {
        "model": model_path,
        "hf_overrides": {"architectures": ["DeepseekOCR2ForCausalLM"]},
        "block_size": 256,
        "enforce_eager": False,
        "trust_remote_code": True,
        "max_model_len": 8192,
        "swap_space": 0,
        "max_num_seqs": max_num_seqs,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    try:
        llm = LLM(disable_mm_preprocessor_cache=True, **llm_kwargs)
    except TypeError:
        llm = LLM(**llm_kwargs)

    logits_processors = []
    try:
        logits_processors = [
            ngram_module.NoRepeatNGramLogitsProcessor(
                ngram_size=20,
                window_size=50,
                whitelist_token_ids={128821, 128822},
            )
        ]
    except Exception:
        logits_processors = []

    sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
    }
    if logits_processors:
        sampling_kwargs["logits_processors"] = logits_processors
    sampling_params = SamplingParams(**sampling_kwargs)

    processor = image_process_module.DeepseekOCR2Processor()
    bundle = (llm, sampling_params, processor, image_process_module, config_module)
    _LOCAL_VLLM_MODEL_CACHE[cache_key] = bundle
    return bundle


def _render_pdf_reference_pages(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
    output_dir: Path,
    dpi: int = 200,
) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("pymupdf is not installed. Install pymupdf to enable local OCR from PDF pages.") from exc

    pdf = Path(pdf_path)
    if not pdf.exists():
        raise RuntimeError(f"PDF path does not exist: {pdf}")

    doc = fitz.open(str(pdf))
    try:
        if len(doc) == 0:
            return []

        first = max(0, start_page)
        last = min(end_page, len(doc) - 1)
        if last < first:
            return []

        scale = max(1.0, float(dpi) / 72.0)
        matrix = fitz.Matrix(scale, scale)
        image_paths: list[Path] = []
        for page_idx in range(first, last + 1):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            page_path = output_dir / f"reference_page_{page_idx + 1}.png"
            pix.save(str(page_path))
            image_paths.append(page_path)
        return image_paths
    finally:
        doc.close()


def _extract_entries_from_markdown_pages(markdown_pages: list[str], output_dir: Path) -> list[str]:
    if not markdown_pages:
        return []

    combined_markdown = "\n\n".join(markdown_pages).strip()
    _safe_write_text(output_dir / "reference_pages_markdown.mmd", combined_markdown)

    tagged_entries = _extract_reference_entries_from_tagged_markdown(combined_markdown)
    tagged_entries = _finalize_raw_entries(tagged_entries)
    if tagged_entries:
        _safe_write_text(
            output_dir / "reference_entries_from_markdown.json",
            json.dumps({"references": [{"raw_reference": value} for value in tagged_entries]}, indent=2, ensure_ascii=False),
        )
        return tagged_entries

    cleaned_markdown = combined_markdown
    cleaned_markdown = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned_markdown)
    cleaned_markdown = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 \2", cleaned_markdown)
    cleaned_markdown = re.sub(r"`{1,3}", "", cleaned_markdown)
    cleaned_markdown = re.sub(r"<\|[^|]+\|>", " ", cleaned_markdown)
    cleaned_markdown = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned_markdown)
    cleaned_markdown = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned_markdown)
    cleaned_markdown = re.sub(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", "", cleaned_markdown)
    cleaned_markdown = re.sub(r"\r\n|\r", "\n", cleaned_markdown)

    entries, _ = split_reference_entries(cleaned_markdown)
    entries = _finalize_raw_entries(entries)
    if not entries:
        paragraphs = [normalize_space(chunk) for chunk in re.split(r"\n\s*\n", cleaned_markdown) if normalize_space(chunk)]
        entries = _finalize_raw_entries(paragraphs)

    _safe_write_text(
        output_dir / "reference_entries_from_markdown.json",
        json.dumps({"references": [{"raw_reference": value} for value in entries]}, indent=2, ensure_ascii=False),
    )
    return entries


def _run_local_vllm_ocr_on_images(
    model_path: str,
    image_paths: list[Path],
    prompt: str,
) -> list[str]:
    from PIL import Image

    llm, sampling_params, processor, image_process_module, config_module = _load_local_vllm_bundle(model_path)
    image_process_module.PROMPT = prompt
    config_module.PROMPT = prompt

    batch_inputs: list[dict[str, Any]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            multimodal = processor.tokenize_with_images(images=[image_rgb], bos=True, eos=True, cropping=True)
            batch_inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": multimodal},
                }
            )

    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    texts: list[str] = []
    for output in outputs:
        if output.outputs:
            texts.append(str(output.outputs[0].text or ""))
        else:
            texts.append("")
    return texts


def _extract_entries_with_local_model_from_images(
    tokenizer: Any,
    model: Any,
    image_paths: list[Path],
    output_dir: Path,
    debug: bool = False,
    debug_raise: bool = False,
    backend: str = "hf",
    model_path: str | None = None,
) -> list[str]:
    backend = _normalize_local_inference_backend(backend)
    markdown_pages: list[str] = []

    markdown_prompt = "<image>\n<|grounding|>Convert the document to markdown."
    fallback_prompt = "<image>\nFree OCR."

    if backend == "vllm":
        if not model_path:
            raise RuntimeError("vLLM backend requires local model_path.")

        markdown_outputs = _run_local_vllm_ocr_on_images(
            model_path=model_path,
            image_paths=image_paths,
            prompt=markdown_prompt,
        )
        page_texts = [str(value or "").strip() for value in markdown_outputs]
        empty_indexes = [idx for idx, text in enumerate(page_texts) if not text]

        if empty_indexes:
            fallback_outputs = _run_local_vllm_ocr_on_images(
                model_path=model_path,
                image_paths=[image_paths[idx] for idx in empty_indexes],
                prompt=fallback_prompt,
            )
            for pos, page_idx in enumerate(empty_indexes):
                page_texts[page_idx] = str(fallback_outputs[pos] or "").strip()

        for idx, image_path in enumerate(image_paths, start=1):
            page_output_dir = output_dir / f"page_{idx}"
            page_output_dir.mkdir(parents=True, exist_ok=True)

            _safe_write_text(page_output_dir / "prompt_markdown.txt", markdown_prompt)
            _safe_write_text(page_output_dir / "prompt_free_ocr.txt", fallback_prompt)
            _safe_write_text(page_output_dir / "image_path.txt", str(image_path))

            text_str = page_texts[idx - 1]
            _safe_write_text(page_output_dir / "raw_markdown_output.txt", text_str)
            _safe_write_text(page_output_dir / "raw_markdown_output_repr.txt", repr(text_str))
            if not text_str:
                _safe_write_text(page_output_dir / "empty_markdown_output.flag", "vllm markdown output is empty")
            else:
                _safe_write_text(page_output_dir / "probe_free_ocr.txt", text_str)
                markdown_pages.append(f"[[PAGE {idx}]]\n{text_str}")
            _safe_write_text(page_output_dir / "page_reference_markdown.mmd", text_str)

        return _extract_entries_from_markdown_pages(markdown_pages, output_dir=output_dir)

    for idx, image_path in enumerate(image_paths, start=1):
        page_output_dir = output_dir / f"page_{idx}"
        page_output_dir.mkdir(parents=True, exist_ok=True)

        _safe_write_text(page_output_dir / "prompt_markdown.txt", markdown_prompt)
        _safe_write_text(page_output_dir / "prompt_free_ocr.txt", fallback_prompt)
        _safe_write_text(page_output_dir / "image_path.txt", str(image_path))

        try:
            text = model.infer(
                tokenizer,
                prompt=markdown_prompt,
                image_file=str(image_path),
                output_path=str(page_output_dir),
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=False,
                eval_mode=True,
            )
        except Exception:
            if debug:
                _safe_write_text(page_output_dir / "debug_infer_exception.txt", traceback.format_exc())
            if debug_raise:
                raise
            continue

        text_str = str(text) if text is not None else ""
        _safe_write_text(page_output_dir / "raw_markdown_output.txt", text_str)
        _safe_write_text(page_output_dir / "raw_markdown_output_repr.txt", repr(text))

        if not text_str.strip():
            _safe_write_text(page_output_dir / "empty_markdown_output.flag", "model.infer returned empty text")
            try:
                probe_text = model.infer(
                    tokenizer,
                    prompt=fallback_prompt,
                    image_file=str(image_path),
                    output_path=str(page_output_dir),
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    save_results=False,
                    eval_mode=True,
                )
                text_str = str(probe_text or "").strip()
                _safe_write_text(page_output_dir / "probe_free_ocr.txt", text_str)
            except Exception:
                _safe_write_text(page_output_dir / "probe_free_ocr_exception.txt", traceback.format_exc())
        if text_str.strip():
            markdown_pages.append(f"[[PAGE {idx}]]\n{text_str.strip()}")
        _safe_write_text(page_output_dir / "page_reference_markdown.mmd", text_str.strip())

    return _extract_entries_from_markdown_pages(markdown_pages, output_dir=output_dir)


def _extract_entries_with_local_model_from_text(
    ref_block: str,
    model_path: str,
    chunk_chars: int,
    output_dir: Path | None = None,
    backend: str = "hf",
) -> list[str]:
    backend = _normalize_local_inference_backend(backend)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is not installed. Install pillow to enable local model extraction.") from exc

    tokenizer = None
    model = None
    if backend == "hf":
        tokenizer, model = _load_local_model_bundle(model_path)
    elif backend == "vllm":
        _load_local_vllm_bundle(model_path)
    else:
        raise RuntimeError(f"Unsupported local inference backend: {backend}")

    cleaned = ref_block
    cleaned = re.sub(r"\u00ad", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    chunks = _split_text_chunks_for_model(cleaned, max_chars=max(2000, chunk_chars))
    raw_entries: list[str] = []

    if output_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="pdf_checker_local_model_")
        temp_dir = Path(temp_ctx.__enter__())
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
        temp_dir = output_dir
        _safe_write_text(output_dir / "reference_text_block.txt", cleaned)
        _safe_write_text(output_dir / "reference_chunks_count.txt", str(len(chunks)))

    try:
        dummy_image = temp_dir / "dummy.png"
        Image.new("RGB", (32, 32), color="white").save(dummy_image)

        for idx, chunk in enumerate(chunks, start=1):
            prompt = (
                "<image>\n"
                "You extract bibliography entries from noisy PDF text.\n"
                "Keep only reference list entries and preserve original order.\n"
                "Return ONLY valid JSON in the exact shape:\n"
                '{"references":[{"raw_reference":"..."}]}\n\n'
                f"Chunk {idx}/{len(chunks)}:\n{chunk}"
            )
            if output_dir is not None:
                _safe_write_text(output_dir / f"chunk_{idx:03d}_prompt.txt", prompt)
            if backend == "hf":
                text = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(dummy_image),
                    output_path=str(temp_dir),
                    # DeepSeek-OCR-2 only supports query sizes mapped from 768/1024 inputs.
                    # Using 512 can trigger `param_img` unbound errors in deepencoderv2.
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    save_results=False,
                    eval_mode=True,
                )
            else:
                text = _run_local_vllm_ocr_on_images(
                    model_path=model_path,
                    image_paths=[dummy_image],
                    prompt=prompt,
                )[0]
            if not text:
                if output_dir is not None:
                    _safe_write_text(output_dir / f"chunk_{idx:03d}_raw_output.txt", "")
                continue

            raw_text = str(text)
            if output_dir is not None:
                _safe_write_text(output_dir / f"chunk_{idx:03d}_raw_output.txt", raw_text)
            parsed = _parse_json_from_text(raw_text)
            if output_dir is not None and parsed:
                _safe_write_text(
                    output_dir / f"chunk_{idx:03d}_parsed_json.json",
                    json.dumps(parsed, indent=2, ensure_ascii=False),
                )
            _append_raw_entries_from_json(parsed, raw_entries)
    finally:
        if temp_ctx is not None:
            temp_ctx.__exit__(None, None, None)

    return _finalize_raw_entries(raw_entries)


def _extract_entries_with_local_model(
    ref_block: str,
    model_path: str,
    chunk_chars: int,
    source_pdf_path: str | Path | None = None,
    reference_page_start: int | None = None,
    reference_page_end: int | None = None,
    local_inference_backend: str = "hf",
) -> tuple[list[str], str, str | None]:
    backend = _normalize_local_inference_backend(local_inference_backend)
    tokenizer = None
    model = None
    if backend == "hf":
        tokenizer, model = _load_local_model_bundle(model_path)
    elif backend == "vllm":
        _load_local_vllm_bundle(model_path)
    else:
        raise RuntimeError(f"Unsupported local inference backend: {backend}")

    debug = _env_flag("CITATION_CHECKER_LOCAL_OCR_DEBUG")
    debug_raise = _env_flag("CITATION_CHECKER_LOCAL_OCR_DEBUG_RAISE")

    local_run_dir: Path | None = None
    if (
        source_pdf_path is not None
        and reference_page_start is not None
        and reference_page_end is not None
    ):
        try:
            tmp_root = _local_ocr_tmp_root()
            tmp_root.mkdir(parents=True, exist_ok=True)
            pdf_stem = Path(source_pdf_path).stem
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_stem).strip("_") or "paper"
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = tmp_root / f"{safe_stem}_p{reference_page_start + 1}-{reference_page_end + 1}_{run_id}"
            local_run_dir = run_dir
            image_dir = run_dir / "images"
            model_output_dir = run_dir / "model_outputs"
            image_dir.mkdir(parents=True, exist_ok=True)
            model_output_dir.mkdir(parents=True, exist_ok=True)

            page_images = _render_pdf_reference_pages(
                pdf_path=source_pdf_path,
                start_page=reference_page_start,
                end_page=reference_page_end,
                output_dir=image_dir,
            )
            if page_images:
                entries = _extract_entries_with_local_model_from_images(
                    tokenizer=tokenizer,
                    model=model,
                    image_paths=page_images,
                    output_dir=model_output_dir,
                    debug=debug,
                    debug_raise=debug_raise,
                    backend=backend,
                    model_path=model_path,
                )
                if entries:
                    return entries, "reference_page_images_markdown", str(run_dir)
        except Exception:
            if debug_raise:
                raise
            pass

    entries = _extract_entries_with_local_model_from_text(
        ref_block=ref_block,
        model_path=model_path,
        chunk_chars=chunk_chars,
        output_dir=(local_run_dir / "text_fallback") if local_run_dir is not None else None,
        backend=backend,
    )
    return entries, "reference_text_block", (str(local_run_dir) if local_run_dir is not None else None)


def split_reference_entries(ref_block: str) -> tuple[list[str], bool]:
    text = ref_block
    text = re.sub(r"\[\[PAGE \d+\]\]", "\n", text)
    text = re.sub(r"(?m)^\s*\d+\s+BibAgent:.*$", "", text)
    text = re.sub(r"(?m)^\s*BibAgent:.*$", "", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    numbered_mode = len(NUMBERED_REF_RE.findall(text)) >= 3
    if numbered_mode:
        chunks = re.split(r"(?m)(?=^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.)\s+)", text)
        entries = [normalize_space(ENTRY_PREFIX_RE.sub("", chunk)) for chunk in chunks if normalize_space(chunk)]
        entries = [entry for entry in entries if len(entry) >= 20]
        return entries, True

    unnumbered = _split_unnumbered_references(text)
    if len(unnumbered) >= 3:
        return _filter_reference_like_entries(unnumbered), False

    paragraphs = [normalize_space(chunk) for chunk in re.split(r"\n\s*\n", text) if normalize_space(chunk)]
    if len(paragraphs) >= 3:
        split_paragraphs: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) > 1000:
                pieces = _split_after_year_boundary(paragraph)
                split_paragraphs.extend([normalize_space(piece) for piece in pieces if normalize_space(piece)])
            else:
                split_paragraphs.append(paragraph)
        return _filter_reference_like_entries(split_paragraphs), False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entries: list[str] = []
    current: list[str] = []
    for line in lines:
        starts_new = bool(
            re.match(r"^(?:\[\d+\]|\(\d+\)|\d+\.)\s+", line)
            or re.match(r"^[A-Z][^.!?]{1,120}\(\d{4}[a-z]?\)", line)
            or re.match(r"^[A-Z][A-Za-z'`.\-]+,\s+(?:[A-Z]\.\s*){1,4}", line)
        )
        if starts_new and current:
            entries.append(normalize_space(" ".join(current)))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(normalize_space(" ".join(current)))
    return _filter_reference_like_entries(entries), False


def _estimate_quality(entries: list[str], numbered_mode: bool, heading_detected: bool) -> ExtractionQuality:
    if not entries:
        return ExtractionQuality.LOW
    if numbered_mode and len(entries) >= 8:
        return ExtractionQuality.HIGH
    if heading_detected and len(entries) >= 6:
        return ExtractionQuality.MEDIUM
    if len(entries) >= 4:
        return ExtractionQuality.MEDIUM
    return ExtractionQuality.LOW


def segment_references_from_pages(
    page_texts: list[str],
    entry_extraction: str = "heuristic",
    model_provider: str = "bedrock",
    model_id: str | None = None,
    region: str = "us-east-1",
    entry_chunk_chars: int = 12000,
    bearer_token: str | None = None,
    local_model_path: str | None = None,
    local_inference_backend: str = "hf",
    source_pdf_path: str | Path | None = None,
) -> tuple[list[str], ExtractionQuality, dict[str, Any]]:
    mode = entry_extraction.strip().lower() if entry_extraction else "heuristic"
    if mode not in {"heuristic", "model"}:
        mode = "heuristic"
    provider = model_provider.strip().lower() if model_provider else "bedrock"
    if provider not in {"bedrock", "local"}:
        provider = "bedrock"

    start_page, end_page, heading_on_pages = find_reference_page_range(page_texts)
    refs_text = build_page_range_text(page_texts, start_page, end_page)
    ref_block, heading_in_block = find_reference_block(refs_text)
    heading_detected = heading_on_pages or heading_in_block

    metadata: dict[str, Any] = {
        "reference_page_start": start_page + 1,
        "reference_page_end": end_page + 1,
        "reference_heading_detected": heading_detected,
        "entry_extraction_mode": "heuristic",
        "entry_extraction_provider": "heuristic",
        "entry_extraction_input": "reference_text_block",
        "entry_extraction_fallback": False,
    }

    allow_local_image_ocr = (
        mode == "model"
        and provider == "local"
        and bool(local_model_path)
        and source_pdf_path is not None
    )
    if not ref_block.strip() and not allow_local_image_ocr:
        return [], ExtractionQuality.LOW, metadata

    numbered_mode = False
    entries: list[str] = []
    if mode == "model":
        try:
            if provider == "local" and local_model_path:
                entries, input_mode, local_ocr_run_dir = _extract_entries_with_local_model(
                    ref_block=ref_block,
                    model_path=local_model_path,
                    chunk_chars=entry_chunk_chars,
                    source_pdf_path=source_pdf_path,
                    reference_page_start=start_page,
                    reference_page_end=end_page,
                    local_inference_backend=local_inference_backend,
                )
                metadata["entry_extraction_mode"] = "model"
                metadata["entry_extraction_provider"] = "local"
                metadata["entry_extraction_local_backend"] = _normalize_local_inference_backend(local_inference_backend)
                metadata["entry_extraction_input"] = input_mode
                if local_ocr_run_dir:
                    metadata["ocr_intermediate_dir"] = local_ocr_run_dir
                numbered_mode = len(NUMBERED_REF_RE.findall(ref_block)) >= 3
            elif provider == "bedrock" and model_id:
                entries = _extract_entries_with_bedrock(
                    ref_block=ref_block,
                    model_id=model_id,
                    region=region,
                    chunk_chars=entry_chunk_chars,
                    bearer_token=bearer_token,
                )
                metadata["entry_extraction_mode"] = "model"
                metadata["entry_extraction_provider"] = "bedrock"
                metadata["entry_extraction_input"] = "reference_text_block"
                numbered_mode = len(NUMBERED_REF_RE.findall(ref_block)) >= 3
        except Exception as exc:
            metadata["entry_extraction_fallback"] = True
            metadata["entry_extraction_error"] = f"{type(exc).__name__}: {exc}"

    if not entries:
        entries, numbered_mode = split_reference_entries(ref_block)
        if mode == "model":
            metadata["entry_extraction_fallback"] = True
            metadata["entry_extraction_provider"] = f"{provider}->heuristic"
        metadata["entry_extraction_mode"] = "heuristic" if mode != "model" else "model->heuristic"

    quality = _estimate_quality(entries, numbered_mode=numbered_mode, heading_detected=heading_detected)
    return entries, quality, metadata


def segment_references(document_text: str) -> tuple[list[str], ExtractionQuality]:
    block, heading_detected = find_reference_block(document_text)
    if not block.strip():
        return [], ExtractionQuality.LOW

    entries, numbered_mode = split_reference_entries(block)
    return entries, _estimate_quality(entries, numbered_mode=numbered_mode, heading_detected=heading_detected)
