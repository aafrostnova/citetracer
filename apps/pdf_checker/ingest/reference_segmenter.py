from __future__ import annotations

import json
import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import Any

from packages.core.models import ExtractionQuality


START_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[\)\.]?\s*)?(references|bibliography|reference list|参考文献)\s*$",
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

_LOCAL_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def normalize_space(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


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


def _extract_reference_entries_from_tagged_markdown(markdown_text: str) -> list[str]:
    blocks: list[tuple[str, str]] = []
    for match in OCR_TAGGED_BLOCK_RE.finditer(markdown_text):
        label = normalize_space(match.group("label").lower())
        content = match.group("content").strip()
        if not content:
            continue
        blocks.append((label, content))

    in_references = False
    extracted: list[str] = []
    for label, content in blocks:
        heading = _normalize_heading_text(content)

        if label in {"sub_title", "title", "section_title", "figure_title"}:
            if START_HEADING_RE.match(heading):
                in_references = True
                continue
            if in_references and heading:
                break

        if in_references and label == "text":
            extracted.append(normalize_space(content))

    return extracted


def _page_has_heading(page_text: str, heading_re: re.Pattern[str]) -> bool:
    for line in page_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if heading_re.match(line.strip()):
            return True
    return False


def _page_has_appendix_style_heading(page_text: str) -> bool:
    lines = page_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line in lines[:24]:
        stripped = line.strip()
        if APPENDIX_STYLE_HEADING_RE.match(stripped) and not YEAR_RE.search(stripped):
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
        if APPENDIX_STYLE_HEADING_RE.match(stripped) and not YEAR_RE.search(stripped):
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
    # Python 3.8-compatible boundary split: "<year>. <Author, ...>".
    boundary_re = re.compile(r"\b(?:19|20)\d{2}[a-z]?\.\s+(?=[A-Z][A-Za-z'`\-]+,\s)")
    matches = list(boundary_re.finditer(text))
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

    # Support bearer-token-only auth environments for Bedrock runtime calls.
    token = (bearer_token or "").strip()
    has_standard_creds = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
    if token and not has_standard_creds:
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            aws_access_key_id="BEDROCK_TOKEN",
            aws_secret_access_key="BEDROCK_TOKEN",
            aws_session_token=token,
        )

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
        normalized = normalize_space(str(raw))
        normalized = re.sub(r"\[\[PAGE \d+\]\]", "", normalized).strip()
        normalized = re.sub(r"\s+\d+(?:\s*,\s*\d+){1,8}\s*$", "", normalized).strip()
        if normalized:
            target.append(normalized)


def _finalize_raw_entries(raw_entries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in raw_entries:
        key = re.sub(r"\s+", " ", entry).strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(entry)

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

    for idx, chunk in enumerate(chunks, start=1):
        prompt = (
            "You extract bibliography entries from noisy PDF text.\n"
            "Keep only reference list entries and preserve original order.\n"
            "Return ONLY valid JSON in the exact shape:\n"
            '{"references":[{"raw_reference":"..."}]}\n\n'
            f"Chunk {idx}/{len(chunks)}:\n{chunk}"
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


def _extract_entries_with_local_model_from_images(
    tokenizer: Any,
    model: Any,
    image_paths: list[Path],
    output_dir: Path,
    debug: bool = False,
    debug_raise: bool = False,
) -> list[str]:
    markdown_pages: list[str] = []

    markdown_prompt = "<image>\n<|grounding|>Convert the document to markdown."
    fallback_prompt = "<image>\nFree OCR."

    for idx, image_path in enumerate(image_paths, start=1):
        page_output_dir = output_dir / f"page_{idx}"
        page_output_dir.mkdir(parents=True, exist_ok=True)

        if debug:
            _safe_write_text(page_output_dir / "debug_prompt_markdown.txt", markdown_prompt)
            _safe_write_text(page_output_dir / "debug_prompt_free_ocr.txt", fallback_prompt)
            _safe_write_text(page_output_dir / "debug_image_path.txt", str(image_path))

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
        if debug:
            _safe_write_text(page_output_dir / "debug_raw_markdown_output.txt", text_str)
            _safe_write_text(page_output_dir / "debug_raw_markdown_output_repr.txt", repr(text))

        if not text_str.strip():
            if debug:
                _safe_write_text(page_output_dir / "debug_empty_markdown_output.flag", "model.infer returned empty text")
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
                    _safe_write_text(page_output_dir / "debug_probe_free_ocr.txt", text_str)
                except Exception:
                    _safe_write_text(page_output_dir / "debug_probe_free_ocr_exception.txt", traceback.format_exc())
        if text_str.strip():
            markdown_pages.append(f"[[PAGE {idx}]]\n{text_str.strip()}")
            _safe_write_text(page_output_dir / "page_reference_markdown.mmd", text_str.strip())

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


def _extract_entries_with_local_model_from_text(
    ref_block: str,
    model_path: str,
    chunk_chars: int,
) -> list[str]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is not installed. Install pillow to enable local model extraction.") from exc

    tokenizer, model = _load_local_model_bundle(model_path)

    cleaned = ref_block
    cleaned = re.sub(r"\u00ad", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    chunks = _split_text_chunks_for_model(cleaned, max_chars=max(2000, chunk_chars))
    raw_entries: list[str] = []

    with tempfile.TemporaryDirectory(prefix="pdf_checker_local_model_") as tempdir:
        temp_dir = Path(tempdir)
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
            if not text:
                continue

            parsed = _parse_json_from_text(str(text))
            _append_raw_entries_from_json(parsed, raw_entries)

    return _finalize_raw_entries(raw_entries)


def _extract_entries_with_local_model(
    ref_block: str,
    model_path: str,
    chunk_chars: int,
    source_pdf_path: str | Path | None = None,
    reference_page_start: int | None = None,
    reference_page_end: int | None = None,
) -> tuple[list[str], str]:
    tokenizer, model = _load_local_model_bundle(model_path)
    debug = _env_flag("CITATION_CHECKER_LOCAL_OCR_DEBUG")
    debug_raise = _env_flag("CITATION_CHECKER_LOCAL_OCR_DEBUG_RAISE")

    if (
        source_pdf_path is not None
        and reference_page_start is not None
        and reference_page_end is not None
    ):
        try:
            tmp_root = Path.cwd() / ".tmp" / "pdf_checker_local_ocr_pages"
            tmp_root.mkdir(parents=True, exist_ok=True)
            pdf_stem = Path(source_pdf_path).stem
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_stem).strip("_") or "paper"
            run_dir = tmp_root / f"{safe_stem}_p{reference_page_start + 1}-{reference_page_end + 1}"
            image_dir = run_dir / "images"
            model_output_dir = run_dir / "model_outputs"
            image_dir.mkdir(parents=True, exist_ok=True)
            model_output_dir.mkdir(parents=True, exist_ok=True)

            if any(image_dir.iterdir()):
                for stale in image_dir.glob("reference_page_*.png"):
                    try:
                        stale.unlink()
                    except OSError:
                        pass

            if any(model_output_dir.iterdir()):
                for stale in model_output_dir.rglob("*"):
                    if stale.is_file():
                        try:
                            stale.unlink()
                        except OSError:
                            pass

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
                )
                if entries:
                    return entries, "reference_page_images_markdown"
        except Exception:
            if debug_raise:
                raise
            pass

    return _extract_entries_with_local_model_from_text(
        ref_block=ref_block,
        model_path=model_path,
        chunk_chars=chunk_chars,
    ), "reference_text_block"


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
    source_pdf_path: str | Path | None = None,
) -> tuple[list[str], ExtractionQuality, dict[str, Any]]:
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

    if not ref_block.strip():
        return [], ExtractionQuality.LOW, metadata

    mode = entry_extraction.strip().lower() if entry_extraction else "heuristic"
    if mode not in {"heuristic", "model"}:
        mode = "heuristic"
    provider = model_provider.strip().lower() if model_provider else "bedrock"
    if provider not in {"bedrock", "local"}:
        provider = "bedrock"

    numbered_mode = False
    entries: list[str] = []
    if mode == "model":
        try:
            if provider == "local" and local_model_path:
                entries, input_mode = _extract_entries_with_local_model(
                    ref_block=ref_block,
                    model_path=local_model_path,
                    chunk_chars=entry_chunk_chars,
                    source_pdf_path=source_pdf_path,
                    reference_page_start=start_page,
                    reference_page_end=end_page,
                )
                metadata["entry_extraction_mode"] = "model"
                metadata["entry_extraction_provider"] = "local"
                metadata["entry_extraction_input"] = input_mode
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
        except Exception:
            metadata["entry_extraction_fallback"] = True

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
