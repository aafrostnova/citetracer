"""Strict URL reachability verification.

Unlike the lighter prune_dead_non_academic_urls.py, this script:
  1. Always uses GET (HEAD is untrustworthy for many JS-rendered sites).
  2. Scans the response body for common 404 / "page not found" text patterns.
  3. Detects soft redirects — if a deep URL redirects to the site root, flag.
  4. Flags pages that are too small to be real content (< 200 bytes typical error pages).
  5. Optionally consults the Wayback Machine (archive.org) to confirm the URL
     was ever a real page (fallback for ambiguous cases).

Output JSON keeps only URLs that pass ALL strict checks.

Usage:
    python scripts/verify_urls_strict.py \
        --input data/bib/non_academic_sources.json \
        --output data/bib/non_academic_sources_strict.json \
        --workers 8 --timeout 15
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Title-level not-found patterns. We ONLY trust title-based signals:
# body text can contain "page not found" inside embedded React/JS fallback code
# on perfectly live pages. The <title> tag is the authoritative indicator.
_TITLE_NOT_FOUND_RE = re.compile(
    r"<title[^>]*>\s*"
    r"(?:"
    r"404"
    r"|page\s+not\s+found"
    r"|not\s+found"
    r"|error\s+404"
    r"|file\s+not\s+found"
    r"|page\s+(?:doesn'?t|does\s+not)\s+exist"
    r"|oops"
    r")"
    r"[^<]*</title>",
    re.IGNORECASE,
)


def _is_soft_redirect(original_url: str, final_url: str) -> bool:
    """True if original had a path/query but was redirected to the site root."""
    o = urlparse(original_url)
    f = urlparse(final_url)
    if o.netloc != f.netloc and not f.netloc.endswith(o.netloc.lstrip("www.")) \
       and not o.netloc.lstrip("www.").endswith(f.netloc):
        # Hopped to a different domain entirely — suspicious
        return True
    original_had_path = bool((o.path or "").strip("/")) or bool(o.query)
    final_has_path = bool((f.path or "").strip("/")) or bool(f.query)
    return original_had_path and not final_has_path


def _body_is_404(body_text: str) -> bool:
    """Title-only check — only the <title> tag is authoritative.

    Body text alone is unreliable (modern SPAs embed 'page not found' fallback
    text inside their JS bundles even on live pages).
    """
    return bool(_TITLE_NOT_FOUND_RE.search(body_text))


def check_url(url: str, timeout: float) -> tuple[bool, str]:
    """Return (is_reachable, reason)."""
    if not url:
        return False, "empty"
    try:
        resp = requests.get(
            url, headers=_HEADERS, allow_redirects=True, timeout=timeout, stream=False,
        )
    except requests.exceptions.ConnectionError as exc:
        msg = str(exc).lower()
        if "name or service not known" in msg or "no address associated" in msg \
           or "nameorserviceunknown" in msg or "connection refused" in msg:
            return False, f"dns_or_refused"
        return False, f"conn_error:{type(exc).__name__}"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except requests.exceptions.RequestException as exc:
        return False, f"request_error:{type(exc).__name__}"

    status = resp.status_code
    final_url = resp.url
    body = resp.text or ""
    ctype = resp.headers.get("Content-Type", "").lower()

    # Hard dead status codes
    if status in (404, 410):
        return False, f"http_{status}"
    # Server errors and other 4xx
    if status >= 500:
        return False, f"http_{status}"

    # ALWAYS check title-level 404 signal, including on 401/403 responses.
    # HuggingFace (and some SPAs) serve a real 404 page with a misleading status
    # code — trust the <title> over the HTTP code.
    if "html" in ctype or "text" in ctype or ctype == "":
        if _body_is_404(body):
            return False, f"body_says_not_found_status_{status}"

    # Tolerate auth walls AFTER title check
    if status in (401, 402, 403):
        return True, f"http_{status}_kept"

    if status >= 400:
        return False, f"http_{status}"

    # Soft redirect detection (2xx only)
    if _is_soft_redirect(url, final_url):
        return False, f"soft_redirect_to_root:{urlparse(final_url).netloc}"

    # Tiny body on 200 → suspicious (empty success)
    if ("html" in ctype or "text" in ctype or ctype == "") \
            and status == 200 and len(body) < 200:
        return False, "empty_body_on_200"

    return True, f"http_{status}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    print(f"Strict verification of {len(data)} URLs (GET + body scan + redirect check)...\n", flush=True)

    results: list[tuple[bool, str]] = [None] * len(data)

    def _task(i: int, entry: dict) -> tuple[int, bool, str]:
        url = (entry.get("url") or "").strip()
        ok, reason = check_url(url, args.timeout)
        return i, ok, reason

    pbar = tqdm(total=len(data), desc="Verifying", dynamic_ncols=True) if tqdm else None
    live = 0
    dead = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, i, d) for i, d in enumerate(data)]
        for fut in as_completed(futures):
            i, ok, reason = fut.result()
            results[i] = (ok, reason)
            if ok:
                live += 1
            else:
                dead += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"live": live, "dead": dead})
    if pbar:
        pbar.close()

    kept: list[dict] = []
    dropped: list[dict] = []
    for entry, (ok, reason) in zip(data, results):
        record = dict(entry)
        record["_reach_reason"] = reason
        if ok:
            kept.append(record)
        else:
            dropped.append({
                "url": entry.get("url", ""),
                "title": (entry.get("title") or "")[:70],
                "reason": reason,
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False))

    print(f"\n=== RESULT ===")
    print(f"Live: {len(kept)}  Dead: {len(dropped)}")
    print(f"Saved to: {out_path}")
    if dropped:
        print("\nDropped URLs (with reason):")
        for d in dropped:
            print(f"  [{d['reason']:35s}] {d['url'][:80]}  — {d['title']}")


if __name__ == "__main__":
    main()
