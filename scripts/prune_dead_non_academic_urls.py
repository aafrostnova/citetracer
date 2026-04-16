"""Check URLs in non_academic_sources.json and remove dead / 404 ones.

Policy:
  - HEAD first; fall back to GET if HEAD is disallowed (405/400/403) or
    returns a non-200 that GET might resolve (some servers misreport HEAD).
  - "Dead" = 404/410, DNS errors, connection refused, or final status >= 500.
  - Transient codes (429, 503 on HEAD with GET success, timeouts) are KEPT.
  - Redirects are followed; the final URL is what we evaluate.

Usage:
    python scripts/prune_dead_non_academic_urls.py \
        --input data/bib/non_academic_sources.json \
        --output data/bib/non_academic_sources_live.json \
        --workers 16 \
        --timeout 10
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

_DEAD_STATUSES = {404, 410}


def check_url(url: str, timeout: float) -> tuple[bool, int | str]:
    """Return (is_live, status_or_reason).

    Strategy:
      HEAD first (fast). But some servers (e.g. alignmentforum.org) lie on HEAD
      (return 200 with no Content-Type while GET returns 404). So:
        - HEAD 404/410 → dead
        - HEAD 200 WITHOUT Content-Type → suspicious, re-verify with GET
        - HEAD 400/403/405/406/501 or 5xx → re-verify with GET
        - HEAD 200 WITH Content-Type → trust as live
      On GET: 404/410 is the final say, anything else considered live.
    """
    if not url:
        return False, "empty"
    try:
        resp = requests.head(url, headers=_HEADERS, allow_redirects=True, timeout=timeout)
        status = resp.status_code
        if status in _DEAD_STATUSES:
            return False, status
        ctype = resp.headers.get("Content-Type", "").strip()
        needs_get = (
            status in (400, 403, 405, 406, 501)
            or status >= 500
            or (status == 200 and not ctype)  # suspicious HEAD — verify with GET
        )
        if needs_get:
            resp = requests.get(
                url, headers=_HEADERS, allow_redirects=True, timeout=timeout, stream=True,
            )
            try:
                resp.close()
            except Exception:
                pass
            status = resp.status_code
            if status in _DEAD_STATUSES:
                return False, f"head200_get{status}" if status in (404, 410) else status
            return True, status
        return True, status
    except requests.exceptions.ConnectionError as exc:
        msg = str(exc).lower()
        if "nameorserviceunknown" in msg or "name or service not known" in msg or \
           "no address associated" in msg or "connection refused" in msg:
            return False, "dns_or_refused"
        return True, "conn_error_kept"
    except requests.exceptions.Timeout:
        return True, "timeout_kept"
    except requests.exceptions.RequestException as exc:
        return True, f"other_kept:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    print(f"Checking {len(data)} URLs...", flush=True)

    results: list[dict] = [None] * len(data)
    statuses: list[tuple[bool, int | str]] = [None] * len(data)

    def _task(i: int, entry: dict) -> tuple[int, bool, int | str]:
        url = (entry.get("url") or "").strip()
        ok, st = check_url(url, args.timeout)
        return i, ok, st

    pbar = tqdm(total=len(data), desc="Checking URLs", dynamic_ncols=True) if tqdm else None
    live = 0
    dead = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, i, d) for i, d in enumerate(data)]
        for fut in as_completed(futures):
            i, ok, st = fut.result()
            statuses[i] = (ok, st)
            if ok:
                live += 1
            else:
                dead += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"live": live, "dead": dead})

    if pbar:
        pbar.close()

    # Build kept and dropped lists
    kept = []
    dropped = []
    for entry, (ok, st) in zip(data, statuses):
        if ok:
            kept.append(entry)
        else:
            dropped.append({
                "url": entry.get("url", ""),
                "title": entry.get("title", "")[:60],
                "reason": st,
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False))

    # Report
    print(f"\nLive: {len(kept)} / Dead: {len(dropped)}")
    print(f"Saved live to: {out_path}")
    if dropped:
        print("\nDropped URLs:")
        for d in dropped:
            print(f"  [{d['reason']}] {d['url'][:80]}  — {d['title']}")


if __name__ == "__main__":
    main()
