from __future__ import annotations

import argparse
import gzip
import html.entities
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PUB_TAGS = {
    "article",
    "inproceedings",
    "proceedings",
    "book",
    "incollection",
    "phdthesis",
    "mastersthesis",
    "www",
}


def _open_xml(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _xml_parser_with_entities() -> ET.XMLParser:
    """Build parser with a broad HTML entity map (e.g. &uuml;)."""
    parser = ET.XMLParser()

    # Common HTML/XML named entities.
    for name, codepoint in html.entities.name2codepoint.items():
        parser.entity[name] = chr(codepoint)

    # html5 map contains keys like 'uuml;' and sometimes multi-char strings.
    for key, value in html.entities.html5.items():
        clean = key[:-1] if key.endswith(";") else key
        if clean and clean not in parser.entity:
            parser.entity[clean] = value

    return parser


def _local_tag(tag: str) -> str:
    # Handle namespaced XML tags like "{ns}article".
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _apply_pragmas(conn: sqlite3.Connection, fast_mode: bool) -> None:
    if fast_mode:
        conn.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -200000;
            PRAGMA locking_mode = EXCLUSIVE;
            """
        )
    else:
        conn.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            """
        )


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
          id INTEGER PRIMARY KEY,
          dblp_key TEXT UNIQUE,
          title TEXT,
          year INTEGER,
          venue TEXT,
          url TEXT
        );

        CREATE TABLE IF NOT EXISTS authors (
          id INTEGER PRIMARY KEY,
          pid TEXT,
          name TEXT
        );

        CREATE TABLE IF NOT EXISTS paper_authors (
          paper_id INTEGER,
          author_id INTEGER,
          author_order INTEGER,
          PRIMARY KEY (paper_id, author_id),
          FOREIGN KEY (paper_id) REFERENCES papers(id),
          FOREIGN KEY (author_id) REFERENCES authors(id)
        );
        """
    )


def _create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
        CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_authors_pid_unique ON authors(pid) WHERE pid IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_authors_name ON authors(name);
        CREATE INDEX IF NOT EXISTS idx_pa_author ON paper_authors(author_id);
        """
    )


def text_of(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    if child is None or child.text is None:
        return ""
    return " ".join(child.text.split())


def extract_authors(elem: ET.Element) -> list[tuple[str, str]]:
    people: list[tuple[str, str]] = []
    for tag in ("author", "editor"):
        for c in elem.findall(tag):
            if c.text:
                name = " ".join(c.text.split())
                if name:
                    people.append((str(c.attrib.get("pid", "")).strip(), name))
    return people


def get_author_id(conn: sqlite3.Connection, cache: dict[str, int], pid: str, name: str) -> int:
    cache_key = pid or f"name::{name}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if pid:
        conn.execute("INSERT OR IGNORE INTO authors(pid, name) VALUES (?, ?)", (pid, name))
        row = conn.execute("SELECT id FROM authors WHERE pid = ?", (pid,)).fetchone()
    else:
        row = conn.execute("SELECT id FROM authors WHERE pid IS NULL AND name = ? LIMIT 1", (name,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO authors(pid, name) VALUES (NULL, ?)", (name,))
            row = conn.execute("SELECT id FROM authors WHERE rowid = last_insert_rowid()").fetchone()
    if row is None:
        raise RuntimeError(f"Failed to fetch author id for: pid={pid}, name={name}")

    aid = int(row[0])
    cache[cache_key] = aid
    return aid


def insert_publication(conn: sqlite3.Connection, elem: ET.Element, author_cache: dict[str, int]) -> bool:
    tag = _local_tag(elem.tag)
    if tag not in PUB_TAGS:
        return False

    dblp_key = elem.attrib.get("key", "")
    title = text_of(elem, "title")
    year_text = text_of(elem, "year")
    venue = text_of(elem, "booktitle") or text_of(elem, "journal")
    url = text_of(elem, "ee")

    if not title:
        return False

    try:
        year = int(year_text) if year_text else None
    except ValueError:
        year = None

    cur = conn.execute(
        """
        INSERT OR IGNORE INTO papers(dblp_key, title, year, venue, url)
        VALUES (?, ?, ?, ?, ?)
        """,
        (dblp_key or None, title, year, venue or None, url or None),
    )

    if cur.rowcount == 0:
        row = conn.execute("SELECT id FROM papers WHERE dblp_key = ?", (dblp_key,)).fetchone()
        if row is None:
            # fallback for items without key uniqueness
            row = conn.execute(
                "SELECT id FROM papers WHERE title = ? AND year IS ? LIMIT 1",
                (title, year),
            ).fetchone()
        if row is None:
            return False
        paper_id = int(row[0])
    else:
        paper_id = int(cur.lastrowid)

    authors = extract_authors(elem)
    for order, (pid, name) in enumerate(authors):
        aid = get_author_id(conn, author_cache, pid, name)
        conn.execute(
            """
            INSERT OR IGNORE INTO paper_authors(paper_id, author_id, author_order)
            VALUES (?, ?, ?)
            """,
            (paper_id, aid, order),
        )

    return True


def build_db(
    input_xml: Path,
    output_sqlite: Path,
    commit_every: int,
    max_records: int,
    fast_mode: bool,
    defer_indexes: bool,
) -> None:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_sqlite)
    _apply_pragmas(conn, fast_mode=fast_mode)
    _create_tables(conn)
    if not defer_indexes:
        _create_indexes(conn)

    author_cache: dict[str, int] = {}
    parsed = 0
    inserted = 0

    parser = _xml_parser_with_entities()
    with _open_xml(input_xml) as f:
        for event, elem in ET.iterparse(f, events=("end",), parser=parser):
            if _local_tag(elem.tag) not in PUB_TAGS:
                continue

            parsed += 1
            if insert_publication(conn, elem, author_cache):
                inserted += 1

            if parsed % commit_every == 0:
                conn.commit()
                print(f"parsed={parsed} inserted={inserted}", file=sys.stderr)

            elem.clear()

            if max_records > 0 and parsed >= max_records:
                break

    conn.commit()
    if defer_indexes:
        print("building indexes...", file=sys.stderr)
        _create_indexes(conn)
        conn.commit()
    conn.close()
    print(f"done parsed={parsed} inserted={inserted} db={output_sqlite}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local DBLP sqlite for Ref_Agent")
    parser.add_argument("--input", required=True, help="Path to dblp.xml or dblp.xml.gz")
    parser.add_argument("--output", required=True, help="Path to output sqlite file")
    parser.add_argument("--commit-every", type=int, default=5000, help="Commit interval")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="For quick test. 0 means no limit.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use aggressive SQLite PRAGMA for faster build (less crash safety during build).",
    )
    parser.add_argument(
        "--defer-indexes",
        action="store_true",
        help="Create indexes only after bulk load (usually much faster).",
    )
    args = parser.parse_args()

    build_db(
        input_xml=Path(args.input),
        output_sqlite=Path(args.output),
        commit_every=max(100, args.commit_every),
        max_records=max(0, args.max_records),
        fast_mode=bool(args.fast_mode),
        defer_indexes=bool(args.defer_indexes),
    )


if __name__ == "__main__":
    main()
