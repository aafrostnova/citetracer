from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from packages.connectors.dblp_sqlite import DblpSQLiteConnector
from packages.connectors.base import RequestPolicy
from packages.core.models import CitationRecord


def _create_test_sqlite(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY, dblp_key TEXT, title TEXT, year INTEGER, venue TEXT, url TEXT)")
        conn.execute("CREATE TABLE authors (id INTEGER PRIMARY KEY, pid TEXT, name TEXT)")
        conn.execute("CREATE TABLE paper_authors (paper_id INTEGER, author_id INTEGER, author_order INTEGER)")

        conn.execute(
            "INSERT INTO papers(id, dblp_key, title, year, venue, url) VALUES (?, ?, ?, ?, ?, ?)",
            (
                1,
                "journals/corr/VaswaniSPUJGKP17",
                "Attention Is All You Need",
                2017,
                "NeurIPS",
                "https://doi.org/10.48550/arXiv.1706.03762",
            ),
        )
        conn.execute("INSERT INTO authors(id, pid, name) VALUES (?, ?, ?)", (1, "p1", "Ashish Vaswani"))
        conn.execute("INSERT INTO authors(id, pid, name) VALUES (?, ?, ?)", (2, "p2", "Noam Shazeer"))
        conn.execute("INSERT INTO paper_authors(paper_id, author_id, author_order) VALUES (?, ?, ?)", (1, 1, 0))
        conn.execute("INSERT INTO paper_authors(paper_id, author_id, author_order) VALUES (?, ?, ?)", (1, 2, 1))
        conn.commit()
    finally:
        conn.close()


class DblpSQLiteConnectorTests(unittest.TestCase):
    def test_search_returns_real_record(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "dblp.sqlite"
            _create_test_sqlite(db_path)

            connector = DblpSQLiteConnector(db_path)
            citation = CitationRecord(citation_id="c1", title="Attention Is All You Need")
            records = connector.search(citation, RequestPolicy())

            self.assertGreaterEqual(len(records), 1)
            first = records[0]
            self.assertEqual(first["title"], "Attention Is All You Need")
            self.assertEqual(first["venue"], "NeurIPS")
            self.assertEqual(first["year"], 2017)
            self.assertIn("Ashish Vaswani", first["authors"])
            self.assertEqual(first["doi"], "10.48550/arxiv.1706.03762")


if __name__ == "__main__":
    unittest.main()

