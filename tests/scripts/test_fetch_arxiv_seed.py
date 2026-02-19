from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fetch_arxiv_seed.py"
SPEC = importlib.util.spec_from_file_location("fetch_arxiv_seed", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["fetch_arxiv_seed"] = MODULE
SPEC.loader.exec_module(MODULE)
build_manifest = MODULE.build_manifest
parse_feed = MODULE.parse_feed


SAMPLE_FEED = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\">
  <entry>
    <id>http://arxiv.org/abs/2501.01234v1</id>
    <updated>2025-01-05T00:00:00Z</updated>
    <published>2025-01-04T00:00:00Z</published>
    <title>  A Useful Paper   </title>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href=\"http://arxiv.org/abs/2501.01234v1\" rel=\"alternate\" type=\"text/html\" />
    <link title=\"pdf\" href=\"http://arxiv.org/pdf/2501.01234v1\" rel=\"related\" type=\"application/pdf\" />
  </entry>
</feed>
"""


class FetchArxivSeedTests(unittest.TestCase):
    def test_parse_feed(self) -> None:
        entries = parse_feed(SAMPLE_FEED)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.arxiv_id, "2501.01234v1")
        self.assertEqual(entry.title, "A Useful Paper")
        self.assertEqual(entry.authors, ["Alice Smith", "Bob Jones"])
        self.assertEqual(entry.year, 2025)
        self.assertEqual(entry.pdf_url, "http://arxiv.org/pdf/2501.01234v1")

    def test_build_manifest(self) -> None:
        entries = parse_feed(SAMPLE_FEED)
        manifest = build_manifest(entries, Path("/tmp/arxiv"))
        self.assertEqual(manifest["source"], "arxiv")
        self.assertEqual(len(manifest["items"]), 1)
        self.assertIn("/tmp/arxiv/2501.01234v1.pdf", manifest["items"][0]["pdf_path"])


if __name__ == "__main__":
    unittest.main()
