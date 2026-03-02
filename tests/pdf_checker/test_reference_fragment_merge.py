from __future__ import annotations

from apps.pdf_checker.ingest.reference_segmenter import (
    _extract_reference_entries_from_tagged_markdown,
    _finalize_raw_entries,
)


def test_finalize_merges_reference_prefix_and_continuation_fragment() -> None:
    raw_entries = [
        "Dziugaite, G. K., Hsu, K., Gharbieh, W., Arpino, G., and Roy, D. "
        "On the Role of Data in PAC-Bayes Bounds.",
        "In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, "
        "volume 130 of Proceedings of Machine Learning Research, pp. 604–612. PMLR, 2021.",
    ]

    finalized = _finalize_raw_entries(raw_entries)

    assert len(finalized) == 1
    assert "On the Role of Data in PAC-Bayes Bounds." in finalized[0]
    assert "Proceedings of The 24th International Conference on Artificial Intelligence and Statistics" in finalized[0]
    assert "PMLR, 2021" in finalized[0]


def test_tagged_markdown_keeps_reference_prefix_fragment_before_merge() -> None:
    markdown = (
        "<|ref|>sub_title<|/ref|><|det|>[[1,1,1,1]]<|/det|>REFERENCES\n"
        "<|ref|>text<|/ref|><|det|>[[1,1,1,1]]<|/det|>"
        "Dziugaite, G. K., Hsu, K., Gharbieh, W., Arpino, G., and Roy, D. "
        "On the Role of Data in PAC-Bayes Bounds.\n"
        "<|ref|>text<|/ref|><|det|>[[1,1,1,1]]<|/det|>"
        "In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, "
        "volume 130 of Proceedings of Machine Learning Research, pp. 604–612. PMLR, 2021.\n"
    )

    extracted = _extract_reference_entries_from_tagged_markdown(markdown)
    finalized = _finalize_raw_entries(extracted)

    assert len(finalized) == 1
    assert "On the Role of Data in PAC-Bayes Bounds." in finalized[0]
    assert "PMLR, 2021" in finalized[0]
