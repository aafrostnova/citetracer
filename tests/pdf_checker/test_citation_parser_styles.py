from __future__ import annotations

import unittest

from apps.pdf_checker.ingest.citation_parser import parse_reference_entry


class CitationParserStyleTests(unittest.TestCase):
    def test_lncs_colon_style(self) -> None:
        entry = (
            "[77] Xu, K., Hu, W., Leskovec, J., Jegelka, S.: How powerful are graph neural networks? "
            "arXiv preprint arXiv:1810.00826 (2018)"
        )
        record = parse_reference_entry(entry, "pdf-ref:77")
        self.assertEqual(record.title, "How powerful are graph neural networks?")
        self.assertEqual(record.authors, ["Xu, K.", "Hu, W.", "Leskovec, J.", "Jegelka, S."])
        self.assertIn("lncs_or_springer_numeric", record.parsed_fields.get("style_hint", ""))

    def test_apa_style(self) -> None:
        entry = (
            "Smith, J. A., & Doe, R. (2020). Deep learning for time series forecasting. "
            "Journal of AI Research, 10(2), 100-120."
        )
        record = parse_reference_entry(entry, "pdf-ref:1")
        self.assertEqual(record.title, "Deep learning for time series forecasting")
        self.assertEqual(record.parsed_fields.get("style_hint"), "apa")

    def test_mla_style(self) -> None:
        entry = (
            'Smith, John, and Jane Doe. "Learning on Graphs." '
            "Journal of Machine Learning, vol. 12, no. 3, 2021, pp. 12-34."
        )
        record = parse_reference_entry(entry, "pdf-ref:2")
        self.assertEqual(record.title, "Learning on Graphs.")
        self.assertEqual(record.parsed_fields.get("style_hint"), "mla_or_chicago")

    def test_chicago_style(self) -> None:
        entry = (
            'Smith, John, and Jane Doe. "Robust Optimization for Vision." '
            "IEEE Transactions on Pattern Analysis and Machine Intelligence 45, no. 2 (2023): 100-120."
        )
        record = parse_reference_entry(entry, "pdf-ref:3")
        self.assertEqual(record.title, "Robust Optimization for Vision.")
        self.assertEqual(record.parsed_fields.get("style_hint"), "mla_or_chicago")

    def test_harvard_style(self) -> None:
        entry = (
            "Smith, J. and Doe, R., 2019. Contrastive representation learning in medical imaging. "
            "Medical Image Analysis, 58, pp.1-12."
        )
        record = parse_reference_entry(entry, "pdf-ref:4")
        self.assertEqual(record.title, "Contrastive representation learning in medical imaging")
        self.assertEqual(record.parsed_fields.get("style_hint"), "harvard")

    def test_vancouver_style(self) -> None:
        entry = "[12] Smith AB, Doe CD. A practical guide to sparse autoencoders. Neural Computation. 2022;34(5):123-140."
        record = parse_reference_entry(entry, "pdf-ref:5")
        self.assertEqual(record.title, "A practical guide to sparse autoencoders")
        self.assertEqual(record.parsed_fields.get("style_hint"), "vancouver_or_numeric")

    def test_backfill_venue_from_title_tail(self) -> None:
        entry = (
            "Basu, A., Harris, I. R., Hjort, N. L., and Jones, M. "
            "Robust and efficient estimation by minimising a density power divergence. "
            "Biometrika, 85(3):549-559, 1998."
        )
        record = parse_reference_entry(entry, "pdf-ref:6")
        self.assertEqual(record.title, "Robust and efficient estimation by minimising a density power divergence")
        # volume + pages now extracted into dedicated fields; venue keeps the rest.
        self.assertEqual(record.venue, "Biometrika, 1998")
        self.assertEqual(record.volume, "85")
        self.assertEqual(record.pages, "549-559")

    def test_backfill_venue_when_title_is_venue_only(self) -> None:
        entry = "In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=Ua6zuk0WRH."
        record = parse_reference_entry(entry, "pdf-ref:7")
        self.assertEqual(record.venue, "In International Conference on Learning Representations, 2021")
        self.assertEqual(record.title, "")

    def test_split_title_venue_without_period_separator(self) -> None:
        entry = (
            "Yiming Sun, Bing Cao, Pengfei Zhu, and Qinghua Hu. "
            "Drone-based rgb-infrared cross-modality vehicle detection via uncertainty-aware learning "
            "IEEE Transactions on Circuits and Systems for Video Technology, 32(10):6700-6713, 2022b."
        )
        record = parse_reference_entry(entry, "pdf-ref:8")
        self.assertEqual(
            record.title,
            "Drone-based rgb-infrared cross-modality vehicle detection via uncertainty-aware learning",
        )
        self.assertEqual(
            record.venue,
            "IEEE Transactions on Circuits and Systems for Video Technology, 2022b",
        )
        self.assertEqual(record.volume, "32")
        self.assertEqual(record.pages, "6700-6713")

    def test_recovers_boundary_for_long_author_list(self) -> None:
        entry = (
            "Abdin, M., Aneja, J., Awadalla, H., and Zhou, X. "
            "Phi-3 technical report: A highly capable language model locally on your phone, 2024. "
            "URL https://arxiv.org/abs/2404.14219. arXiv:2404.14219."
        )
        record = parse_reference_entry(entry, "pdf-ref:9")
        self.assertTrue(record.title.startswith("Phi-3 technical report"))
        self.assertTrue(len(record.authors) >= 4)
        self.assertIn("arxiv", record.venue.lower())

    def test_arxiv_segment_can_backfill_venue(self) -> None:
        entry = (
            "[4] A Bulatov, M S, G S, M S, D V, and D D. Recurrent memory transformer. "
            "arXiv preprint arXiv:2207.06881, 2022."
        )
        record = parse_reference_entry(entry, "pdf-ref:10")
        self.assertEqual(record.title, "Recurrent memory transformer")
        self.assertIn("arxiv", record.venue.lower())

    def test_in_venue_segment_not_used_as_title(self) -> None:
        entry = (
            "Balle, B., Cherubin, G., and Hayes, J. Reconstructing training data with informed adversaries. "
            "In 2022 IEEE Symposium on Security and Privacy (SP), pp. 1138-1156. IEEE, 2022."
        )
        record = parse_reference_entry(entry, "pdf-ref:11")
        self.assertEqual(record.title, "Reconstructing training data with informed adversaries")
        self.assertTrue(record.venue.startswith("In 2022 IEEE Symposium on Security and Privacy"))

    def test_title_not_cut_by_internal_in_word(self) -> None:
        entry = (
            "Dziugaite, G. K., Hsu, K., Gharbieh, W., Arpino, G., and Roy, D. "
            "On the Role of Data in PAC-Bayes Bounds. "
            "In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, "
            "volume 130 of Proceedings of Machine Learning Research, pp. 604-612. PMLR, 2021."
        )
        record = parse_reference_entry(entry, "pdf-ref:12")
        self.assertEqual(record.title, "On the Role of Data in PAC-Bayes Bounds")

    def test_harvard_does_not_overmatch_into_conference_sentence(self) -> None:
        entry = (
            "Goodfellow, I. J., Erhan, D., and Lee, D.-H. Challenges in representation learning: "
            "A report on three machine learning contests. "
            "In Neural information processing: 20th international conference, ICONIP 2013. "
            "Proceedings, Part III 20, pp. 117-124. Springer, 2013."
        )
        record = parse_reference_entry(entry, "pdf-ref:13")
        self.assertTrue(record.title.startswith("Challenges in representation learning"))
        # Springer is now extracted to the dedicated publisher field (no longer
        # left in venue text alongside other journal/conference metadata).
        self.assertEqual(record.publisher, "Springer")

    def test_split_embedded_journal_suffix_from_title(self) -> None:
        entry = (
            "Devroye, L., Lerasle, M., Lugosi, G., and Oliveira, R. I. "
            "Sub-Gaussian mean estimators. Ann. Stat, 44(6):2695-2725, 2016."
        )
        record = parse_reference_entry(entry, "pdf-ref:14")
        self.assertEqual(record.title, "Sub-Gaussian mean estimators")
        self.assertEqual(record.venue, "Ann. Stat, 2016")
        self.assertEqual(record.volume, "44")
        self.assertEqual(record.pages, "2695-2725")

    def test_neurips_venue_not_taken_as_title(self) -> None:
        entry = (
            "Ament, S., Daulton, S., Eriksson, D., Balandat, M., and Bakshy, E. "
            "Unexpected Improvements to Expected Improvement for Bayesian Optimization. "
            "Advances in Neural Information Processing Systems, 36:20577-20612, 2023."
        )
        record = parse_reference_entry(entry, "pdf-ref:15")
        self.assertEqual(record.title, "Unexpected Improvements to Expected Improvement for Bayesian Optimization")
        self.assertEqual(record.venue, "Advances in Neural Information Processing Systems, 2023")
        self.assertEqual(record.volume, "36")
        self.assertEqual(record.pages, "20577-20612")

    def test_book_style_single_author_publisher_year(self) -> None:
        entry = "Heath, T. L. The thirteen books of Euclid's Elements. Dover Publications, Inc, 1956."
        record = parse_reference_entry(entry, "pdf-ref:16")
        self.assertEqual(record.authors, ["Heath, T. L"])
        self.assertEqual(record.title, "The thirteen books of Euclid's Elements")
        # "Dover Publications, Inc" is not in our literal publisher list and
        # the wildcard does not match "Publications", so it currently stays in
        # venue. The corporate suffix "Inc" no longer gets mis-matched as a
        # country name (the country regex requires ≥3 lowercase chars).
        self.assertIn("Dover Publications", record.venue)
        self.assertIn("1956", record.venue)

    def test_book_style_multi_author_publisher_year(self) -> None:
        entry = "Diakonikolas, I. and Kane, D. M. Algorithmic high-dimensional robust statistics. Cambridge university press, 2023."
        record = parse_reference_entry(entry, "pdf-ref:17")
        self.assertEqual(record.authors, ["Diakonikolas, I.", "Kane, D. M"])
        self.assertEqual(record.title, "Algorithmic high-dimensional robust statistics")
        self.assertEqual(record.venue, "Cambridge university press, 2023")

    def test_journal_epage_stays_in_venue(self) -> None:
        entry = (
            "Avecilla, G., Chuong, J. N., Li, F., Sherlock, G., Gresham, D., and Ram, Y. "
            "Neural networks enable efficient and accurate simulation-based inference of evolutionary parameters "
            "from adaptation dynamics. PLoS biology, 20(5): e3001633, 2022."
        )
        record = parse_reference_entry(entry, "pdf-ref:18")
        self.assertEqual(
            record.title,
            "Neural networks enable efficient and accurate simulation-based inference of evolutionary parameters from adaptation dynamics",
        )
        self.assertEqual(record.venue, "PLoS biology, 20(5): e3001633, 2022")

    def test_org_publisher_segment_is_venue(self) -> None:
        entry = "Collett, E. Field guide to polarization. International society for optics and photonics, 2005."
        record = parse_reference_entry(entry, "pdf-ref:19")
        self.assertEqual(record.authors, ["Collett, E"])
        self.assertEqual(record.title, "Field guide to polarization")
        self.assertEqual(record.venue, "International society for optics and photonics, 2005")

    def test_arxiv_subject_and_particle_initials_preserved(self) -> None:
        entry = (
            "Laurent, T. and Brecht, J. v. A recurrent neural network without chaos, December 2016. "
            "URL http://arxiv.org/abs/1612.06212. arXiv:1612.06212 [cs]."
        )
        record = parse_reference_entry(entry, "pdf-ref:20")
        self.assertIn("Brecht, J. v", record.authors)
        self.assertTrue(record.venue.lower().startswith("arxiv"))
        self.assertIn("[cs]", record.venue)

if __name__ == "__main__":
    unittest.main()
