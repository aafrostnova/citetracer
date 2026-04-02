from __future__ import annotations

from typing import Any

from packages.connectors.arxiv import ArxivConnector
from packages.connectors.base import RequestPolicy
from packages.connectors.openalex import OpenAlexConnector
from packages.connectors.semanticscholar import SemanticScholarConnector
from packages.connectors.url_direct import URLDirectConnector

from .models import CandidateMatch, CitationRecord, DiscrepancyEvidence
from .normalize import NAME_ALIASES, normalize_author, normalize_text, normalize_title, similarity


class SecondaryVerifier:
    """Gathers positive evidence to explain field discrepancies.

    Only when ALL discrepancies are explained can a candidate be upgraded
    from FAKE_REFERENCE to POTENTIAL_REFERENCE.
    """

    def __init__(
        self,
        arxiv_connector: ArxivConnector | None = None,
        semantic_scholar_connector: SemanticScholarConnector | None = None,
        openalex_connector: OpenAlexConnector | None = None,
        url_direct_connector: URLDirectConnector | None = None,
        policy: RequestPolicy | None = None,
    ) -> None:
        self.arxiv = arxiv_connector
        self.semantic_scholar = semantic_scholar_connector
        self.openalex = openalex_connector
        self.url_direct = url_direct_connector or URLDirectConnector()
        self.policy = policy or RequestPolicy()

    def gather_evidence(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        field_status: dict[str, dict[str, Any]],
        conflicts: list[str],
    ) -> list[DiscrepancyEvidence]:
        evidence_list: list[DiscrepancyEvidence] = []

        for field_name, status_info in field_status.items():
            status = str(status_info.get("status", ""))
            if status in ("match", "both_missing", "reference_missing", "candidate_missing", "reordered_match"):
                continue

            handler = self._handlers.get(field_name)
            if handler:
                evidence_list.append(handler(self, citation, candidate, status_info, conflicts))
            else:
                evidence_list.append(DiscrepancyEvidence(
                    field=field_name,
                    discrepancy_type="unknown",
                    evidence_found=False,
                    explanation=f"No handler for field '{field_name}' discrepancy.",
                ))

        return evidence_list

    # ------------------------------------------------------------------
    # Handler: Author name variants / author list changes
    # ------------------------------------------------------------------

    _ET_AL_MARKERS = {"others", "et al", "et al.", "and others", "and others."}

    def _check_authors(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        ref_authors: list[str] = status_info.get("reference", []) or []
        cand_authors: list[str] = status_info.get("candidate", []) or []
        has_et_al = status_info.get("has_et_al", False)

        # Filter out et al. markers
        ref_actual = [a for a in ref_authors if normalize_text(a) not in self._ET_AL_MARKERS]

        # If citation uses et al. and listed authors are a subset of candidate → OK
        if has_et_al and len(ref_actual) < len(cand_authors):
            # Check if all listed ref authors can be found in candidate
            all_found = True
            for r in ref_actual:
                r_norm = normalize_author(r)
                found = False
                for c in cand_authors:
                    c_norm = normalize_author(c)
                    if r_norm == c_norm or _is_initial_match(r_norm, c_norm):
                        found = True
                        break
                if not found:
                    all_found = False
                    break
            if all_found:
                return DiscrepancyEvidence(
                    field="authors",
                    discrepancy_type="et_al_subset",
                    evidence_found=True,
                    explanation=(
                        f"Citation uses 'et al.' with {len(ref_actual)} listed authors; "
                        f"all match the first authors of the {len(cand_authors)}-author candidate list."
                    ),
                    evidence_sources=["structural_check"],
                )

        # Author list count differs → check if version history explains it
        if len(ref_actual) != len(cand_authors):
            return self._check_author_list_change(citation, candidate, ref_actual, cand_authors)

        # partial_overlap or mismatch → check name variants
        return self._check_author_name_variants(citation, candidate, ref_actual, cand_authors)

    def _check_author_name_variants(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        ref_authors: list[str],
        cand_authors: list[str],
    ) -> DiscrepancyEvidence:
        mismatched_pairs: list[tuple[str, str]] = []
        ref_norm = [normalize_author(a) for a in ref_authors]
        cand_norm = [normalize_author(a) for a in cand_authors]

        for r_name in ref_norm:
            found = False
            for c_name in cand_norm:
                if r_name == c_name:
                    found = True
                    break
                if _is_initial_match(r_name, c_name):
                    found = True
                    break
            if not found:
                best_cand = _find_closest_author(r_name, cand_norm)
                mismatched_pairs.append((r_name, best_cand))

        if not mismatched_pairs:
            return DiscrepancyEvidence(
                field="authors",
                discrepancy_type="author_name_variant",
                evidence_found=True,
                explanation="All author name differences resolved by normalization or initial matching.",
                evidence_sources=["structural_check"],
            )

        # Try OpenAlex for remaining mismatches
        all_resolved = True
        explanations: list[str] = []
        raw: dict[str, Any] = {}

        for ref_name, cand_name in mismatched_pairs:
            resolved = False
            if self.openalex and ref_name and cand_name:
                try:
                    ref_info = self.openalex.fetch_author_info(ref_name, self.policy)
                    cand_info = self.openalex.fetch_author_info(cand_name, self.policy)
                    raw[f"{ref_name}_vs_{cand_name}"] = {"ref": ref_info, "cand": cand_info}

                    if ref_info and cand_info:
                        # Same OpenAlex entity
                        if (ref_info.get("openalex_id") and
                                ref_info["openalex_id"] == cand_info.get("openalex_id")):
                            resolved = True
                            explanations.append(
                                f"'{ref_name}' and '{cand_name}' map to same OpenAlex entity "
                                f"{ref_info['openalex_id']}"
                            )
                        # Same ORCID
                        elif (ref_info.get("orcid") and
                              ref_info["orcid"] == cand_info.get("orcid")):
                            resolved = True
                            explanations.append(
                                f"'{ref_name}' and '{cand_name}' share ORCID {ref_info['orcid']}"
                            )
                        # Name appears in alternate names
                        elif _name_in_alternates(ref_name, cand_info) or _name_in_alternates(cand_name, ref_info):
                            resolved = True
                            explanations.append(
                                f"'{ref_name}' found in alternate names of '{cand_name}' or vice versa"
                            )
                except Exception:
                    pass

            if not resolved:
                all_resolved = False
                explanations.append(f"Cannot confirm '{ref_name}' = '{cand_name}'")

        return DiscrepancyEvidence(
            field="authors",
            discrepancy_type="author_name_variant",
            evidence_found=all_resolved,
            explanation="; ".join(explanations) if explanations else "Author name variant check inconclusive.",
            evidence_sources=["openalex"] if self.openalex else [],
            raw_evidence=raw,
        )

    def _check_author_list_change(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        ref_authors: list[str],
        cand_authors: list[str],
    ) -> DiscrepancyEvidence:
        # If candidate has arXiv ID, check version history for author count changes
        arxiv_id = candidate.arxiv_id or (candidate.raw_record.get("arxiv_id") if candidate.raw_record else "")
        if self.arxiv and arxiv_id:
            try:
                versions = self.arxiv.fetch_version_history(arxiv_id, self.policy)
                if len(versions) > 1:
                    return DiscrepancyEvidence(
                        field="authors",
                        discrepancy_type="author_list_change",
                        evidence_found=True,
                        explanation=(
                            f"Paper has {len(versions)} arXiv versions; "
                            f"author list may have changed across versions "
                            f"(ref: {len(ref_authors)} authors, candidate: {len(cand_authors)} authors)."
                        ),
                        evidence_sources=["arxiv_version_history"],
                        raw_evidence={"versions": versions},
                    )
            except Exception:
                pass

        return DiscrepancyEvidence(
            field="authors",
            discrepancy_type="author_list_change",
            evidence_found=False,
            explanation=(
                f"Author count differs (ref: {len(ref_authors)}, candidate: {len(cand_authors)}) "
                f"with no version history to explain the change."
            ),
        )

    # ------------------------------------------------------------------
    # Handler: Venue mismatch (preprint ↔ published)
    # ------------------------------------------------------------------

    def _check_venue(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        ref_venue = normalize_text(str(status_info.get("reference", "") or ""))
        cand_venue = normalize_text(str(status_info.get("candidate", "") or ""))

        arxiv_venues = {"arxiv", "arxiv preprint", "corr"}
        one_is_arxiv = ref_venue in arxiv_venues or cand_venue in arxiv_venues

        if not one_is_arxiv:
            return DiscrepancyEvidence(
                field="venue",
                discrepancy_type="venue_mismatch",
                evidence_found=False,
                explanation=f"Venue mismatch ('{ref_venue}' vs '{cand_venue}') not a preprint-publication case.",
            )

        # One is arXiv, the other is a conference/journal → check preprint linking
        arxiv_id = candidate.arxiv_id or citation.arxiv_id
        doi = candidate.doi or citation.doi

        if self.semantic_scholar and (arxiv_id or doi):
            try:
                paper_id = f"ArXiv:{arxiv_id}" if arxiv_id else f"DOI:{doi}"
                details = self.semantic_scholar.fetch_paper_details(paper_id, self.policy)
                if details:
                    pub_types = details.get("publication_types", [])
                    has_doi = bool(details.get("doi"))
                    has_arxiv = bool(details.get("arxiv_id"))

                    if has_doi and has_arxiv:
                        return DiscrepancyEvidence(
                            field="venue",
                            discrepancy_type="preprint_publication",
                            evidence_found=True,
                            explanation=(
                                f"Paper has both arXiv ({details.get('arxiv_id')}) and "
                                f"DOI ({details.get('doi')}), confirming preprint-to-publication link. "
                                f"Publication types: {pub_types}"
                            ),
                            evidence_sources=["semantic_scholar"],
                            raw_evidence=details,
                        )
            except Exception:
                pass

        return DiscrepancyEvidence(
            field="venue",
            discrepancy_type="preprint_publication",
            evidence_found=False,
            explanation="Could not confirm preprint-to-publication link.",
        )

    # ------------------------------------------------------------------
    # Handler: Year mismatch
    # ------------------------------------------------------------------

    def _check_year(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        ref_year = status_info.get("reference")
        cand_year = status_info.get("candidate")
        arxiv_id = candidate.arxiv_id or citation.arxiv_id

        if self.arxiv and arxiv_id:
            try:
                versions = self.arxiv.fetch_version_history(arxiv_id, self.policy)
                version_years = {v["year"] for v in versions if v.get("year")}
                if ref_year and int(str(ref_year)) in version_years:
                    matching_version = next(
                        (v for v in versions if v.get("year") == int(str(ref_year))), None
                    )
                    return DiscrepancyEvidence(
                        field="year",
                        discrepancy_type="arxiv_version",
                        evidence_found=True,
                        explanation=(
                            f"Citation year {ref_year} matches arXiv version "
                            f"{matching_version['version']} ({matching_version['date']}). "
                            f"Candidate year is {cand_year}."
                        ),
                        evidence_sources=["arxiv_version_history"],
                        raw_evidence={"versions": versions},
                    )
            except Exception:
                pass

        # Check if ±1 year difference is a preprint→publication offset
        try:
            r = int(str(ref_year))
            c = int(str(cand_year))
            if abs(r - c) == 1:
                doi = candidate.doi or citation.doi
                if self.semantic_scholar and (arxiv_id or doi):
                    try:
                        paper_id = f"ArXiv:{arxiv_id}" if arxiv_id else f"DOI:{doi}"
                        details = self.semantic_scholar.fetch_paper_details(paper_id, self.policy)
                        if details and details.get("doi") and details.get("arxiv_id"):
                            return DiscrepancyEvidence(
                                field="year",
                                discrepancy_type="preprint_publication",
                                evidence_found=True,
                                explanation=(
                                    f"Year differs by 1 ({ref_year} vs {cand_year}), "
                                    f"consistent with preprint-to-publication gap. "
                                    f"Paper has both arXiv and DOI."
                                ),
                                evidence_sources=["semantic_scholar"],
                                raw_evidence=details,
                            )
                    except Exception:
                        pass
        except (TypeError, ValueError):
            pass

        return DiscrepancyEvidence(
            field="year",
            discrepancy_type="year_mismatch",
            evidence_found=False,
            explanation=f"Year mismatch ({ref_year} vs {cand_year}) could not be explained.",
        )

    # ------------------------------------------------------------------
    # Handler: DOI mismatch
    # ------------------------------------------------------------------

    def _check_doi(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        ref_doi = str(status_info.get("reference", "") or "").strip()
        cand_doi = str(status_info.get("candidate", "") or "").strip()

        # Resolve citation's DOI via https://doi.org/{doi} to verify it points to the same paper
        if ref_doi and self.url_direct:
            try:
                doi_url = f"https://doi.org/{ref_doi}"
                probe = CitationRecord(citation_id="doi-probe", url=doi_url)
                results = self.url_direct.search(probe, self.policy)
                if results:
                    resolved = results[0]
                    resolved_title = normalize_title(str(resolved.get("title", "") or ""))
                    citation_title = normalize_title(citation.title)
                    candidate_title = normalize_title(candidate.title)

                    title_sim = similarity(resolved_title, citation_title) if resolved_title and citation_title else 0.0

                    if title_sim >= 0.85:
                        return DiscrepancyEvidence(
                            field="doi",
                            discrepancy_type="doi_resolved",
                            evidence_found=True,
                            explanation=(
                                f"DOI {ref_doi} resolves to '{resolved.get('title', '')[:80]}' "
                                f"which matches citation title (similarity={title_sim:.2f}). "
                                f"Candidate DOI '{cand_doi}' differs but paper is the same."
                            ),
                            evidence_sources=["url_direct_doi_resolution"],
                            raw_evidence={"doi_url": doi_url, "resolved": resolved},
                        )
                    else:
                        return DiscrepancyEvidence(
                            field="doi",
                            discrepancy_type="doi_mismatch",
                            evidence_found=False,
                            explanation=(
                                f"DOI {ref_doi} resolves to '{resolved.get('title', '')[:80]}' "
                                f"which does NOT match citation title (similarity={title_sim:.2f})."
                            ),
                            evidence_sources=["url_direct_doi_resolution"],
                            raw_evidence={"doi_url": doi_url, "resolved": resolved},
                        )
            except Exception:
                pass

        return DiscrepancyEvidence(
            field="doi",
            discrepancy_type="doi_mismatch",
            evidence_found=False,
            explanation=(
                f"DOI mismatch: citation='{ref_doi}' vs candidate='{cand_doi}'. "
                f"Could not resolve DOI to verify."
            ),
        )

    # ------------------------------------------------------------------
    # Handler: arXiv ID mismatch
    # ------------------------------------------------------------------

    def _check_arxiv_id(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        return DiscrepancyEvidence(
            field="arxiv_id",
            discrepancy_type="arxiv_id_mismatch",
            evidence_found=False,
            explanation=(
                f"arXiv ID mismatch: citation='{status_info.get('reference')}' "
                f"vs candidate='{status_info.get('candidate')}'. "
                f"arXiv ID is a unique identifier; mismatches indicate different papers."
            ),
        )

    # ------------------------------------------------------------------
    # Handler: URL — different URLs for the same paper are expected
    # ------------------------------------------------------------------

    def _check_url(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        return DiscrepancyEvidence(
            field="url",
            discrepancy_type="url_difference",
            evidence_found=True,
            explanation="URL differences are expected (same paper hosted at different locations).",
        )

    # ------------------------------------------------------------------
    # Handler: pages/publisher/location — verify via url_direct or web_search
    # ------------------------------------------------------------------

    def _check_pages(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        return self._verify_extended_field("pages", citation, candidate, status_info)

    def _check_publisher(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        return self._verify_extended_field("publisher", citation, candidate, status_info)

    def _check_location(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
        conflicts: list[str],
    ) -> DiscrepancyEvidence:
        return self._verify_extended_field("location", citation, candidate, status_info)

    def _verify_extended_field(
        self,
        field_name: str,
        citation: CitationRecord,
        candidate: CandidateMatch,
        status_info: dict[str, Any],
    ) -> DiscrepancyEvidence:
        """Verify pages/publisher/location by resolving citation URL or candidate URL via url_direct."""
        ref_value = normalize_text(str(status_info.get("reference", "") or ""))
        if not ref_value:
            return DiscrepancyEvidence(
                field=field_name, discrepancy_type=f"{field_name}_missing",
                evidence_found=True, explanation=f"Citation does not specify {field_name}.",
            )

        # Try to resolve via URL (citation url or candidate url)
        check_url = citation.url or candidate.url
        if check_url and self.url_direct:
            try:
                probe = CitationRecord(citation_id="field-probe", url=check_url)
                results = self.url_direct.search(probe, self.policy)
                if results:
                    resolved_value = normalize_text(str(results[0].get(field_name, "") or ""))
                    if resolved_value and ref_value in resolved_value or resolved_value in ref_value:
                        return DiscrepancyEvidence(
                            field=field_name, discrepancy_type=f"{field_name}_verified",
                            evidence_found=True,
                            explanation=f"{field_name} '{ref_value}' confirmed by url_direct from {check_url}.",
                            evidence_sources=["url_direct"],
                        )
                    elif resolved_value:
                        return DiscrepancyEvidence(
                            field=field_name, discrepancy_type=f"{field_name}_mismatch",
                            evidence_found=False,
                            explanation=f"{field_name} mismatch: citation='{ref_value}' vs resolved='{resolved_value}'.",
                            evidence_sources=["url_direct"],
                        )
            except Exception:
                pass

        return DiscrepancyEvidence(
            field=field_name, discrepancy_type=f"{field_name}_unverified",
            evidence_found=False,
            explanation=f"Could not verify {field_name} '{ref_value}' — no URL available to resolve.",
        )

    # ------------------------------------------------------------------
    # Handler dispatch
    # ------------------------------------------------------------------

    _handlers: dict[str, Any] = {
        "authors": _check_authors,
        "venue": _check_venue,
        "year": _check_year,
        "doi": _check_doi,
        "arxiv_id": _check_arxiv_id,
        "url": _check_url,
        "pages": _check_pages,
        "publisher": _check_publisher,
        "location": _check_location,
    }


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _is_initial_match(name_a: str, name_b: str) -> bool:
    """Check if two author names match when one uses initials.

    E.g., "a vaswani" matches "ashish vaswani".
    """
    parts_a = name_a.split()
    parts_b = name_b.split()
    if not parts_a or not parts_b:
        return False
    # Last name must match
    if parts_a[-1] != parts_b[-1]:
        return False
    # Check if one has initial where the other has full name
    shorter = parts_a[:-1] if len(parts_a) <= len(parts_b) else parts_b[:-1]
    longer = parts_b[:-1] if len(parts_a) <= len(parts_b) else parts_a[:-1]
    if not shorter or not longer:
        return len(parts_a) == 1 or len(parts_b) == 1  # single-token last-name-only
    for s_part, l_part in zip(shorter, longer):
        if s_part == l_part:
            continue
        if len(s_part) == 1 and l_part.startswith(s_part):
            continue
        if len(l_part) == 1 and s_part.startswith(l_part):
            continue
        return False
    return True


def _find_closest_author(target: str, candidates: list[str]) -> str:
    """Find the candidate author name closest to the target."""
    if not candidates:
        return ""
    target_parts = set(target.split())
    best = ""
    best_overlap = -1
    for cand in candidates:
        cand_parts = set(cand.split())
        overlap = len(target_parts & cand_parts)
        if overlap > best_overlap:
            best_overlap = overlap
            best = cand
    return best


def _name_in_alternates(name: str, author_info: dict[str, Any]) -> bool:
    """Check if a name appears in the alternate_names list of an OpenAlex author record."""
    alternates = author_info.get("alternate_names", [])
    name_norm = normalize_text(name)
    for alt in alternates:
        if normalize_text(str(alt)) == name_norm:
            return True
    return False
