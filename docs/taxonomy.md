## Citation Hallucination Taxonomy

### REAL

All information provided in the citation is verifiable against authoritative sources.

| Subtype                     | Description                                                                                                                                                                                                                                          | Example                                                                   |
| -----------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ---------------------------------------------------------------------------|
| **R1. Exact match**         | Every field the reference provides is matched by the candidate -- core fields (title, authors, year) plus any optional fields present (venue, DOI, arxiv_id, volume, pages, publisher, location). Fields the reference does not provide are skipped. | Reference has title/authors/venue/year/pages; candidate matches all five. |
| **R2. Format variant**      | Auto-normalizable formatting differences: case, venue abbreviation, first name initial abbreviation                                                                                                                                                  | "Artif. Intell." vs "Artificial Intelligence", "G. Hao" vs "Gao Hao"      |
| **R3. Et al. abbreviation** | Author list uses "et al."/"Others", all listed authors are correct but full list cannot be compared                                                                                                                                                  | "S. Bubeck, V. Chandrasekaran, et al." matches first 2 of 14              |

---

### POTENTIAL HALLUCINATED

A strong anchor paper is found, but discrepancies exist that can be explained by positive evidence, source is unstable, or evidence is insufficient.

| Subtype                             | Description                                                                                                                                                                                                                                                                                                                                       | Example                                                                                                      |
| -------------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------------------------------------------------------------------------------|
| **P1. Author name variant**         | Author name is a plausible nickname, shortened form, or transliteration variant                                                                                                                                                                                                                                                                   | "Katherine" vs "Kate", "Mike" vs "Michael"                                                                   |
| **P2. Non-academic source**         | Citation references a non-academic source (blog, tweet, GitHub, personal website) whose existence cannot be fully verified through structured databases. The source may have been deleted/moved, or may still exist but is not indexed by academic databases. Indirect evidence (reposts, caches, web search snippets) may or may not confirm it. | Tweet deleted but Facebook repost quotes the exact text                                                      |
| **P3. Cross-candidate coverage**    | No single candidate matches all fields, but every reference field is independently matched by at least one candidate. Needs human review to determine if it is the same paper.                                                                                                                                                                    | Title+authors match in DBLP, venue+year match in CrossRef, but no single source has all four                 |
| **P4. Insufficient field evidence** | All core fields (title, authors, venue, year) match, but volume, pages, or publisher cannot be verified because no candidate source provides them. Not a mismatch -- the data is simply absent from all sources.                                                                                                                                  | DBLP confirms title/authors/venue/year, but has no volume or pages data to verify "vol. 37, pp. 24081-24125" |

---

### HALLUCINATED

Default stance. Paper not found, or discrepancies cannot be explained by retrieved evidence. **If ANY error exists, the citation is HALLUCINATED.** Output lists ALL detected errors.

| Subtype                      | Category | Description                                                                                               | Example                                                                  |
| ------------------------------| ----------| -----------------------------------------------------------------------------------------------------------| --------------------------------------------------------------------------|
| **H1. Title error**          | Title    | Title differs from real paper (word substitution, paraphrase, or fabrication)                             | "Attention Is **What** You Need" vs real "Attention Is **All** You Need" |
| **H2. Author error**         | Author   | Author list wrong: addition, deletion, reordering, or fabrication                                         | Real: "Alice, Bob" -> "Alice, Bob, **Carol**" or "**Bob, Alice**"        |
| **H3. Venue error**          | Meta     | Paper exists but at a different venue than claimed                                                        | "In EACL, 2024" but paper only on arXiv                                  |
| **H4. Year error**           | Meta     | Year is verifiably wrong                                                                                  | arXiv:24...04 with year 2021 but actually 2024                           |
| **H5. DOI/identifier error** | Meta     | DOI resolves to a different paper or does not exist                                                       | DOI points to paper with different title/authors                         |
| **H6. Pages/volume error**   | Meta     | Pages, volume, or publisher is **verifiably wrong** (candidate has a different value, not merely missing) | "pages 1234-1245" but publisher shows "pages 789-800"                    |
