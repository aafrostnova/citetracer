# Annotation Guidelines

## Labels

- `VALID`: reference exists and metadata aligns.
- `FLAWED_CITATION`: real reference with non-trivial metadata mismatch.
- `SUSPECTED_HALLUCINATION`: likely fabricated or highly inconsistent reference.
- `INSUFFICIENT_EVIDENCE`: cannot adjudicate with confidence.

## Protocol

- Use at least two bibliographic sources for positive confirmation when possible.
- Preserve source URLs and notes in annotation logs.
- If sources conflict, prefer `INSUFFICIENT_EVIDENCE` unless mismatch is decisive.
- Track name normalization edge cases (nicknames, initials, middle names, spacing).
