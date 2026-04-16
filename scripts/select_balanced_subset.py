"""Select a balanced subset from generated mutations covering all subtypes.

Input:  hallucination_test_data.json (from generate_mutations.py)
Output: A directory with balanced_test.json + _ground_truth.json
        ready for citation_verification_demo.py
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Select balanced mutation subset for testing.")
    parser.add_argument("--input", required=True, help="Path to mutations JSON (from generate_mutations.py)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--total", type=int, default=25, help="Target total samples (default 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} mutations")

    # Group by subtype
    by_subtype: dict[str, list] = defaultdict(list)
    for s in samples:
        by_subtype[s["subtype"]].append(s)

    subtypes = sorted(by_subtype.keys())
    n_subtypes = len(subtypes)
    print(f"Found {n_subtypes} subtypes: {subtypes}")

    # Distribute: at least 1 per subtype, then fill remaining evenly
    per_type = max(1, args.total // n_subtypes)
    remainder = args.total - per_type * n_subtypes

    selected: list[dict] = []
    for st in subtypes:
        pool = by_subtype[st]
        n = per_type + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        chosen = random.sample(pool, min(n, len(pool)))
        selected.extend(chosen)
        print(f"  {st:12s}: {len(chosen)} selected (from {len(pool)} available)")

    random.shuffle(selected)
    print(f"\nTotal selected: {len(selected)}")

    # Build output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    citations = []
    ground_truth = []

    for idx, sample in enumerate(selected):
        mutated = sample["mutated"]
        cid = f"bal-{idx + 1:04d}"

        citation = {
            "citation_id": cid,
            "raw_text": "",
            "title": mutated.get("title", ""),
            "authors": mutated.get("authors", []),
            "venue": mutated.get("venue", ""),
            "year": mutated.get("year"),
            "doi": mutated.get("doi", ""),
            "arxiv_id": mutated.get("arxiv_id", ""),
            "url": mutated.get("url", ""),
            "volume": mutated.get("volume", ""),
            "pages": mutated.get("pages", ""),
            "publisher": mutated.get("publisher", ""),
            "location": mutated.get("location", ""),
            "source_span": "",
            "provenance": {},
            "parsed_fields": {},
        }
        citations.append(citation)

        ground_truth.append({
            "citation_id": cid,
            "original_citation_id": sample.get("original_citation_id", ""),
            "label": sample["label"],
            "subtype": sample["subtype"],
            "category": sample.get("category", ""),
            "mutation_type": sample.get("mutation_type", ""),
            "explanation": sample.get("explanation", ""),
            "changed_fields": sample.get("changed_fields", []),
        })

    # Write verification input
    payload = {
        "input_pdf": "mutation_balanced_test",
        "reference_entries_count": len(citations),
        "extraction_quality": "HIGH",
        "metadata": {"pipeline_method": "mutation_balanced_test"},
        "citations": citations,
    }
    out_path = output_dir / "balanced_test.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Write ground truth
    gt_path = output_dir / "_ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Summary
    from collections import Counter
    label_counts = Counter(g["label"] for g in ground_truth)
    subtype_counts = Counter(g["subtype"] for g in ground_truth)

    print(f"\n{'='*50}")
    print(f"Output: {out_path}")
    print(f"Ground truth: {gt_path}")
    print(f"\nTotal: {len(ground_truth)} samples")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print()
    for subtype, count in sorted(subtype_counts.items()):
        print(f"  {subtype:5s}: {count}")


if __name__ == "__main__":
    main()
