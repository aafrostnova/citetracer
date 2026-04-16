"""Convert mutation test data to verification pipeline input format.

Input:  hallucination_test_data.json (from generate_mutations.py)
Output: A directory of JSON files ready for citation_verification_demo.py,
        plus a ground_truth.json with labels for evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert mutation data to verification input.")
    parser.add_argument("--input", required=True, help="Path to hallucination_test_data.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for verification input")
    parser.add_argument("--batch-size", type=int, default=50, help="Citations per JSON file (0 = all in one file)")
    args = parser.parse_args()

    with open(args.input) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} mutation samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build citations list from mutated data
    citations = []
    ground_truth = []

    for idx, sample in enumerate(samples):
        mutated = sample["mutated"]
        cid = f"mut-{idx + 1:04d}"

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

    # Write verification input files
    if args.batch_size <= 0 or args.batch_size >= len(citations):
        # All in one file
        payload = {
            "input_pdf": "synthetic_mutation_test",
            "reference_entries_count": len(citations),
            "extraction_quality": "HIGH",
            "metadata": {"pipeline_method": "mutation_test"},
            "citations": citations,
        }
        out_path = output_dir / "mutation_test_all.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Written {len(citations)} citations to {out_path}")
    else:
        # Split into batches
        for batch_idx in range(0, len(citations), args.batch_size):
            batch = citations[batch_idx:batch_idx + args.batch_size]
            batch_num = batch_idx // args.batch_size + 1
            payload = {
                "input_pdf": f"synthetic_mutation_test_batch_{batch_num}",
                "reference_entries_count": len(batch),
                "extraction_quality": "HIGH",
                "metadata": {"pipeline_method": "mutation_test", "batch": batch_num},
                "citations": batch,
            }
            out_path = output_dir / f"mutation_test_batch_{batch_num:03d}.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Written batch {batch_num}: {len(batch)} citations to {out_path}")

    # Write ground truth
    gt_path = output_dir / "_ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nGround truth: {gt_path}")

    # Summary
    from collections import Counter
    label_counts = Counter(g["label"] for g in ground_truth)
    subtype_counts = Counter(g["subtype"] for g in ground_truth)

    print(f"\nTotal: {len(ground_truth)} samples")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print()
    for subtype, count in sorted(subtype_counts.items()):
        print(f"  {subtype}: {count}")


if __name__ == "__main__":
    main()
