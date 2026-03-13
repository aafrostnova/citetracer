from __future__ import annotations

import argparse
import json
from pathlib import Path


MAIN_TEX = r"""\documentclass{article}
\begin{document}
This synthetic paper intentionally mixes citation quality for testing.

A verified citation: \cite{vaswani2017attention}.
A likely flawed metadata citation: \cite{bert_wrongvenue}.
A fabricated citation: \cite{quantum_unicorn_2025}.

\bibliographystyle{plain}
\bibliography{refs}
\end{document}
"""


REFS_BIB = r"""@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017},
  doi={10.48550/arXiv.1706.03762}
}

@inproceedings{bert_wrongvenue,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@inproceedings{quantum_unicorn_2025,
  title={Quantum Unicorn Gradient Descent for Sentient Loss Surfaces},
  author={Doe, Jane and Doe, John},
  booktitle={Proceedings of the Imaginary Learning Conference},
  year={2025}
}
"""


EXPECTED_LABELS = {
    "bib:vaswani2017attention": "VALID",
    "bib:bert_wrongvenue": "FLAWED_CITATION",
    "bib:quantum_unicorn_2025": "INSUFFICIENT_EVIDENCE",
}



def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "main.tex"
    bib_path = output_dir / "refs.bib"
    labels_path = output_dir / "expected_labels.json"

    tex_path.write_text(MAIN_TEX, encoding="utf-8")
    bib_path.write_text(REFS_BIB, encoding="utf-8")
    labels_path.write_text(json.dumps(EXPECTED_LABELS, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "main_tex": str(tex_path),
        "refs_bib": str(bib_path),
        "expected_labels": str(labels_path),
        "citation_count": len(EXPECTED_LABELS),
    }



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic LaTeX fixture with mixed citation integrity.")
    parser.add_argument(
        "--output-dir",
        default="data/fixtures/latex_papers/synthetic_mixed_source",
        help="Where to write the synthetic LaTeX fixture.",
    )
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(Path(args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
