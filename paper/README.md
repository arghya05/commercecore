# CommerceCore paper

Author: [Arghya Mukherjee](https://orcid.org/0009-0008-3423-8574) · [arghya05@gmail.com](mailto:arghya05@gmail.com) · ORCID: 0009-0008-3423-8574

This folder contains the revised **research paper**, its complete LaTeX source, and the evidence audit. The PDF is a public preprint. It is not an accepted conference paper and has not been submitted to arXiv by this workflow.

- [Research paper PDF](CommerceCore_Paper.pdf)
- [Main LaTeX source](commercecore_paper.tex)
- [Appendices](appendices.tex)
- [arXiv source ZIP](CommerceCore_arXiv_Source.zip)
- [Evidence review and publication status](REVIEW_AND_PUBLICATION_STATUS.md)
- [Machine-readable offline audit](evidence/audit_results.json)
- [arXiv metadata](arxiv_metadata.txt)

## Build and verify

From the repository root, with Python 3.10+ and [Tectonic](https://tectonic-typesetting.github.io/) installed:

```sh
python3 paper/audit_evidence.py
python3 paper/build_paper.py
```

The first command uses only the Python standard library and performs no model inference or paid API calls. The second compiles the PDF, checks the log, and creates a source ZIP. Tectonic may download standard LaTeX packages on its first run.

The source is also compatible with a conventional PDFLaTeX installation:

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error commercecore_paper.tex
pdflatex -interaction=nonstopmode -halt-on-error commercecore_paper.tex
```

The bibliography is embedded in the main `.tex`; no BibTeX step or `.bib` file is needed. Figures are native TikZ/PGFPlots and use the included CSV, so no image-generation service or external graphics path is required. The standard source build requires no shell escape. Local validation uses Tectonic; arXiv's compiler preview remains a separate final check.

## arXiv package

Upload `CommerceCore_arXiv_Source.zip`, then select `commercecore_paper.tex` as the top-level file if needed. The archive contains the manuscript, appendices, official style, generated numeric macros, and checkpoint CSV. It excludes build logs, this review document, historical drafts, and the compiled paper PDF. The manuscript can compile from the archive without the repository's datasets or Python code.

The named author and repository URLs are intentional in the public preprint. The package does not claim acceptance, select an arXiv license on the author's behalf, or submit anything externally. See [arXiv's source instructions](https://info.arxiv.org/help/submit_tex.html).

## Evidence scope

The selected checkpoint's recorded scores are 0.725 constraint F1 on 20 **development** queries and 0.498 QueryNER F1 on 993 records, including 30 exposed during selection. Existing QueryNER API baselines use 50 examples with an incomplete label vocabulary and do not support a matched superiority claim. Six saved diagnostic cases reproduce mean F1 0.400 and yield 0/6 exact-set matches.

The audit adds data-consistency, overlap, prompt-ceiling, and checkpoint analyses. It does not create missing baseline/ablation/serving results. The exact missing experiments are in the review report and paper appendix.

## Template and version maintenance

`neurips_2026.sty` is unmodified from the [official 2026 author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip). The main source uses `preprint`, preserving author attribution without claiming conference acceptance. The optional `checklist.tex` follows that kit and is excluded from the default arXiv manuscript. A future anonymized submission requires a separate review build, anonymous artifact access, updated author attestations, and the applicable conference year's template.

`commercecore_paper.tex` is the canonical manuscript. The Markdown and HTML entries link to the same compiled PDF to avoid divergent versions. The old `architecture_diagram.*` and `sample_reference/` files are historical assets, unused by the revised manuscript and excluded from the source ZIP.
