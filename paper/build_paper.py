"""Compile the canonical preprint and package its self-contained LaTeX sources."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_FILES = [
    "commercecore_paper.tex",
    "appendices.tex",
    "benchmark_results.tex",
    "figures/workflow.tex",
    "figures/training_dynamics.tex",
    "neurips_2026.sty",
    "evidence/numbers.tex",
    "evidence/checkpoint_history.csv",
    "evidence/checkpoint_rows.tex",
    "evidence/training_loss_history.csv",
    "evidence/training_loss_curve.csv",
    "evidence/loss_window_rows.tex",
    "checklist.tex",
]


def main():
    subprocess.run([sys.executable, str(HERE / "audit_evidence.py")], check=True)
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    compiler = shutil.which("tectonic")
    if not compiler:
        raise SystemExit("Install Tectonic, or use the PDFLaTeX commands in paper/README.md.")
    subprocess.run(
        [compiler, "--keep-logs", "--outdir", str(build), "commercecore_paper.tex"],
        cwd=HERE, check=True,
    )
    log = (build / "commercecore_paper.log").read_text()
    problems = re.findall(
        r"^.*(?:undefined references|undefined citations|(?:Reference|Citation) .+ undefined|Overfull \\[hv]box|Missing character).*$",
        log, re.MULTILINE | re.IGNORECASE,
    )
    if problems:
        raise SystemExit("Resolve publication build warnings before packaging:\n" + "\n".join(problems))
    source = (HERE / "commercecore_paper.tex").read_text()
    if "[preprint,nonatbib]" not in source:
        raise SystemExit("Public build must use the preprint style option.")
    # Source checks run before replacing the public PDF or ZIP.
    for name in SOURCE_FILES:
        if not (HERE / name).is_file():
            raise SystemExit(f"Missing package dependency: {name}")
    public_pdf = HERE / "CommerceCore_Paper.pdf"
    shutil.copyfile(build / "commercecore_paper.pdf", public_pdf)
    archive = HERE / "CommerceCore_arXiv_Source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name in SOURCE_FILES:
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 27, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(entry, (HERE / name).read_bytes())
    manifest = {
        "compiler": subprocess.check_output([compiler, "--version"], text=True).strip(),
        "mode": "public preprint; default build excludes optional conference checklist",
        "top_level_tex": "commercecore_paper.tex",
        "source_files_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_FILES
        },
        "public_pdf_sha256": hashlib.sha256(public_pdf.read_bytes()).hexdigest(),
        "source_zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "validation": "Compiled successfully; no undefined citations/references, missing glyphs, or overfull boxes in final log",
        "boundary": "Local Tectonic compilation; arXiv server compilation and moderation have not been performed",
    }
    (HERE / "evidence" / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {public_pdf.name} and {archive.name}")


if __name__ == "__main__":
    main()
