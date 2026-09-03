#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
mkdir -p figures
if ! command -v pdflatex >/dev/null 2>&1; then
  echo "ERROR: pdflatex not found. Install TeX Live or MiKTeX."
  exit 1
fi
pdflatex -interaction=nonstopmode replication_progress.tex
bibtex replication_progress 2>/dev/null || true
pdflatex -interaction=nonstopmode replication_progress.tex
pdflatex -interaction=nonstopmode replication_progress.tex
echo "PDF: $(pwd)/replication_progress.pdf"
