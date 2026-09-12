#!/usr/bin/env bash
# Compile every TikZ figure to a standalone PDF for \includegraphics.
#   bash paper/figures/tikz/build.sh
set -euo pipefail
cd "$(dirname "$0")"
for f in T*.tex; do
  echo "== $f"
  pdflatex -interaction=nonstopmode -halt-on-error "$f" >/dev/null || { echo "FAILED: $f"; exit 1; }
done
rm -f ./*.aux ./*.log
echo "done: $(ls T*.pdf | tr '\n' ' ')"
