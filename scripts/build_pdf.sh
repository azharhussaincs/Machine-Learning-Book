#!/usr/bin/env bash
# =============================================================================
# build_pdf.sh — compiles the whole book into one professional PDF.
#
#   Pipeline:  Markdown  ->  (Pandoc)  ->  HTML + CSS  ->  (WeasyPrint)  ->  PDF
#
# Run from the PROJECT ROOT:   bash scripts/build_pdf.sh
# Output:                      build/Mastering-Machine-Learning.pdf
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="build/Mastering-Machine-Learning.pdf"
CSS="assets/css/book.css"
FRONT_HTML="build/front.html"

echo "==> Regenerating diagrams..."
python3 scripts/make_diagrams.py

echo "==> Building front matter (cover + copyright + preface)..."
# The cover is hand-written raw HTML (injected verbatim). Copyright + preface
# are markdown -> HTML, then concatenated AFTER the cover.
pandoc book/front/01-copyright.md book/front/02-preface.md \
  -f markdown+raw_html+fenced_divs -t html5 -o "build/_frontmatter.html"
cat book/front/00-cover.html "build/_frontmatter.html" > "$FRONT_HTML"

# Collect chapters (sorted) and back matter, skipping any file starting with "_".
mapfile -t CHAPTERS < <(find book/chapters -name 'ch[0-9]*.md' | sort)
mapfile -t BACK     < <(find book/back     -name '[0-9]*.md' | sort)

if [ ${#CHAPTERS[@]} -eq 0 ]; then
  echo "!! No chapters found in book/chapters/ — nothing to build."; exit 1
fi

echo "==> Chapters found: ${#CHAPTERS[@]}"
printf '    - %s\n' "${CHAPTERS[@]}"

echo "==> Rendering PDF with WeasyPrint..."
pandoc "${CHAPTERS[@]}" "${BACK[@]}" \
  -f markdown+raw_html+fenced_divs+pipe_tables+tex_math_dollars \
  --standalone --embed-resources \
  --resource-path=".:assets:assets/images:book/chapters" \
  --css "$CSS" \
  --toc --toc-depth=2 \
  --number-sections \
  --include-before-body="$FRONT_HTML" \
  --syntax-highlighting=tango \
  --metadata pagetitle="Mastering Machine Learning" \
  -V toc-title="Table of Contents" \
  --pdf-engine=weasyprint \
  -o "$OUT"

echo "==> DONE: $OUT"
ls -lh "$OUT"
