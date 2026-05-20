# Mastering Machine Learning — Book Source

**From Beginner to Expert with Theory and Practical Implementation**
by **Azhar Hussain** · 📞 +92 300 8687258 · ✉️ azharhussaincs@gmail.com

This repository holds the full editable source of the book plus the build
pipeline that compiles everything into a professional **PDF** and a
**PowerPoint** summary deck.

## Folder structure
```
mastering-ml-book/
├── book/
│   ├── front/      # cover, copyright, preface  (00,01,02 .md)
│   ├── chapters/   # one .md per chapter (ch01-...md, ch02-...md, ...)
│   └── back/       # glossary, references, appendix, index
├── assets/
│   ├── css/book.css        # the "premium book" stylesheet
│   └── images/             # diagrams (auto-generated PNGs)
├── scripts/
│   ├── build_pdf.sh        # Markdown -> PDF (Pandoc + WeasyPrint)
│   ├── build_ppt.py        # generates the summary slide deck
│   ├── make_diagrams.py    # generates all figures with matplotlib
│   └── requirements.txt
├── build/                  # output PDF / PPTX (git-ignored)
└── ROADMAP.md              # master plan + progress tracker
```

## Prerequisites
- **Pandoc** (system package) — https://pandoc.org
- **Python 3.10+** with: `pip install -r scripts/requirements.txt`

## Build commands
```bash
# 1. Build the full PDF (also regenerates diagrams)
bash scripts/build_pdf.sh

# 2. Build the PowerPoint summary deck
python3 scripts/build_ppt.py
```

Outputs land in `build/`.

## How the book is written
- One Markdown file per chapter, in **simple English**.
- Callout boxes use Pandoc fenced divs: `::: note`, `::: tip`, `::: warning`, `::: keyidea`.
- Diagrams are generated as code in `scripts/make_diagrams.py` (no binary blobs to maintain).
- Math uses `$ ... $` (inline) and `$$ ... $$` (display).

See **ROADMAP.md** for the full chapter list and progress.
