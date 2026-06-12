"""Convert README.md to a styled README.pdf (pure-Python, Windows-friendly)."""
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "README.md"
OUT = ROOT / "README.pdf"

CSS = """
@page { size: letter; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt;
       color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 20pt; color: #0b4f6c; border-bottom: 2px solid #0b4f6c;
     padding-bottom: 4px; }
h2 { font-size: 14pt; color: #0b4f6c; margin-top: 18px;
     border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { font-size: 11.5pt; color: #145374; margin-top: 14px; }
p { margin: 6px 0; }
a { color: #1565c0; text-decoration: none; }
code { font-family: "Courier New", monospace; font-size: 9pt;
       background: #f2f2f2; padding: 1px 3px; }
pre { font-family: "Courier New", monospace; font-size: 8.5pt;
      background: #f5f5f5; border: 1px solid #ddd; padding: 8px;
      -pdf-keep-with-next: true; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td { border: 1px solid #bbb; padding: 4px 6px; font-size: 9pt;
         text-align: left; }
th { background: #e7eff3; }
blockquote { border-left: 3px solid #0b4f6c; margin: 8px 0;
             padding: 2px 10px; color: #444; background: #f7fafc; }
hr { border: none; border-top: 1px solid #ccc; }
"""

def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    with OUT.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} error(s)")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
