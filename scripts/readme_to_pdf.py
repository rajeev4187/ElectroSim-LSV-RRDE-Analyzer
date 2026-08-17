"""Convert README.md to a styled README.pdf (pure-Python, Windows-friendly)."""
import re
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


_FENCE_RE = re.compile(r"^([ \t]*)```")


def _dedent_fenced_code(md_text: str) -> str:
    """Strip leading indentation from fenced code blocks.

    python-markdown's fenced_code extension only recognizes ``` fences that
    start at column 0; fences indented under a list item (as GitHub's
    renderer allows) fall through to being parsed as literal inline code,
    leaking the language tag (e.g. "text") into the output. Dedenting here
    keeps README.md's GitHub-friendly nesting intact while fixing the PDF.
    """
    lines = md_text.split("\n")
    out = []
    indent = None
    for line in lines:
        match = _FENCE_RE.match(line)
        if match and indent is None and match.group(1):
            indent = match.group(1)
            out.append(line[len(indent):])
        elif match and indent is not None:
            out.append(line[len(indent):] if line.startswith(indent) else line.lstrip())
            indent = None
        elif indent is not None:
            out.append(line[len(indent):] if line.startswith(indent) else line.lstrip())
        else:
            out.append(line)
    return "\n".join(out)


# The base-14 Helvetica used by xhtml2pdf can't render subscript/superscript
# digits (they show as tofu boxes). Rewrite them as <sub>/<sup> markup, which
# pisa handles natively — except inside inline `code` spans, where markdown
# escapes raw HTML, so those get an ASCII fallback instead.
_SUBSUP_REPLACEMENTS = [
    ("₂", "<sub>2</sub>"),  # CO₂RR, O₂, H₂SO₄, ...
    ("₃", "<sub>3</sub>"),  # NO₃RR
    ("₄", "<sub>4</sub>"),  # HClO₄, H₂SO₄
]

# Glyphs outside Helvetica's WinAnsi set that also show as tofu boxes. Greek
# gets a spelled-out fallback; arrows and comparisons go through HTML entities
# (preserved by markdown), so keep these out of inline `code` spans, where
# entities would be escaped and shown literally.
_GLYPH_FALLBACKS = [
    ("Ω", "Ohm"),
    ("ω", "omega"),
    ("η", "eta"),
    ("↔", "&lt;-&gt;"),
    ("→", "-&gt;"),
    ("≥", "&gt;="),
    ("≤", "&lt;="),
    ("−", "-"),  # U+2212 minus sign; safe anywhere, code spans included
]


def _fix_unsupported_glyphs(md_text: str) -> str:
    md_text = md_text.replace("pH⁻¹", "pH^-1")  # inside an inline `code` span
    for old, new in _SUBSUP_REPLACEMENTS + _GLYPH_FALLBACKS:
        md_text = md_text.replace(old, new)
    return md_text.replace("\U0001f517 ", "")  # 🔗 emoji has no PDF glyph either


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    md_text = _dedent_fenced_code(md_text)
    md_text = _fix_unsupported_glyphs(md_text)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )
    html = (
        f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    with OUT.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} error(s)")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
