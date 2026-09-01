"""Chemical-formula text formatting shared across every GUI tab.

Ported from the sibling ``Working-Apps/XRD`` project
(``scripts/xrd_structure.py``), which solved the same problem: a sample name
typed as ``TiO2`` should read as ``TiO₂`` wherever it appears — on a Plotly
axis/legend, in a results table, in a CSV, in a Word/PNG export, and in a
download's file name. Unicode subscript/superscript characters (not HTML/LaTeX
markup) are the only representation that survives all of those unchanged.
"""

from __future__ import annotations

import re

# Unicode subscript digits, and the characters a formula may legitimately
# carry alongside them (a decimal point, for non-stoichiometric formulas like
# "Li0.5FePO4").
_SUBSCRIPTS = str.maketrans("0123456789.", "₀₁₂₃₄₅₆₇₈₉.")
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?|\)|\])(\d+(?:\.\d+)?)")


def format_formula(text: str | None) -> str:
    """Chemical formula with proper subscripts: ``CuWO4`` -> ``CuWO₄``.

    Only digits that directly follow an element symbol or a closing bracket
    are lowered, so a name like "LaB6 (NIST SRM 660c)" keeps its catalogue
    number intact while its formula part is set correctly.
    """
    if not text:
        return ""

    def sub(match: re.Match) -> str:
        return match.group(1) + match.group(2).translate(_SUBSCRIPTS)

    return _FORMULA_TOKEN.sub(sub, str(text))
