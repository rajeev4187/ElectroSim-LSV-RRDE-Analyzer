"""Chemical-formula subscript formatting."""

from scripts.modules import formatting


def test_simple_formula():
    assert formatting.format_formula("CuWO4") == "CuWO₄"


def test_formula_keeps_catalogue_number():
    assert (
        formatting.format_formula("LaB6 (NIST SRM 660c)")
        == "LaB₆ (NIST SRM 660c)"
    )


def test_formula_with_decimal_stoichiometry():
    assert formatting.format_formula("Li0.5FePO4") == "Li₀.₅FePO₄"


def test_formula_with_closing_bracket():
    assert formatting.format_formula("Mg(OH)2") == "Mg(OH)₂"


def test_empty_and_none():
    assert formatting.format_formula("") == ""
    assert formatting.format_formula(None) == ""


def test_no_digits_unchanged():
    assert formatting.format_formula("Pt/C") == "Pt/C"
