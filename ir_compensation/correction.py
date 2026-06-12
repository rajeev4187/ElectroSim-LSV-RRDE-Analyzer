"""Ohmic-drop (iR) correction of LSV data.

The measured electrode potential includes an ohmic loss across the
uncompensated resistance Ru:

    E_applied = E_true + I * Ru

so the corrected potential is

    E_corrected = E_measured - factor * I * Ru

where ``factor`` is the user-chosen compensation fraction. Per the project
requirement the GUI exposes ``factor`` as a percentage selectable from
**5 % to 85 %** (e.g. to emulate partial / safe positive-feedback
compensation and avoid over-correction / oscillation).

Units: ``I * Ru`` must come out in volts. Current is stored in the file's
native unit; :data:`CURRENT_UNITS` maps a unit label to the factor that
converts it to amperes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Multipliers converting a current value in the given unit to amperes.
CURRENT_UNITS: dict[str, float] = {
    "A": 1.0,
    "mA": 1e-3,
    "µA": 1e-6,
    "uA": 1e-6,
    "nA": 1e-9,
}

# Allowed compensation-factor range (percent), per project requirement.
MIN_FACTOR_PERCENT = 5
MAX_FACTOR_PERCENT = 85


@dataclass
class CorrectionResult:
    potential_corrected: np.ndarray  # V
    ir_drop: np.ndarray              # V, the subtracted factor * I * Ru term
    ru: float                        # ohm
    factor_percent: float            # %
    current_unit: str

    @property
    def factor(self) -> float:
        return self.factor_percent / 100.0


def clamp_factor_percent(value: float) -> float:
    """Clamp a percentage into the supported [5, 85] range."""
    return float(min(MAX_FACTOR_PERCENT, max(MIN_FACTOR_PERCENT, value)))


def apply_ir_correction(
    potential: np.ndarray,
    current: np.ndarray,
    ru: float,
    factor_percent: float = 85.0,
    current_unit: str = "mA",
) -> CorrectionResult:
    """Return the iR-corrected potential for an LSV sweep.

    Parameters
    ----------
    potential : measured potential, V.
    current   : measured current in ``current_unit``.
    ru        : uncompensated resistance, ohm (from EIS fitting).
    factor_percent : compensation fraction in percent (clamped to 5..85).
    current_unit   : one of :data:`CURRENT_UNITS`.
    """
    if current_unit not in CURRENT_UNITS:
        raise ValueError(
            f"Unknown current unit {current_unit!r}; "
            f"expected one of {sorted(CURRENT_UNITS)}."
        )
    e = np.asarray(potential, dtype=float)
    i_amp = np.asarray(current, dtype=float) * CURRENT_UNITS[current_unit]
    factor_percent = clamp_factor_percent(factor_percent)
    ir_drop = (factor_percent / 100.0) * i_amp * float(ru)  # volts
    return CorrectionResult(
        potential_corrected=e - ir_drop,
        ir_drop=ir_drop,
        ru=float(ru),
        factor_percent=factor_percent,
        current_unit=current_unit,
    )


@dataclass
class CorrectionAssessment:
    """Diagnostic for whether a correction is over-compensated.

    Over-compensation manifests as the corrected potential *folding back*:
    instead of advancing monotonically with the sweep, the curve reverses and
    re-enters potentials it already covered. ``reverted_fraction`` is the share
    of points lying behind the running extreme of the corrected sweep.
    """

    direction: int               # +1 anodic (increasing), -1 cathodic
    reverted_fraction: float     # fraction of corrected points that fold back
    raw_reverted_fraction: float # same metric on the raw data (noise baseline)
    over_compensated: bool
    message: str


def assess_correction(
    raw_potential: np.ndarray, corrected_potential: np.ndarray
) -> CorrectionAssessment:
    """Flag over-compensation by detecting fold-back in the corrected sweep."""
    e_raw = np.asarray(raw_potential, dtype=float)
    e_cor = np.asarray(corrected_potential, dtype=float)
    n = len(e_cor)
    direction = 1 if (n < 2 or e_raw[-1] >= e_raw[0]) else -1

    span = float(np.ptp(e_raw)) or 1.0
    tol = 0.005 * span  # ignore reversals smaller than 0.5 % of the sweep span

    def _fold_fraction(arr: np.ndarray) -> float:
        if n < 3:
            return 0.0
        if direction > 0:
            running = np.maximum.accumulate(arr)
            reverted = arr < running - tol
        else:
            running = np.minimum.accumulate(arr)
            reverted = arr > running + tol
        return float(np.mean(reverted))

    raw_frac = _fold_fraction(e_raw)
    cor_frac = _fold_fraction(e_cor)
    # Flag only if the corrected curve folds back noticeably more than the raw.
    over = cor_frac > max(0.02, raw_frac * 1.5 + 0.01)

    if over:
        msg = (
            f"Over-compensated: {cor_frac:.0%} of the corrected sweep folds back "
            "on itself (potential reverses). Reduce the compensation factor."
        )
    elif cor_frac > raw_frac + 0.005:
        msg = "Borderline: slight fold-back appearing — near the safe limit."
    else:
        msg = "Good: the corrected sweep stays monotonic (no fold-back)."

    return CorrectionAssessment(
        direction=direction,
        reverted_fraction=cor_frac,
        raw_reverted_fraction=raw_frac,
        over_compensated=over,
        message=msg,
    )


def recommend_factor(
    potential: np.ndarray,
    current: np.ndarray,
    ru: float,
    current_unit: str = "mA",
    candidates: tuple[int, ...] | None = None,
) -> int:
    """Return the highest factor (%, ≤ 85) whose correction does not fold back.

    Scans candidate factors from high to low; the first that is not
    over-compensated is the recommendation. If every candidate folds back, the
    smallest is returned.
    """
    if candidates is None:
        candidates = (85, 80, 70, 60, 50, 40, 30, 20, 10, 5)
    ordered = sorted({int(clamp_factor_percent(c)) for c in candidates}, reverse=True)
    for f in ordered:
        res = apply_ir_correction(potential, current, ru, f, current_unit)
        if not assess_correction(potential, res.potential_corrected).over_compensated:
            return f
    return ordered[-1]
