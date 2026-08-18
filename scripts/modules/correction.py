"""Ohmic-drop (iR) correction of LSV data.

The measured electrode potential includes an ohmic loss across the
uncompensated resistance Ru:

    E_applied = E_true + I * Ru

so the corrected potential is

    E_corrected = E_measured - factor * I * Ru

where ``factor`` is the user-chosen compensation fraction. The GUI exposes
``factor`` as a percentage selectable from **5 % to 100 %**. 100 % is the full
ohmic-drop correction; partial compensation (≤ 85 % is the recommended default)
emulates safe positive-feedback compensation and guards against
over-correction / oscillation when Ru is uncertain.

Units: ``I * Ru`` must come out in volts. There are two consistent ways to
reach volts, depending on how the LSV current is reported:

* **Absolute current** (e.g. mA) with **Ru in Ω**: ``I[A] * Ru[Ω] = V``.
* **Current density** (e.g. mA/cm²) with **Ru in Ω·cm²**:
  ``j[A/cm²] * Ru[Ω·cm²] = V``.

When the two halves disagree on whether they are area-normalised, the
electrode area reconciles them: ``Ru[Ω·cm²] = Ru[Ω] * area[cm²]`` (and the
inverse). The current is always scaled to its base SI unit first
(:data:`CURRENT_UNITS` / :data:`CURRENT_DENSITY_UNITS` — e.g. mA → A divides
by 1000), which is what keeps the final drop in volts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Multipliers converting an absolute current value in the given unit to amperes.
CURRENT_UNITS: dict[str, float] = {
    "A": 1.0,
    "mA": 1e-3,
    "µA": 1e-6,
    "uA": 1e-6,
    "nA": 1e-9,
}

# Multipliers converting a current-density value in the given unit to A/cm².
CURRENT_DENSITY_UNITS: dict[str, float] = {
    "A/cm²": 1.0,
    "mA/cm²": 1e-3,
    "µA/cm²": 1e-6,
    "uA/cm²": 1e-6,
    "nA/cm²": 1e-9,
}

# Recognised units for the uncompensated resistance Ru (EIS).
RU_OHM = "Ω"
RU_OHM_CM2 = "Ω·cm²"
RU_UNITS = (RU_OHM, RU_OHM_CM2)

# Allowed compensation-factor range (percent). 100 % = full iR correction;
# 85 % is the recommended safe default to avoid over-correction.
MIN_FACTOR_PERCENT = 5
MAX_FACTOR_PERCENT = 100


def is_density_unit(current_unit: str) -> bool:
    """True if ``current_unit`` is a current *density* (per-area) unit."""
    return current_unit in CURRENT_DENSITY_UNITS


def needs_area(current_unit: str, ru_unit: str) -> bool:
    """Whether an electrode area is required to reconcile the two units.

    Area is only needed when one side is area-normalised and the other is
    not (current density vs Ru in Ω, or absolute current vs Ru in Ω·cm²).
    When both agree the units are already consistent and area is bypassed.
    """
    density = is_density_unit(current_unit)
    return density != (ru_unit == RU_OHM_CM2)


@dataclass
class CorrectionResult:
    potential_corrected: np.ndarray  # V
    ir_drop: np.ndarray              # V, the subtracted factor * I * Ru term
    ru: float                        # value as supplied (in ``ru_unit``)
    factor_percent: float            # %
    current_unit: str
    ru_unit: str = RU_OHM            # unit Ru was supplied in
    ru_effective: float | None = None  # Ru after area reconciliation
    area_cm2: float | None = None    # electrode area used (cm²), if any

    @property
    def factor(self) -> float:
        return self.factor_percent / 100.0


def clamp_factor_percent(value: float) -> float:
    """Clamp a percentage into the supported [5, 100] range."""
    return float(min(MAX_FACTOR_PERCENT, max(MIN_FACTOR_PERCENT, value)))


def reconcile_ru(
    ru: float,
    ru_unit: str = RU_OHM,
    current_unit: str = "mA",
    area_cm2: float | None = None,
) -> float:
    """Return Ru in the unit that pairs with ``current_unit`` to give volts.

    * current density (A/cm²) needs Ru in Ω·cm²; if Ru is in Ω it is
      multiplied by the electrode area.
    * absolute current (A) needs Ru in Ω; if Ru is in Ω·cm² it is divided by
      the electrode area.

    When current and Ru units already agree the value is returned unchanged
    (area is not needed). Raises ``ValueError`` if an area is required but not
    given (or non-positive).
    """
    if ru_unit not in RU_UNITS:
        raise ValueError(
            f"Unknown Ru unit {ru_unit!r}; expected one of {RU_UNITS}."
        )
    ru = float(ru)
    if not needs_area(current_unit, ru_unit):
        return ru  # units already consistent -> bypass
    if not area_cm2 or area_cm2 <= 0:
        raise ValueError(
            "Electrode area (cm²) is required to reconcile current density "
            "with Ru when their area-normalisation differs."
        )
    if is_density_unit(current_unit):
        return ru * float(area_cm2)   # Ω -> Ω·cm²
    return ru / float(area_cm2)       # Ω·cm² -> Ω


def convert_ru_unit(
    ru: float,
    from_unit: str,
    to_unit: str,
    area_cm2: float | None = None,
) -> float:
    """Convert a Ru value between the two supported units (Ω, Ω·cm²).

    Unlike :func:`reconcile_ru` (which picks the target unit from a current
    unit's density-ness), this converts directly between the two Ru units —
    e.g. for a value typed in by the user that isn't already in the unit the
    rest of the analysis is using. Returns ``ru`` unchanged when the units
    already match (area not required in that case).
    """
    if from_unit not in RU_UNITS:
        raise ValueError(f"Unknown Ru unit {from_unit!r}; expected one of {RU_UNITS}.")
    if to_unit not in RU_UNITS:
        raise ValueError(f"Unknown Ru unit {to_unit!r}; expected one of {RU_UNITS}.")
    ru = float(ru)
    if from_unit == to_unit:
        return ru
    if not area_cm2 or area_cm2 <= 0:
        raise ValueError(
            "Electrode area (cm²) is required to convert between Ω and Ω·cm²."
        )
    if from_unit == RU_OHM and to_unit == RU_OHM_CM2:
        return ru * float(area_cm2)   # Ω -> Ω·cm²
    return ru / float(area_cm2)       # Ω·cm² -> Ω


def apply_ir_correction(
    potential: np.ndarray,
    current: np.ndarray,
    ru: float,
    factor_percent: float = 85.0,
    current_unit: str = "mA",
    ru_unit: str = RU_OHM,
    area_cm2: float | None = None,
) -> CorrectionResult:
    """Return the iR-corrected potential for an LSV sweep.

    Parameters
    ----------
    potential : measured potential, V.
    current   : measured current (or current density) in ``current_unit``.
    ru        : uncompensated resistance from EIS fitting, in ``ru_unit``.
    factor_percent : compensation fraction in percent, clamped to
                     ``[MIN_FACTOR_PERCENT, MAX_FACTOR_PERCENT]`` by
                     :func:`clamp_factor_percent`.
    current_unit   : one of :data:`CURRENT_UNITS` (absolute) or
                     :data:`CURRENT_DENSITY_UNITS` (per-area).
    ru_unit        : ``"Ω"`` or ``"Ω·cm²"``.
    area_cm2       : electrode area, only used (and required) when the current
                     and Ru units disagree on area-normalisation.

    The current is first scaled to its base SI unit (A or A/cm²; e.g. mA → A
    divides by 1000) and Ru is reconciled to the matching unit, so the drop
    ``factor · I · Ru`` always comes out in volts.
    """
    density = is_density_unit(current_unit)
    if not density and current_unit not in CURRENT_UNITS:
        raise ValueError(
            f"Unknown current unit {current_unit!r}; expected one of "
            f"{sorted(CURRENT_UNITS)} or {sorted(CURRENT_DENSITY_UNITS)}."
        )
    scale = (CURRENT_DENSITY_UNITS if density else CURRENT_UNITS)[current_unit]
    e = np.asarray(potential, dtype=float)
    i_base = np.asarray(current, dtype=float) * scale  # A or A/cm²
    ru_eff = reconcile_ru(ru, ru_unit, current_unit, area_cm2)
    factor_percent = clamp_factor_percent(factor_percent)
    ir_drop = (factor_percent / 100.0) * i_base * ru_eff  # volts
    return CorrectionResult(
        potential_corrected=e - ir_drop,
        ir_drop=ir_drop,
        ru=float(ru),
        factor_percent=factor_percent,
        current_unit=current_unit,
        ru_unit=ru_unit,
        ru_effective=ru_eff,
        area_cm2=float(area_cm2) if area_cm2 else None,
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
    raw_reverted_fraction: float  # same metric on raw data (noise baseline)
    over_compensated: bool
    message: str


def assess_correction(
    raw_potential: np.ndarray, corrected_potential: np.ndarray
) -> CorrectionAssessment:
    """Flag over-compensation by detecting fold-back in the corrected sweep."""
    e_raw = np.asarray(raw_potential, dtype=float)
    e_cor = np.asarray(corrected_potential, dtype=float)
    n = len(e_cor)
    # Sweep direction from the *net* travel of the whole record rather than
    # the two endpoints: an LSV that begins with a short approach leg running
    # the other way (a very common instrument export) would otherwise be
    # classified backwards, and the fold-back detector below -- which is
    # simply "does the potential ever reverse against this direction" -- would
    # then report ~100 % fold-back on a perfectly good correction.
    if n < 2:
        direction = 1
    else:
        steps = np.diff(e_raw)
        net = float(np.sum(steps))
        if net != 0:
            direction = 1 if net > 0 else -1
        else:  # pure round trip: fall back to the dominant step direction
            direction = 1 if float(np.sum(np.sign(steps))) >= 0 else -1

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
    ru_unit: str = RU_OHM,
    area_cm2: float | None = None,
) -> int:
    """Return the highest factor (%, ≤ 100) whose correction does not fold back.

    Scans candidate factors from high to low; the first that is not
    over-compensated is the recommendation. If every candidate folds back, the
    smallest is returned.
    """
    if candidates is None:
        candidates = (100, 95, 90, 85, 80, 70, 60, 50, 40, 30, 20, 10, 5)
    ordered = sorted({int(clamp_factor_percent(c)) for c in candidates}, reverse=True)
    for f in ordered:
        res = apply_ir_correction(
            potential, current, ru, f, current_unit, ru_unit, area_cm2
        )
        if not assess_correction(potential, res.potential_corrected).over_compensated:
            return f
    return ordered[-1]
