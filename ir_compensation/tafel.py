"""Tafel-slope extraction from a polarization curve (Potential vs Current).

The Tafel equation relates the electrode potential to the log of the current
in the activation-controlled region of a polarization curve:

    E = a + b * log10(|i|)

``b`` (the **Tafel slope**, V/decade or mV/decade) is obtained by a linear
regression of potential against log10(|current|) restricted to the linear
(kinetic) portion of the curve — mass-transport limitation at high current
and background/capacitive noise near the open-circuit potential both curve
away from this line and must be excluded.

``a`` is the potential-axis intercept at log10(|i|) = 0, i.e. the potential at
which the extrapolated Tafel line predicts unit current (1 in whatever unit
the data uses). If the input potential is an *overpotential* (referenced to
the equilibrium potential, eta = E - E_eq), the extrapolated current at
eta = 0 is the **exchange current** i0 = 10 ** (-a / b). When the input is a
raw (non-referenced) potential this "intercept current" is not physically the
exchange current, so callers should label it accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Reference-electrode -> RHE conversion                                       #
# --------------------------------------------------------------------------- #
# Standard potentials (V) of common reference electrodes vs. the normal
# hydrogen electrode (NHE / SHE) at 25 degC. Converting a measured potential
# to the reversible hydrogen electrode (RHE) scale removes the pH dependence
# of the reference and is the community convention for comparing HER/OER/ORR
# Tafel data across labs and electrolytes:
#
#     E(RHE) = E(measured) + E_ref_vs_NHE + NERNST_SLOPE_V_PER_PH * pH
#
# Values are widely tabulated (e.g. Bard & Faulkner, "Electrochemical
# Methods"); minor (<5 mV) variations exist between sources/temperatures.
NERNST_SLOPE_V_PER_PH = 0.05916  # V/pH at 25 degC: (R*T*ln10)/F

REFERENCE_ELECTRODES: dict[str, float] = {
    "SHE / NHE": 0.000,
    "SCE, saturated KCl": 0.241,
    "SCE, 1 M KCl": 0.280,
    "Ag/AgCl, saturated KCl": 0.197,
    "Ag/AgCl, 3 M KCl": 0.210,
    "Ag/AgCl, 3 M NaCl": 0.209,
    "Ag/AgCl, 1 M KCl": 0.235,
    "Hg/HgO, 1 M NaOH": 0.140,
    "Hg/Hg2SO4, saturated K2SO4": 0.640,
}


def to_rhe(potential: np.ndarray, e_ref_vs_nhe: float, ph: float) -> np.ndarray:
    """Convert ``potential`` (measured vs a reference electrode) to the RHE scale.

    ``e_ref_vs_nhe`` is the reference electrode's standard potential vs NHE
    (V); see :data:`REFERENCE_ELECTRODES` for common values. Pass ``0.0`` and
    ``ph=0`` (or simply skip the call) for data that is already reported vs
    RHE.
    """
    return (np.asarray(potential, dtype=float)
            + float(e_ref_vs_nhe) + NERNST_SLOPE_V_PER_PH * float(ph))


@dataclass
class TafelResult:
    """Outcome of a Tafel linear-region fit."""

    slope_v_per_dec: float   # dE / d(log10|i|), V/decade
    intercept_v: float       # potential at log10|i| = 0, V
    r_squared: float         # goodness of fit of the linear region
    fit_slice: tuple[int, int]  # (start, stop) indices used, in the caller's order

    @property
    def slope_mv_per_dec(self) -> float:
        return self.slope_v_per_dec * 1000.0

    @property
    def exchange_current(self) -> float | None:
        """Current at potential = 0 (extrapolated). See module docstring for
        when this is physically the exchange current i0."""
        if self.slope_v_per_dec == 0:
            return None
        return float(10.0 ** (-self.intercept_v / self.slope_v_per_dec))


def log_current(current: np.ndarray) -> np.ndarray:
    """Return log10(|current|), dropping non-positive/non-finite samples is
    the caller's responsibility (zero current has no logarithm)."""
    return np.log10(np.abs(np.asarray(current, dtype=float)))


def _r_squared(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _grow_from_onset(x: np.ndarray, y: np.ndarray, onset_idx: int,
                     min_points: int, r2_threshold: float, patience: int) -> int:
    """Grow a fit window forward from ``onset_idx`` while it stays linear.

    Extends the window one point at a time and keeps the running best stop
    index as long as R^2 >= ``r2_threshold``; gives up after ``patience``
    consecutive points fail to meet it (this tolerates a little noise while
    still stopping once real curvature — e.g. the mass-transport plateau —
    sets in).
    """
    n = len(x)
    stop = min(onset_idx + min_points, n)
    if stop - onset_idx < 3:
        return stop
    best_stop = stop
    bad_streak = 0
    for candidate_stop in range(stop, n + 1):
        xs, ys = x[onset_idx:candidate_stop], y[onset_idx:candidate_stop]
        if len(xs) < 3 or np.ptp(xs) == 0:
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        if _r_squared(xs, ys, slope, intercept) >= r2_threshold:
            best_stop = candidate_stop
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= patience:
                break
    return best_stop


def _best_r2_window(x: np.ndarray, y: np.ndarray, min_frac: float) -> tuple[int, int]:
    """Legacy fallback: scan candidate windows (coarse grid) and score each
    by R^2 * window_size, favouring windows that are both linear and wide."""
    n = len(x)
    min_size = max(4, int(round(min_frac * n)))
    step = max(1, n // 40)

    best_start, best_stop, best_score = 0, n, -np.inf
    for start in range(0, n - min_size + 1, step):
        for stop in range(start + min_size, n + 1, step):
            xs, ys = x[start:stop], y[start:stop]
            if len(xs) < 3 or np.ptp(xs) == 0:
                continue
            slope, intercept = np.polyfit(xs, ys, 1)
            r2 = _r_squared(xs, ys, slope, intercept)
            score = r2 * len(xs)
            if score > best_score:
                best_start, best_stop, best_score = start, stop, score
    return best_start, best_stop


def _onset_index(current: np.ndarray, baseline_frac: float,
                 onset_multiplier: float) -> int | None:
    """First index where ``|current|`` departs from the flat pre-onset
    baseline, estimated from the sweep's own first points (assumed to be
    near open-circuit / background capacitive current).

    A threshold relative to the *baseline* (rather than the sweep's overall
    max) is what makes this track the true onset even when the current
    later spans several more decades on its way to a mass-transport
    plateau — using the endpoint max would push "onset" deep into the
    exponential rise on any wide-dynamic-range sweep.
    """
    n = len(current)
    baseline_n = max(3, min(n // 4, int(round(baseline_frac * n))))
    baseline = current[:baseline_n]
    level = float(np.median(baseline))
    spread = float(np.std(baseline))
    threshold = max(level * onset_multiplier, level + 5.0 * spread,
                    level + 1e-12)
    crossings = np.flatnonzero(current[baseline_n:] >= threshold)
    return int(crossings[0]) + baseline_n if len(crossings) else None


def auto_tafel_range(potential: np.ndarray, log_i: np.ndarray,
                     current: np.ndarray | None = None,
                     baseline_frac: float = 0.1, onset_multiplier: float = 4.0,
                     r2_threshold: float = 0.99,
                     min_points: int = 5, patience: int = 4,
                     min_frac: float = 0.2) -> tuple[int, int]:
    """Auto-detect the linear Tafel (kinetic) region.

    When ``current`` is supplied, the region is chosen the way it would be
    by hand: starting close to the reaction **onset** — where ``|current|``
    first departs from the flat background/capacitive baseline seen at the
    start of the sweep (see :func:`_onset_index`) — and growing outward
    while the potential vs log|i| relationship stays linear (R^2 >=
    ``r2_threshold``), stopping once real curvature sets in (typically the
    mass-transport-limited plateau at high overpotential). This mirrors how
    a Tafel region is picked scientifically, independent of reaction type
    (HER/OER/ORR): it is defined by the onset and the extent of linearity,
    not by the sign of the current.

    Falls back to a coarse global best-R^2-times-width window search (the
    original heuristic) if ``current`` is omitted, if no clear onset is
    found, or if the onset-grown window is degenerately small.
    """
    x = np.asarray(log_i, dtype=float)
    y = np.asarray(potential, dtype=float)
    n = len(x)
    if n < 5:
        return 0, n

    if current is not None:
        abs_i = np.abs(np.asarray(current, dtype=float))
        onset_idx = _onset_index(abs_i, baseline_frac, onset_multiplier)
        if onset_idx is not None:
            stop = _grow_from_onset(x, y, onset_idx, min_points,
                                    r2_threshold, patience)
            if stop - onset_idx >= max(3, min_points):
                return onset_idx, stop

    return _best_r2_window(x, y, min_frac)


def fit_tafel(
    potential: np.ndarray, log_i: np.ndarray,
    start: int | None = None, stop: int | None = None,
) -> TafelResult:
    """Linear-fit ``potential`` vs ``log_i`` over ``[start, stop)``.

    Parameters
    ----------
    potential : electrode potential (or overpotential), V.
    log_i     : log10(|current|), same length as ``potential``.
    start, stop : index range of the linear region to fit; auto-detected if
                  omitted.
    """
    pot = np.asarray(potential, dtype=float)
    x = np.asarray(log_i, dtype=float)
    if start is None or stop is None:
        a0, a1 = auto_tafel_range(pot, x)
        start = a0 if start is None else start
        stop = a1 if stop is None else stop
    start = max(0, int(start))
    stop = min(len(pot), int(stop))
    if stop - start < 3:
        raise ValueError("Need at least 3 points to fit the Tafel region.")

    xs, ys = x[start:stop], pot[start:stop]
    if np.ptp(xs) == 0:
        raise ValueError("Selected region has no spread in log|current|.")

    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return TafelResult(
        slope_v_per_dec=float(slope),
        intercept_v=float(intercept),
        r_squared=float(r_squared),
        fit_slice=(start, stop),
    )


# --------------------------------------------------------------------------- #
# Mechanistic interpretation                                                  #
# --------------------------------------------------------------------------- #
# Canonical benchmark slopes (mV/decade, always reported as the positive
# magnitude used in the literature) for the rate-determining step of common
# multi-electron reactions. These are textbook reference points for
# interpretation, not universal constants — real catalysts often fall between
# them or shift with coverage/mechanism changes.
REACTION_REFERENCES: dict[str, tuple[tuple[float, str], ...]] = {
    "HER": (
        (30.0, "Volmer–Heyrovsky–Tafel mechanism, chemical "
               "(Tafel) recombination step rate-determining"),
        (40.0, "Volmer–Heyrovsky mechanism, electrochemical "
               "desorption (Heyrovsky) step rate-determining"),
        (120.0, "Volmer–Heyrovsky mechanism, initial discharge "
                "(Volmer) step rate-determining"),
    ),
    # HOR is the reverse of HER and shares its elementary steps, so the same
    # canonical slopes apply — read in the oxidation direction.
    "HOR": (
        (30.0, "Tafel–Volmer mechanism, dissociative H2 adsorption "
               "(Tafel) step rate-determining"),
        (40.0, "Heyrovsky–Volmer mechanism, electrochemical H2 oxidation "
               "(Heyrovsky) step rate-determining"),
        (120.0, "Volmer step (discharge of adsorbed H) rate-determining"),
    ),
    "OER": (
        (40.0, "favorable multi-electron-transfer kinetics with a "
               "chemical step rate-limiting"),
        (60.0, "chemical step involving a surface-bound intermediate is "
               "rate-limiting"),
        (120.0, "first electron-transfer step is rate-determining "
                "(Krasil'shchikov-type pathway)"),
    ),
    "ORR": (
        (60.0, "second electron transfer to an adsorbed intermediate is "
               "rate-determining (Temkin-like adsorption)"),
        (120.0, "first electron-transfer step is rate-determining "
                "(Langmuir-like adsorption)"),
    ),
    # The multi-electron small-molecule reductions below are conventionally
    # read through the same two limiting cases: an initial one-electron
    # transfer that is rate-determining (~2.303RT/alpha·F with alpha = 0.5,
    # i.e. ~118-120 mV/dec), or a fast pre-equilibrium electron transfer
    # followed by a rate-determining chemical step (~2.303RT/F, ~59-60
    # mV/dec). Intermediate values usually indicate mixed control or a
    # coverage-dependent mechanism rather than a clean assignment.
    "CO₂RR": (
        (60.0, "fast one-electron pre-equilibrium followed by a "
               "rate-determining chemical step (e.g. protonation of "
               "adsorbed *CO2⁻)"),
        (120.0, "initial single-electron transfer to CO2 (formation of the "
                "*CO2•⁻ radical anion) is rate-determining"),
    ),
    "N₂RR": (
        (60.0, "electron transfer in pre-equilibrium with a rate-determining "
               "chemical/protonation step of an adsorbed N-intermediate"),
        (120.0, "first electron transfer to adsorbed N2 (*N2 → *N2H) is "
                "rate-determining"),
    ),
    "NO₃RR": (
        (60.0, "fast initial electron transfer followed by a rate-determining "
               "chemical step (e.g. oxygen transfer, NO3⁻ → NO2⁻)"),
        (120.0, "first electron transfer to adsorbed nitrate is "
                "rate-determining"),
    ),
}


def nearest_reference(slope_mv_abs: float, reaction: str) -> tuple[float, str] | None:
    """Nearest canonical mechanistic benchmark for ``reaction``, or ``None``
    if ``reaction`` has no reference table (e.g. an unspecified/other
    reaction)."""
    refs = REACTION_REFERENCES.get(reaction)
    if not refs:
        return None
    return min(refs, key=lambda t: abs(t[0] - slope_mv_abs))


def analysis_paragraph(
    reaction: str, entries: list[tuple[str, float, float]]
) -> str:
    """Short prose summary of a batch of Tafel fits.

    ``entries`` is a list of ``(label, slope_mv_abs, r_squared)`` — the sign-
    corrected (positive) slope, as reported in the literature.
    """
    if not entries:
        return ""
    n = len(entries)
    slopes = [s for _, s, _ in entries]
    mean_slope = sum(slopes) / n
    lo_label, lo_slope, _ = min(entries, key=lambda e: e[1])
    hi_label, hi_slope, _ = max(entries, key=lambda e: e[1])
    rlabel = reaction if reaction != "Other / unspecified" else "the reaction"

    parts = []
    if n == 1:
        label, slope, r2 = entries[0]
        parts.append(
            f"The fitted Tafel slope for {label} is {slope:.0f} mV/dec "
            f"(R² = {r2:.3f}) for {rlabel}."
        )
    else:
        parts.append(
            f"Across the {n} samples analyzed for {rlabel}, the Tafel slope "
            f"ranges from {lo_slope:.0f} mV/dec ({lo_label}) to "
            f"{hi_slope:.0f} mV/dec ({hi_label}), averaging "
            f"{mean_slope:.0f} mV/dec."
        )
        parts.append(
            f"{lo_label} shows the lowest slope and therefore the most "
            "favorable (fastest) reaction kinetics among the samples "
            "compared, since a smaller Tafel slope means overpotential "
            "increases more slowly per decade of current."
        )

    ref = nearest_reference(lo_slope if n > 1 else slopes[0], reaction)
    if ref is not None:
        parts.append(
            f"A slope near {ref[0]:.0f} mV/dec is classically associated "
            f"with {ref[1]}, so the best-performing sample's kinetics are "
            "broadly consistent with that regime — a guide for mechanistic "
            "discussion, not a definitive assignment."
        )

    low_r2 = [(lbl, r2) for lbl, _, r2 in entries if r2 < 0.98]
    if low_r2:
        names = ", ".join(f"{lbl} (R²={r2:.3f})" for lbl, r2 in low_r2)
        parts.append(
            f"Fit quality is comparatively low for {names}; consider "
            "narrowing the fitted linear region to exclude curvature from "
            "near-OCP capacitive current or mass-transport limitation."
        )

    return " ".join(parts)
