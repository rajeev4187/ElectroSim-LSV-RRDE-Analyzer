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

from . import sweep

# --------------------------------------------------------------------------- #
# Reference-electrode -> RHE conversion                                       #
# --------------------------------------------------------------------------- #
# Standard potentials (V) of common reference electrodes vs. the normal
# hydrogen electrode (NHE / SHE) at 25 degC. Converting a measured potential
# to the reversible hydrogen electrode (RHE) scale removes the pH dependence
# of the reference and is the community convention for comparing HER/OER/ORR
# Tafel data across labs and electrolytes:
#
#     E(RHE) = E(measured) + E_ref_vs_NHE + nernst_slope(T) * pH
#
# Values are widely tabulated (e.g. Bard & Faulkner, "Electrochemical
# Methods"); minor (<5 mV) variations exist between sources/temperatures.

# Physical constants (CODATA 2018).
GAS_CONSTANT_J_PER_MOL_K = 8.314462618
FARADAY_C_PER_MOL = 96485.33212
STANDARD_TEMPERATURE_K = 298.15

# V/pH at 25 degC: (R*T*ln10)/F. Kept as a module constant for the common
# case; use nernst_slope() for any other temperature.
NERNST_SLOPE_V_PER_PH = 0.05916

REFERENCE_ELECTRODES: dict[str, float] = {
    "SHE / NHE": 0.000,
    "SCE, saturated KCl": 0.241,
    "SCE, 1 M KCl": 0.280,
    "Ag/AgCl, saturated KCl": 0.197,
    "Ag/AgCl, 3 M KCl": 0.210,
    "Ag/AgCl, 3 M NaCl": 0.209,
    "Ag/AgCl, 1 M KCl": 0.235,
    "Ag/AgCl, 0.1 M KCl": 0.288,
    "Hg/HgO, 1 M NaOH": 0.140,
    "Hg/HgO, 1 M KOH": 0.098,
    "Hg/HgO, 0.1 M NaOH": 0.165,
    "Hg/Hg2SO4, saturated K2SO4": 0.640,
    "Hg/Hg2SO4, 0.5 M H2SO4": 0.680,
    "Hg/HgSO4, 1 M H2SO4": 0.674,
}


def nernst_slope(temperature_k: float = STANDARD_TEMPERATURE_K) -> float:
    """Nernstian pH slope ``(R*T*ln10)/F`` in V/pH.

    59.16 mV/pH at 25 degC, but the RHE conversion is done at the temperature
    the experiment ran at — a 60 degC cell shifts the conversion by ~7 mV/pH,
    i.e. ~90 mV at pH 13, which is far from negligible when comparing onset
    potentials.
    """
    return (GAS_CONSTANT_J_PER_MOL_K * float(temperature_k) * np.log(10.0)
            / FARADAY_C_PER_MOL)


def to_rhe(potential: np.ndarray, e_ref_vs_nhe: float, ph: float,
           temperature_k: float = STANDARD_TEMPERATURE_K) -> np.ndarray:
    """Convert ``potential`` (measured vs a reference electrode) to the RHE scale.

    ``e_ref_vs_nhe`` is the reference electrode's standard potential vs NHE
    (V); see :data:`REFERENCE_ELECTRODES` for common values. Pass ``0.0`` and
    ``ph=0`` (or simply skip the call) for data that is already reported vs
    RHE. ``temperature_k`` sets the Nernstian pH slope (see
    :func:`nernst_slope`).
    """
    return (np.asarray(potential, dtype=float)
            + float(e_ref_vs_nhe) + nernst_slope(temperature_k) * float(ph))


def theoretical_tafel_slope_mv(alpha: float = 0.5, n_electrons: int = 1,
                               temperature_k: float = STANDARD_TEMPERATURE_K) -> float:
    """Butler-Volmer Tafel slope ``2.303 R T / (alpha n F)`` in mV/decade.

    The canonical 120 / 60 / 40 / 30 mV/dec benchmarks in
    :data:`REACTION_REFERENCES` are just this expression at 25 degC for
    ``alpha*n`` = 0.5, 1, 1.5 and 2 — worth computing explicitly when the
    experiment was not at 25 degC.
    """
    return 1000.0 * nernst_slope(temperature_k) / (float(alpha) * float(n_electrons))


@dataclass
class TafelResult:
    """Outcome of a Tafel linear-region fit."""

    slope_v_per_dec: float   # dE / d(log10|i|), V/decade
    intercept_v: float       # potential at log10|i| = 0, V
    r_squared: float         # goodness of fit of the linear region
    fit_slice: tuple[int, int]  # (start, stop) indices used, in the caller's order
    slope_stderr_v_per_dec: float = float("nan")  # 1-sigma uncertainty on the slope
    intercept_stderr_v: float = float("nan")
    n_points: int = 0
    decades: float = float("nan")  # span of log10|i| covered by the fit
    # True only when the fit's y-axis was *overpotential* rather than
    # electrode potential. Set by fit_tafel from its own argument; it cannot
    # be inferred from the numbers, since an overpotential axis and a
    # potential axis look identical once they are in the array.
    fitted_on_overpotential: bool = False

    @property
    def slope_mv_per_dec(self) -> float:
        return self.slope_v_per_dec * 1000.0

    @property
    def slope_stderr_mv_per_dec(self) -> float:
        return self.slope_stderr_v_per_dec * 1000.0

    @property
    def slope_ci95_mv_per_dec(self) -> float:
        """Half-width of the ~95 % confidence interval on the slope, mV/dec.

        A Tafel slope quoted without an uncertainty is not comparable between
        papers; 1.96 sigma is the usual shorthand (exact for large samples,
        mildly optimistic for the handful of points a narrow fit window
        holds).
        """
        return 1.96 * self.slope_stderr_mv_per_dec

    @property
    def exchange_current(self) -> float | None:
        """The exchange current density i0, or ``None`` when the fit cannot
        yield one.

        i0 is the current extrapolated to **zero overpotential**. This fit
        extrapolates to zero on whatever y-axis it was given, so the answer is
        i0 only when that axis was overpotential -- if it was electrode
        potential (V vs RHE, the usual case in this app), the same arithmetic
        returns the current extrapolated to 0 V vs RHE, which for an OER
        catalyst is a meaningless number many orders of magnitude from i0.
        Nothing in the numbers distinguishes the two cases, so rather than
        return a confident wrong value this returns ``None`` unless the caller
        declared an overpotential axis (``fit_tafel(..., overpotential=True)``).

        Also ``None`` for a flat (near-zero-slope) fit, where the
        extrapolation is undefined and the exponent would overflow a float.
        """
        if not self.fitted_on_overpotential:
            return None
        if self.slope_v_per_dec == 0:
            return None
        exponent = -self.intercept_v / self.slope_v_per_dec
        if abs(exponent) > 300:  # 10**308 ~ float64 max; stay well clear
            return None
        return float(10.0 ** exponent)

    @property
    def quality_warnings(self) -> list[str]:
        """Reasons this fit should not be quoted as-is.

        A high R^2 alone does not make a Tafel slope credible: the usual
        failure is a *short* window, where a handful of adjacent points on any
        smooth curve look perfectly linear. Convention in the electrocatalysis
        literature is to fit at least one decade of current; less than half a
        decade is not a Tafel region in any meaningful sense.
        """
        issues = []
        if np.isfinite(self.decades):
            if self.decades < 0.5:
                issues.append(
                    f"fitted over only {self.decades:.2f} decades of current "
                    "(convention is >= 1 decade) — the slope is not reliable"
                )
            elif self.decades < 1.0:
                issues.append(
                    f"fitted over {self.decades:.2f} decades (< 1 decade); "
                    "widen the range if the data allows"
                )
        if self.n_points < 5:
            issues.append(f"only {self.n_points} points in the fit")
        if self.r_squared < 0.98:
            issues.append(f"R² = {self.r_squared:.3f} — noticeable curvature in the window")
        return issues


def log_current(current: np.ndarray) -> np.ndarray:
    """Return log10(|current|).

    Exact zeros have no logarithm; they map to ``-inf`` here rather than
    raising, and callers filter with ``numpy.isfinite`` (an exactly-zero
    current sample carries no kinetic information anyway).
    """
    cur = np.abs(np.asarray(current, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(cur)


def _regression_prefix_sums(x: np.ndarray, y: np.ndarray):
    """Cumulative sums enabling O(1) least-squares stats for *any* slice.

    The window searches below evaluate a linear fit over O(n) nested windows.
    Calling ``numpy.polyfit`` per window makes that O(n^2) work — measurably
    slow (~1.7 s for a 4000-point sweep) and, because Streamlit re-runs the
    whole script on every widget interaction, it was paid again on every
    slider drag. Since ordinary least squares depends on the data only
    through the five sums below, prefix sums give every window's slope,
    intercept and R^2 in constant time and the whole scan in O(n).

    The sums are accumulated on **mean-centred** copies of x and y. A sum of
    squares is a textbook cancellation trap: the centred sum of squares is
    recovered as ``Sxx - (Sx)^2 / n``, a difference of two large numbers whose
    leading digits agree when the data sits far from the origin. Potentials in
    V vs RHE are small, but ``log10|i|`` for currents in A runs around -5 and
    an overpotential axis can be offset by volts, so the guard is cheap
    insurance. Centring is exact for the slope and R^2 (both are invariant
    under a shift) and the intercept is shifted back in
    :func:`_stats_from_sums`.
    """
    n = len(x)
    x0 = float(np.mean(x)) if n else 0.0
    y0 = float(np.mean(y)) if n else 0.0
    xc, yc = x - x0, y - y0
    zero = np.zeros(1)
    return (
        np.concatenate([zero, np.cumsum(xc)]),
        np.concatenate([zero, np.cumsum(yc)]),
        np.concatenate([zero, np.cumsum(xc * xc)]),
        np.concatenate([zero, np.cumsum(yc * yc)]),
        np.concatenate([zero, np.cumsum(xc * yc)]),
        n, x0, y0,
    )


def _stats_from_sums(sums, start, stop):
    """``(slope, intercept, r_squared)`` for ``[start, stop)`` from prefix
    sums. Returns ``None`` where the window is degenerate (fewer than 3
    points, or no spread in x)."""
    sx, sy, sxx, syy, sxy, _, x0, y0 = sums
    k = stop - start
    if k < 3:
        return None
    n = float(k)
    x_sum = sx[stop] - sx[start]
    y_sum = sy[stop] - sy[start]
    xx = sxx[stop] - sxx[start]
    yy = syy[stop] - syy[start]
    xy = sxy[stop] - sxy[start]
    sxx_c = xx - x_sum * x_sum / n
    if sxx_c <= 0:
        return None
    sxy_c = xy - x_sum * y_sum / n
    syy_c = yy - y_sum * y_sum / n
    slope = sxy_c / sxx_c
    # Undo the centring: the fit was made on (x - x0, y - y0), so a point at
    # x = 0 in the caller's own frame sits at -x0 in the centred one.
    intercept = (y_sum - slope * x_sum) / n + y0 - slope * x0
    if syy_c <= 0:
        return slope, intercept, 0.0
    r2 = float(np.clip((sxy_c * sxy_c) / (sxx_c * syy_c), 0.0, 1.0))
    return float(slope), float(intercept), r2


def _r_squared(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _grow_from_onset(x: np.ndarray, y: np.ndarray, onset_idx: int,
                     min_points: int, r2_threshold: float, patience: int,
                     e_eq: float | None = None,
                     max_overpotential: float | None = None) -> int:
    """Grow a fit window forward from ``onset_idx`` while it stays linear.

    Extends the window one point at a time and keeps the running best stop
    index as long as R^2 >= ``r2_threshold``; gives up after ``patience``
    consecutive points fail to meet it (this tolerates a little noise while
    still stopping once real curvature — e.g. the mass-transport plateau —
    sets in).

    When ``e_eq`` and ``max_overpotential`` are both given, growth also stops
    once the window reaches ``max_overpotential`` from ``e_eq`` (the
    reaction's equilibrium potential) — a genuine activation-controlled
    Tafel region rarely extends past a few hundred mV of overpotential, so
    this keeps a reaction-appropriate default even when a wider window
    still happens to pass the R^2 test (e.g. a long, accidentally-linear
    mass-transport-limited stretch).

    Runs in O(n) via :func:`_regression_prefix_sums`.
    """
    n = len(x)
    stop = min(onset_idx + min_points, n)
    if stop - onset_idx < 3:
        return stop

    # Hard cap from the overpotential limit, resolved up front so the scan
    # below is a single pass with no per-step branching.
    limit = n
    if e_eq is not None and max_overpotential is not None:
        beyond = np.flatnonzero(np.abs(y[onset_idx:] - e_eq) > max_overpotential)
        if len(beyond):
            limit = onset_idx + int(beyond[0])
        # The cap assumes the onset lies on the faradaic side of e_eq, within
        # max_overpotential of it. When it does not -- the reaction was
        # mis-assigned, or e_eq belongs to a different couple than the one
        # actually running -- the *first* point is already past the cap and
        # this would pin the window to its minimum width, quietly reporting a
        # slope fitted to five points. An inapplicable cap should not truncate
        # the fit: drop it and let the R^2 test alone decide the extent.
        if limit <= onset_idx:
            limit = n
        elif limit < stop:
            limit = stop

    sums = _regression_prefix_sums(x, y)
    best_stop = stop
    bad_streak = 0
    for candidate_stop in range(stop, limit + 1):
        stats = _stats_from_sums(sums, onset_idx, candidate_stop)
        if stats is None:
            continue
        if stats[2] >= r2_threshold:
            best_stop = candidate_stop
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= patience:
                break
    return best_stop


def _best_r2_window(x: np.ndarray, y: np.ndarray, min_frac: float) -> tuple[int, int]:
    """Fallback: scan candidate windows (coarse grid) and score each by
    R^2 * window_size, favouring windows that are both linear and wide.
    Uses the same O(1)-per-window prefix sums as :func:`_grow_from_onset`."""
    n = len(x)
    min_size = max(4, int(round(min_frac * n)))
    step = max(1, n // 40)
    sums = _regression_prefix_sums(x, y)

    best_start, best_stop, best_score = 0, n, -np.inf
    for start in range(0, n - min_size + 1, step):
        for stop in range(start + min_size, n + 1, step):
            stats = _stats_from_sums(sums, start, stop)
            if stats is None:
                continue
            score = stats[2] * (stop - start)
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

    ``current`` must already be ``|current|`` **oriented rest-end-first**;
    callers go through :func:`scripts.modules.sweep.orient_rest_first`.
    """
    n = len(current)
    if n < 4:
        return None
    baseline_n = max(3, min(n // 4, int(round(baseline_frac * n))))
    baseline = current[:baseline_n]
    level = float(np.median(baseline))
    spread = float(np.std(baseline))
    # Scale-aware floor: a baseline sitting at exactly zero (or at the noise
    # floor of a high-resolution ADC) makes the multiplicative and
    # spread-based terms vanish, and a bare +1e-12 absolute floor then fires
    # on the first quantisation step of a nanoamp-scale sweep. Anchor the
    # floor to the sweep's own dynamic range instead.
    reach = float(np.max(current)) - level
    threshold = max(level * onset_multiplier,
                    level + 5.0 * spread,
                    level + 1e-4 * max(reach, 0.0),
                    level + 1e-15)
    crossings = np.flatnonzero(current[baseline_n:] >= threshold)
    return int(crossings[0]) + baseline_n if len(crossings) else None


# --------------------------------------------------------------------------- #
# Onset potential & fixed-current-density benchmarks (HER/OER convention)     #
# --------------------------------------------------------------------------- #
# Equilibrium potential (V vs RHE) of common reactions, for converting a
# potential read off a curve into an overpotential eta = |E - E_eq|. HER/HOR
# share 0 V exactly (the RHE scale's own reference point); OER/ORR share
# 1.23 V (standard water oxidation/reduction potential). The others are
# approximate and product-dependent (see the app's own caption for caveats);
# reactions without a well-defined single E_eq are omitted, and callers
# should report the raw potential instead of an overpotential for those.
REACTION_E_EQ_V_RHE: dict[str, float] = {
    "HER": 0.0,
    "HOR": 0.0,
    "OER": 1.23,
    "ORR": 1.23,
    "NO₃RR": 0.69,   # -> NH3, approximate
    "N₂RR": 0.09,    # -> NH3, approximate
    "CO₂RR": -0.10,  # product-dependent, approximate
}


# --------------------------------------------------------------------------- #
# Automatic reaction identification                                           #
# --------------------------------------------------------------------------- #
@dataclass
class ReactionGuess:
    """A reaction inferred from the shape of a polarization curve."""

    reaction: str
    confidence: str        # "high" | "medium" | "low"
    reason: str            # human-readable justification, shown in the GUI

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.reaction} ({self.confidence}): {self.reason}"


def infer_reaction(potential: np.ndarray, current: np.ndarray,
                   default: str = "Other / unspecified") -> ReactionGuess:
    """Best guess at which reaction a polarization curve represents.

    Two facts identify most electrocatalysis sweeps, and both are already in
    the data — so defaulting every sample to one fixed reaction (which then
    silently drives the E_eq used for overpotentials, the mechanistic
    benchmark table, and the auto Tafel window's overpotential cap) throws
    away information the file itself carries:

    * **Sign of the faradaic current** — anodic (positive, oxidation) vs
      cathodic (negative, reduction), measured where the current is actually
      flowing rather than near the open-circuit crossover.
    * **Where the active window sits on the RHE scale** — water's
      thermodynamic window pins the common reactions apart: OER above 1.23 V,
      ORR just below it, HER/HOR at 0 V.

    ``potential`` must already be on the RHE scale; the caller converts
    first. Ambiguous cases (e.g. a cathodic sweep at strongly negative
    potentials, where HER and CO2RR/N2RR overlap and only the electrolyte
    distinguishes them) return the more common assignment with a
    ``"low"``/``"medium"`` confidence and a reason saying so, so the GUI can
    prompt rather than assert.
    """
    pot, cur = sweep.clean_sweep(potential, current)
    if len(pot) < 3:
        return ReactionGuess(default, "low", "too few points to judge")

    mag = np.abs(cur)
    if not np.isfinite(mag).any() or float(np.max(mag)) <= 0:
        return ReactionGuess(default, "low", "current is zero throughout")
    peak = float(np.nanmax(mag))

    # Sign is judged on the top decile of |current| only: near the
    # open-circuit crossover, noise and capacitive charging flip the sign at
    # random, so anywhere else risks reading the wrong direction entirely.
    strong = mag >= np.nanpercentile(mag, 90)
    cathodic = float(np.nanmedian(cur[strong])) < 0

    # The *window* uses a much lower bar — everything above 5 % of the peak.
    # The two thresholds answer different questions: an exponential Tafel
    # branch spends most of its potential range at small current, so the top
    # decile is only its extreme tip. Using that tip as the window put an HOR
    # sweep's apparent range up at the end of the scan instead of at the
    # H2/H+ equilibrium it starts from, and the reaction came out
    # unidentified. 5 % of peak captures the branch from its onset.
    active = mag >= 0.05 * peak
    if not active.any():
        active = strong
    e_active = pot[active]
    e_lo, e_hi = float(np.nanmin(e_active)), float(np.nanmax(e_active))
    e_mid = float(np.nanmedian(e_active))
    window = f"{e_lo:.2f}–{e_hi:.2f} V vs RHE"

    if cathodic:
        if e_mid > 0.95:
            # Above 1.23 V a cathodic current cannot be O₂ reduction driven by
            # overpotential at all, so the wording must not claim it sits below
            # the equilibrium.
            where = ("above" if e_mid > 1.23 else "just below")
            return ReactionGuess(
                "ORR", "medium",
                f"cathodic current at {window}, {where} the O₂/H₂O "
                "equilibrium (1.23 V) — but check this is not a reduction "
                "of the electrode itself",
            )
        if 0.15 <= e_mid <= 0.95:
            return ReactionGuess(
                "ORR", "high",
                f"cathodic current at {window} — the classic O₂-reduction "
                "window below 1.23 V and well above the HER onset",
            )
        if e_mid < -0.35:
            return ReactionGuess(
                "HER", "medium",
                f"cathodic current at {window}; that is HER territory, though "
                "CO₂RR/N₂RR/NO₃RR run at similar potentials — set it "
                "manually if your electrolyte says otherwise",
            )
        if e_mid >= 0.0:
            # A poor ORR catalyst can push its wave down into this band; do not
            # claim HER with high confidence where both are plausible.
            return ReactionGuess(
                "HER", "medium",
                f"cathodic current at {window}, just above the H⁺/H₂ "
                "equilibrium (0 V) — HER is the likeliest, but a poor ORR "
                "catalyst reaches this low too; set it manually if you are "
                "running O₂-saturated electrolyte",
            )
        return ReactionGuess(
            "HER", "high",
            f"cathodic current at {window}, at/below the H⁺/H₂ equilibrium (0 V)",
        )

    if e_mid >= 1.23:
        return ReactionGuess(
            "OER", "high",
            f"anodic current at {window}, above the O₂/H₂O equilibrium (1.23 V)",
        )
    # HOR is identified by where the current *starts*, not where it peaks:
    # it is the only anodic reaction already passing current at the H2/H+
    # equilibrium, and its branch then climbs well past 0 V. This test must
    # come before the "reaches 1.35 V" OER branch: an HOR sweep swept all the
    # way up past 1.4 V satisfies both, and the starting potential is the
    # discriminating evidence.
    if e_lo <= 0.25:
        return ReactionGuess(
            "HOR", "high",
            f"anodic current from {e_lo:.2f} V vs RHE — already flowing at "
            "the H₂/H⁺ equilibrium (0 V), which only H₂ oxidation does",
        )
    if e_hi >= 1.35:
        return ReactionGuess(
            "OER", "medium",
            f"anodic current reaching {e_hi:.2f} V vs RHE — past the OER onset",
        )
    return ReactionGuess(
        default, "low",
        f"anodic current at {window} — between the HOR and OER windows, "
        "which is where small-molecule oxidations (MOR/EOR/UOR) sit; "
        "pick the reaction manually",
    )


def onset_potential(potential: np.ndarray, current: np.ndarray,
                    baseline_frac: float = 0.1,
                    onset_multiplier: float = 4.0) -> float:
    """Potential at which ``|current|`` first departs from the flat
    pre-onset baseline — see :func:`_onset_index`.

    The arrays are cleaned and oriented first (approach/vertex leg stripped,
    rest-potential end put at index 0), so the answer no longer depends on
    which direction the instrument happened to record the sweep in. Reversing
    the input returns the same onset potential.
    """
    pot, cur = sweep.clean_sweep(potential, current)
    if len(pot) < 4:
        raise ValueError("Not enough points to locate an onset.")
    pot, cur = sweep.orient_rest_first(pot, cur)
    idx = _onset_index(np.abs(cur), baseline_frac, onset_multiplier)
    if idx is None:
        raise ValueError("No clear onset found (current never departs from baseline).")
    return float(pot[idx])


def potential_at_current_density(potential: np.ndarray, current_density: np.ndarray,
                                 target_j: float) -> float | None:
    """Potential at which ``|current_density|`` first reaches ``target_j``
    (same units as ``current_density``), by linear interpolation between the
    two bracketing points. ``None`` if the sweep never reaches ``target_j``.

    "First" means first *along the sweep, starting from the rest-potential
    end* — not first in file order. Those differ whenever the instrument
    recorded the scan plateau-end first, which would otherwise report the
    potential at which the curve *falls back* through j rather than the
    benchmark potential the literature means (a large error on any curve that
    overshoots the target).
    """
    pot, cur = sweep.clean_sweep(potential, current_density)
    if len(pot) == 0:
        return None
    pot, cur = sweep.orient_rest_first(pot, cur)
    j = np.abs(cur)
    target = abs(float(target_j))
    idx = np.flatnonzero(j >= target)
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(pot[0])
    x0, x1, y0, y1 = pot[i - 1], pot[i], j[i - 1], j[i]
    frac = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def overpotential_at_current_density(
    potential: np.ndarray, current_density: np.ndarray, target_j: float,
    reaction: str,
) -> tuple[float | None, bool]:
    """``(value, is_overpotential)`` at ``target_j`` for ``reaction``.

    Returns the overpotential ``eta = |E - E_eq|`` when the reaction has a
    well-defined equilibrium potential (see :data:`REACTION_E_EQ_V_RHE`), and
    otherwise the raw potential with ``is_overpotential=False`` so the caller
    can label the column honestly.
    """
    e_at_j = potential_at_current_density(potential, current_density, target_j)
    if e_at_j is None:
        return None, False
    e_eq = REACTION_E_EQ_V_RHE.get(reaction)
    if e_eq is None:
        return float(e_at_j), False
    return float(abs(e_at_j - e_eq)), True


def auto_tafel_range(potential: np.ndarray, log_i: np.ndarray,
                     current: np.ndarray | None = None,
                     baseline_frac: float = 0.1, onset_multiplier: float = 4.0,
                     r2_threshold: float = 0.99,
                     min_points: int = 5, patience: int = 4,
                     min_frac: float = 0.2,
                     e_eq: float | None = None,
                     max_overpotential: float = 0.35) -> tuple[int, int]:
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

    ``e_eq`` (the reaction's equilibrium potential vs RHE — see
    :data:`REACTION_E_EQ_V_RHE`) makes the growth reaction-aware: when
    given, the window is also capped at ``max_overpotential`` (default
    350 mV) of overpotential from ``e_eq``, since a genuine activation-
    controlled Tafel region rarely extends further regardless of how
    linear a wider window happens to score — this keeps the auto-detected
    default anchored to the selected reaction rather than to the raw
    current shape alone.

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
        # The onset detector reads its baseline off the *first* points, so it
        # only works on a sweep that starts at the rest potential. Detect the
        # orientation here and map the resulting window back to the caller's
        # own indexing, so a plateau-first recording (an equally common
        # instrument convention) auto-detects the same physical region rather
        # than failing outright.
        reversed_input = False
        if n >= 4:
            k = max(1, min(5, n // 4))
            if float(np.median(abs_i[-k:])) < float(np.median(abs_i[:k])):
                reversed_input = True
        if reversed_input:
            x, y, abs_i = x[::-1], y[::-1], abs_i[::-1]

        onset_idx = _onset_index(abs_i, baseline_frac, onset_multiplier)
        if onset_idx is not None:
            stop = _grow_from_onset(x, y, onset_idx, min_points,
                                    r2_threshold, patience,
                                    e_eq=e_eq, max_overpotential=max_overpotential)
            if stop - onset_idx >= max(3, min_points):
                if reversed_input:
                    return n - stop, n - onset_idx
                return onset_idx, stop
        if reversed_input:  # restore for the fallback scan below
            x, y = x[::-1], y[::-1]

    return _best_r2_window(x, y, min_frac)


def fit_tafel(
    potential: np.ndarray, log_i: np.ndarray,
    start: int | None = None, stop: int | None = None,
    current: np.ndarray | None = None, e_eq: float | None = None,
    overpotential: bool = False,
) -> TafelResult:
    """Linear-fit ``potential`` vs ``log_i`` over ``[start, stop)``.

    Parameters
    ----------
    potential : electrode potential (or overpotential), V.
    log_i     : log10(|current|), same length as ``potential``.
    start, stop : index range of the linear region to fit; auto-detected if
                  omitted.
    current, e_eq : forwarded to :func:`auto_tafel_range` when the range is
                  auto-detected. Without ``current`` that detector cannot
                  locate the onset and silently drops to its coarse
                  best-R^2 window scan, so a caller that has the signed
                  current (and the reaction's equilibrium potential) should
                  pass them to get the onset-anchored, overpotential-capped
                  region instead.

    overpotential : declare that ``potential`` is an overpotential axis, not
                  an electrode potential. Only then does
                  :attr:`TafelResult.exchange_current` return a value -- see
                  its docstring for why this cannot be detected automatically.

    ``fit_slice`` on the result is the **requested** window, not the fitted
    one: points that are non-finite in either array are dropped inside the
    window before the regression. ``n_points`` is the count that actually
    entered the fit, and ``decades`` the span they covered.
    """
    pot = np.asarray(potential, dtype=float)
    x = np.asarray(log_i, dtype=float)
    if start is None or stop is None:
        a0, a1 = auto_tafel_range(pot, x, current=current, e_eq=e_eq)
        start = a0 if start is None else start
        stop = a1 if stop is None else stop
    start = max(0, int(start))
    stop = min(len(pot), int(stop))
    if stop - start < 3:
        raise ValueError("Need at least 3 points to fit the Tafel region.")

    xs, ys = x[start:stop], pot[start:stop]
    # log10|i| is -inf wherever the current is exactly zero, and a single such
    # point silently poisons the whole least-squares fit (every sum becomes
    # nan). Drop non-finite pairs rather than returning a nan slope.
    finite = np.isfinite(xs) & np.isfinite(ys)
    if finite.sum() < 3:
        raise ValueError(
            "Fewer than 3 usable points in the selected region "
            "(zero or non-finite current)."
        )
    xs, ys = xs[finite], ys[finite]
    if np.ptp(xs) == 0:
        raise ValueError("Selected region has no spread in log|current|.")

    k = len(xs)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    resid = ys - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors of the OLS coefficients. A Tafel slope is only
    # comparable between studies when quoted with its uncertainty, and the
    # residual scatter is the honest estimate of it.
    sxx = float(np.sum((xs - np.mean(xs)) ** 2))
    if k > 2 and sxx > 0:
        resid_var = ss_res / (k - 2)
        slope_se = float(np.sqrt(resid_var / sxx))
        intercept_se = float(np.sqrt(resid_var * (1.0 / k + np.mean(xs) ** 2 / sxx)))
    else:
        slope_se = intercept_se = float("nan")

    return TafelResult(
        slope_v_per_dec=float(slope),
        intercept_v=float(intercept),
        r_squared=float(r_squared),
        fit_slice=(start, stop),
        slope_stderr_v_per_dec=slope_se,
        intercept_stderr_v=intercept_se,
        n_points=k,
        decades=float(np.ptp(xs)),
        fitted_on_overpotential=bool(overpotential),
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
