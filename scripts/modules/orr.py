"""RRDE (rotating ring-disk electrode) oxygen-reduction-reaction analysis.

Two complementary views of the same set of polarization curves:

- A single rotation rate (conventionally 1600 rpm) gives the onset potential
  E_onset, the half-wave potential E_1/2, and — after removing the
  mass-transport contribution — the Tafel slope of the kinetic region. Ring
  and disk current at that same rotation rate also give the electron-transfer
  number ``n`` and the peroxide yield directly, with no multi-rotation-rate
  fit required.
- Several rotation rates of the same sample let the disk/ring response be
  compared side by side (the mass-transport-limited current grows with the
  square root of rotation rate).

Electron-transfer number and peroxide yield (ring-disk method; Bard &
Faulkner, *Electrochemical Methods*, 2nd ed.):

    n = 4 |I_d| / (|I_d| + |I_r| / N)
    %H2O2 = 200 (|I_r| / N) / (|I_d| + |I_r| / N)

where ``I_d``, ``I_r`` are the disk and ring currents (or current densities —
the ratio is scale-invariant as long as both use the same units) and ``N`` is
the ring collection efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sweep

# Faraday constant, C/mol (CODATA).
FARADAY_C_PER_MOL = 96485.33212

# O2/electrolyte transport parameters at ~25 degC, commonly cited in the ORR
# RRDE/RDE literature (D: O2 diffusion coefficient, cm^2/s; nu: kinematic
# viscosity, cm^2/s; c: bulk O2 solubility, mol/cm^3). Approximate -- values
# vary somewhat by source/temperature/exact concentration; check against your
# own electrolyte when precision matters. See e.g. Zhou et al., *J. Mater.
# Chem. A* **2016**; Bard & Faulkner, *Electrochemical Methods*, 2nd ed.,
# Wiley, 2001, Table 9.3.1 and refs. therein.
ELECTROLYTE_PRESETS: dict[str, tuple[float, float, float]] = {
    "0.1 M KOH (alkaline)": (1.9e-5, 1.00e-2, 1.2e-6),
    "0.1 M HClO4 (acidic)": (1.93e-5, 1.009e-2, 1.117e-6),
    "0.5 M H2SO4 (acidic)": (1.4e-5, 1.03e-2, 1.1e-6),
}


# Below this fraction of the sweep's own peak |disk current|, the ring/disk
# ratio stops meaning anything: before the ORR onset the disk carries only
# capacitive/background current, so |Id| -> 0 while the ring keeps its own
# small background, and n and %H2O2 collapse onto their limits (n -> 0,
# %H2O2 -> 100) purely as an artifact of dividing by ~nothing. Masking that
# region to nan is what keeps a plot of %H2O2 from showing a confident-looking
# flat 100 % line across the entire pre-onset half of the sweep.
#
# 5 % is not an arbitrary cutoff: it is the same fraction of the limiting
# current the ORR literature conventionally uses to define the onset potential
# itself, so the masked region is exactly "before the reaction has started".
# On the bundled sample this suppresses all 159 pre-onset points (at 2 % four
# of them still leaked through, because the background sits right at that
# level).
MIN_DISK_FRACTION = 0.05


def _ring_disk_terms(disk_current, ring_current, collection_efficiency: float,
                     min_disk_fraction: float):
    """Shared ``(|Id|, |Ir|/N, valid_mask)`` for the two ring-disk formulas."""
    id_ = np.abs(np.asarray(disk_current, dtype=float))
    ir = np.abs(np.asarray(ring_current, dtype=float))
    if collection_efficiency <= 0:
        raise ValueError("Ring collection efficiency N must be positive.")
    ir_n = ir / float(collection_efficiency)
    denom = id_ + ir_n

    peak = float(np.nanmax(id_)) if np.isfinite(id_).any() else 0.0
    if peak <= 0:
        # No disk current anywhere: the ring signal alone carries no
        # information about how the disk's electrons were split, and the
        # formulas would confidently report n = 0 / 100 % H2O2 for every
        # point. A zero floor would let that through, since |Id| >= 0 always.
        return id_, ir_n, denom, np.zeros(len(id_), dtype=bool)
    floor = float(min_disk_fraction) * peak
    valid = np.isfinite(denom) & (denom > 0) & (id_ >= floor)
    return id_, ir_n, denom, valid


def electron_number(disk_current, ring_current,
                    collection_efficiency: float,
                    min_disk_fraction: float = MIN_DISK_FRACTION) -> np.ndarray:
    """RRDE electron-transfer number ``n = 4|Id| / (|Id| + |Ir|/N)``.

    Clipped to its physically meaningful range [0, 4] (the formula is already
    bounded there analytically; the clip only guards floating-point edge
    cases). Points where the disk current has not yet risen to
    ``min_disk_fraction`` of its own peak return ``nan`` rather than a
    meaningless number — see :data:`MIN_DISK_FRACTION`.
    """
    id_, ir_n, denom, valid = _ring_disk_terms(
        disk_current, ring_current, collection_efficiency, min_disk_fraction
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(valid, 4.0 * id_ / denom, np.nan)
    return np.clip(n, 0.0, 4.0)


def peroxide_percent(disk_current, ring_current,
                     collection_efficiency: float,
                     min_disk_fraction: float = MIN_DISK_FRACTION) -> np.ndarray:
    """Peroxide yield ``%H2O2 = 200(|Ir|/N) / (|Id| + |Ir|/N)``.

    Clipped to [0, 100]; points below ``min_disk_fraction`` of the peak disk
    current return ``nan`` (see :data:`MIN_DISK_FRACTION`) instead of pinning
    at exactly 100 % across the whole pre-onset region.
    """
    id_, ir_n, denom, valid = _ring_disk_terms(
        disk_current, ring_current, collection_efficiency, min_disk_fraction
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(valid, 200.0 * ir_n / denom, np.nan)
    return np.clip(pct, 0.0, 100.0)


def ring_disk_average(potential, disk_current, ring_current,
                      collection_efficiency: float,
                      window: tuple[float, float] | None = None,
                      min_disk_fraction: float = MIN_DISK_FRACTION):
    """Mean ``(n, %H2O2)`` over a potential window, the way ORR papers quote
    them.

    A single-point read-off at E1/2 is sensitive to exactly where E1/2 landed;
    the literature convention is an average across the diffusion-limited
    plateau (commonly ~0.2-0.6 V vs RHE). Returns ``(n_mean, pct_mean,
    n_points)``, with ``nan`` means when the window holds no valid points.
    """
    pot = np.asarray(potential, dtype=float)
    n_arr = electron_number(disk_current, ring_current, collection_efficiency,
                            min_disk_fraction)
    pct_arr = peroxide_percent(disk_current, ring_current, collection_efficiency,
                               min_disk_fraction)
    mask = np.isfinite(n_arr) & np.isfinite(pct_arr)
    if window is not None:
        lo, hi = sorted(window)
        mask &= (pot >= lo) & (pot <= hi)
    if not mask.any():
        return float("nan"), float("nan"), 0
    return float(np.mean(n_arr[mask])), float(np.mean(pct_arr[mask])), int(mask.sum())


@dataclass
class OnsetResult:
    """Onset / half-wave read-off from one rotation rate's polarization curve."""

    onset_potential: float
    half_wave_potential: float
    limiting_current: float  # signed plateau value used as j_d


def _smoothed_derivative(pot: np.ndarray, cur: np.ndarray, smooth_window: int) -> np.ndarray:
    """dI/dE of a lightly smoothed copy of ``cur`` over ``pot`` (both already
    oriented/aligned by the caller) — the ``smooth_window``-point moving
    average keeps raw measurement noise from creating a spurious sharp peak
    when :func:`onset_and_half_wave` looks for the steepest point."""
    w = max(1, int(smooth_window)) | 1  # force odd, for symmetric edge padding
    n = len(cur)
    if w > 1 and n >= w:
        pad = w // 2
        padded = np.pad(cur, (pad, pad), mode="edge")
        cur_smooth = np.convolve(padded, np.ones(w) / w, mode="valid")
    else:
        cur_smooth = cur
    # sweep.safe_gradient, not np.gradient: np.gradient divides by the
    # potential step, so any repeated potential divides by zero and any
    # direction reversal (an approach/vertex leg the caller did not strip)
    # divides by a negative step while the current keeps rising. On the
    # bundled ORR sample that alone produced a dI/dE spike ~26x the true
    # maximum, right where the E1/2 search looks for its peak.
    return sweep.safe_gradient(cur_smooth, pot)


def half_wave_derivative(potential, disk_current,
                         smooth_window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """The dI/dE curve :func:`onset_and_half_wave` searches for its steepest
    point to locate E1/2 — exposed separately so a caller can plot it and
    see directly where the reported E1/2 comes from (its peak, restricted to
    potentials at/past onset). Applies the same rest-potential-first
    reorientation ``onset_and_half_wave`` does, so the returned ``potential``
    is not necessarily in the file's original order.

    Returns ``(potential, deriv)``, same length and order.
    """
    pot, cur = sweep.clean_sweep(potential, disk_current)
    if len(pot) < 2:
        return pot, np.zeros_like(pot)
    pot, cur = sweep.orient_rest_first(pot, cur)
    return pot, _smoothed_derivative(pot, cur, smooth_window)


def _search_window_indices(pot: np.ndarray, onset_idx: int,
                           half_wave_search_range: tuple[float, float] | None) -> tuple[int, int]:
    """[lo, hi) index bounds for the E1/2 search: from onset to the sweep's
    end, narrowed to ``half_wave_search_range`` (V vs RHE) when at least one
    point falls inside it — falling back to the unrestricted onset-to-end
    span otherwise (e.g. a catalyst whose E1/2 genuinely falls outside the
    typical window, or a non-RHE reference scale)."""
    n = len(pot)
    if half_wave_search_range is None:
        return onset_idx, n
    lo, hi = sorted(half_wave_search_range)
    in_range = np.flatnonzero((pot[onset_idx:] >= lo) & (pot[onset_idx:] <= hi))
    if len(in_range) == 0:
        return onset_idx, n
    return onset_idx + int(in_range.min()), onset_idx + int(in_range.max()) + 1


def half_wave_potential_interpolated(potential, disk_current,
                                     baseline_frac: float = 0.1) -> float:
    """E1/2 via the **literature-standard** method used by most published
    ORR papers: linear interpolation for the potential where the disk
    current first crosses the midpoint between the pre-onset baseline and
    the mass-transport-limited plateau, j = (baseline + j_lim) / 2. Simple
    and directly comparable to reported literature values, but needs the
    plateau to be reasonably flat — see :func:`onset_and_half_wave`'s
    steepest-point (or :func:`half_wave_potential_second_derivative`'s
    inflection-point) methods for sweeps where it isn't.
    """
    pot, cur = sweep.clean_sweep(potential, disk_current)
    n = len(pot)
    if n < 5:
        raise ValueError("Need at least 5 points to locate E1/2.")
    pot, cur = sweep.orient_rest_first(pot, cur)
    tail_n = max(3, int(round(0.05 * n)))
    limiting_current = float(np.median(cur[-tail_n:]))
    baseline_n = max(3, int(round(baseline_frac * n)))
    baseline = float(np.median(cur[:baseline_n]))
    if limiting_current == baseline:
        raise ValueError("Current does not depart from baseline; cannot locate E1/2.")
    half_level = (baseline + limiting_current) / 2.0
    departure = (cur - half_level) * np.sign(limiting_current - baseline)
    crossings = np.flatnonzero(departure[baseline_n:] >= 0)
    if len(crossings) == 0:
        raise ValueError("Sweep never reaches the half-plateau current.")
    i = int(crossings[0]) + baseline_n
    if i == 0:
        return float(pot[0])
    x0, x1, y0, y1 = pot[i - 1], pot[i], cur[i - 1], cur[i]
    frac = 0.0 if y1 == y0 else (half_level - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def half_wave_potential_second_derivative(
    potential, disk_current, onset_idx: int, smooth_window: int = 5,
    half_wave_search_range: tuple[float, float] | None = (0.4, 0.8),
) -> float:
    """E1/2 via **double differentiation**: the S-curve's inflection point
    is, by definition, where d^2I/dE^2 crosses zero (the first derivative
    is at its own extremum there) — found directly as a zero-crossing,
    refined by linear interpolation between the two bracketing points for
    sub-grid precision, rather than by the coarser discrete-grid argmax of
    ``|dI/dE|`` that :func:`onset_and_half_wave` uses by default. In
    principle more precise on smooth data, but differentiating twice
    amplifies measurement noise well beyond what a single derivative does,
    so this is more sensitive to ``smooth_window`` matching the data's
    actual noise level. Falls back to the first-derivative argmax (within
    the same search window) if no sign change is found.
    """
    pot, cur = sweep.clean_sweep(potential, disk_current)
    if len(pot) < 5:
        raise ValueError("Need at least 5 points to locate E1/2.")
    pot, cur = sweep.orient_rest_first(pot, cur)
    deriv = _smoothed_derivative(pot, cur, smooth_window)
    second_deriv = sweep.safe_gradient(deriv, pot)
    lo_idx, hi_idx = _search_window_indices(pot, onset_idx, half_wave_search_range)
    window_sd = second_deriv[lo_idx:hi_idx]
    window_d = deriv[lo_idx:hi_idx]
    # Any sign change in d^2I/dE^2 marks an extremum of dI/dE (a max where
    # it goes + -> -, a min where - -> +; a sweep's dI/dE can peak either
    # way depending on current sign/orientation, so both count). A noisy
    # window can have several such wiggles, so pick whichever crossing sits
    # nearest the dominant |dI/dE| peak — the same peak the "steepest"
    # method finds — rather than just the first crossing encountered.
    sign_changes = np.flatnonzero(np.diff(np.sign(window_sd)) != 0)
    peak_rel = int(np.argmax(np.abs(window_d)))
    if len(sign_changes) == 0:
        return float(pot[lo_idx + peak_rel])
    nearest = int(sign_changes[np.argmin(np.abs(sign_changes - peak_rel))])
    i = lo_idx + nearest
    x0, x1, y0, y1 = pot[i], pot[i + 1], second_deriv[i], second_deriv[i + 1]
    frac = 0.0 if y1 == y0 else (0.0 - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


_HALF_WAVE_METHODS = ("steepest", "interpolated", "second_derivative")


def onset_and_half_wave(potential, disk_current,
                        baseline_frac: float = 0.1,
                        onset_frac_of_limit: float = 0.05,
                        smooth_window: int = 5,
                        half_wave_search_range: tuple[float, float] | None = (0.4, 0.8),
                        method: str = "steepest",
                        ) -> OnsetResult:
    """Locate the onset potential and half-wave potential of an RDE sweep.

    The limiting (mass-transport-plateau) current ``j_d`` is the median of
    the last 5 % of points (robust to plateau noise). The onset is the first
    point where the current departs from the flat pre-onset baseline (the
    first ``baseline_frac`` of points) by more than ``onset_frac_of_limit``
    of the baseline-to-plateau span — mirroring
    :func:`scripts.modules.tafel._onset_index`'s baseline-relative approach,
    generalised to work regardless of whether the sweep runs from positive to
    negative potential or the reverse.

    ``method`` picks how E1/2 is located — the three ways ORR researchers
    commonly report it:

    - ``"steepest"`` (default): the potential at the *steepest point* of a
      lightly smoothed copy of the disk current (max ``|dI/dE|``) — the
      inflection point of the S-shaped sweep. Doesn't depend on the plateau
      being perfectly flat, unlike ``"interpolated"``.
    - ``"interpolated"``: linear interpolation for the potential where the
      current crosses the literature-standard j = j_lim/2 — see
      :func:`half_wave_potential_interpolated`. The most common convention
      in published ORR papers when the plateau is well-defined.
    - ``"second_derivative"``: the inflection point found directly as a
      d^2I/dE^2 zero-crossing (double differentiation) rather than a
      discrete-grid argmax — see
      :func:`half_wave_potential_second_derivative`. In principle more
      precise, but more sensitive to noise/smoothing than ``"steepest"``.

    ``half_wave_search_range`` restricts the ``"steepest"``/
    ``"second_derivative"`` search to a scientifically typical ORR E1/2
    window (default 0.4-0.8 V vs RHE, spanning common non-precious-metal to
    good Pt-group catalysts) — without it, a sharp edge artifact right at
    onset (e.g. from smoothing a noisy departure from baseline) can
    occasionally out-score the true inflection point deeper into the sweep
    and report an E1/2 implausibly close to E_onset. Falls back to the
    unrestricted onset-to-plateau search if the sweep never enters the
    window (e.g. a very poor catalyst, or a reference scale where 0.4-0.8 V
    doesn't apply) rather than silently returning a wrong value.
    """
    if method not in _HALF_WAVE_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {_HALF_WAVE_METHODS}.")
    pot, cur = sweep.clean_sweep(potential, disk_current)
    n = len(pot)
    if n < 5:
        raise ValueError("Need at least 5 points to locate onset/E1/2.")

    # Orient the arrays so index 0 is the rest-potential end and -1 is deep
    # into the mass-transport plateau (largest |current|), regardless of the
    # sweep direction the file was recorded in. Compares a short block at each
    # end rather than the two single endpoints, so one noisy final sample
    # cannot flip the whole record; clean_sweep above has already dropped any
    # approach/vertex leg, which would otherwise put a non-plateau point last.
    pot, cur = sweep.orient_rest_first(pot, cur)

    tail_n = max(3, int(round(0.05 * n)))
    limiting_current = float(np.median(cur[-tail_n:]))

    baseline_n = max(3, int(round(baseline_frac * n)))
    baseline = float(np.median(cur[:baseline_n]))

    span = limiting_current - baseline
    if span == 0:
        raise ValueError(
            "Current does not depart from baseline; cannot locate onset."
        )
    threshold_mag = abs(onset_frac_of_limit * span)
    departure = (cur - baseline) * np.sign(span)  # positive & growing toward jd
    crossings = np.flatnonzero(departure[baseline_n:] >= threshold_mag)
    onset_idx = int(crossings[0]) + baseline_n if len(crossings) else baseline_n
    onset_potential = float(pot[onset_idx])

    if method == "interpolated":
        half_wave_potential = half_wave_potential_interpolated(pot, cur, baseline_frac)
    elif method == "second_derivative":
        half_wave_potential = half_wave_potential_second_derivative(
            pot, cur, onset_idx, smooth_window, half_wave_search_range
        )
    else:
        deriv = _smoothed_derivative(pot, cur, smooth_window)
        lo_idx, hi_idx = _search_window_indices(pot, onset_idx, half_wave_search_range)
        rel = int(np.argmax(np.abs(deriv[lo_idx:hi_idx])))
        half_wave_potential = float(pot[lo_idx + rel])

    return OnsetResult(
        onset_potential=onset_potential,
        half_wave_potential=half_wave_potential,
        limiting_current=limiting_current,
    )


# How close to the plateau j is still allowed to get. j_k = j*jd/(jd - j) has
# a pole at j = jd, so the last few percent before the plateau amplify ordinary
# measurement noise without limit: on the bundled ORR sample the raw formula
# returned |j_k| up to 169 against a |j_d| of 1.34 (a factor of 126), and 63
# points came out with the *opposite sign* to the current that produced them —
# because plateau noise pushed |j| past |jd| and flipped the denominator.
# Feeding those into a log10|j_k| Tafel fit is what makes an ORR Tafel slope
# depend on where the plateau noise happened to land.
MAX_PLATEAU_APPROACH = 0.95


def mass_transport_corrected_current(disk_current, limiting_current: float,
                                     max_plateau_approach: float = MAX_PLATEAU_APPROACH):
    """Kinetic current density ``j_k = j * j_d / (j_d - j)``.

    Removes the mass-transport contribution from a mixed kinetic-diffusion
    RDE curve so the low-overpotential (kinetic) region can be Tafel-fit;
    ``limiting_current`` is the plateau value from :func:`onset_and_half_wave`.

    Two guards keep the result physical (see :data:`MAX_PLATEAU_APPROACH`):
    points closer to the plateau than ``max_plateau_approach`` of ``j_d``, and
    points whose corrected current would come out with a different sign from
    ``j_d``, both return ``nan`` instead of a huge or sign-flipped value.
    ``nan`` (rather than a clipped number) is deliberate — it propagates
    through ``log10`` and is dropped by the caller's ``isfinite`` filter, so a
    guarded point is excluded from the Tafel fit rather than silently biasing
    it.
    """
    j = np.asarray(disk_current, dtype=float)
    jd = float(limiting_current)
    if jd == 0 or not np.isfinite(jd):
        return np.full_like(j, np.nan)

    denom = jd - j
    with np.errstate(divide="ignore", invalid="ignore"):
        jk = np.where(denom != 0, j * jd / denom, np.nan)

    # |j| must stay clear of the plateau, on the same side of zero as jd.
    frac = j / jd  # 0 at rest, 1 at the plateau
    usable = np.isfinite(jk) & (frac < float(max_plateau_approach))
    # A physical kinetic current runs the same direction as the limiting one.
    usable &= np.sign(jk) == np.sign(jd)
    return np.where(usable, jk, np.nan)


# --------------------------------------------------------------------------- #
# Koutecký–Levich analysis (multi-rotation-rate RDE)                          #
# --------------------------------------------------------------------------- #
# Koutecký, J.; Levich, V. G. *Zh. Fiz. Khim.* **1958**, *32*, 1565 (original
# derivation); Levich, V. G. *Physicochemical Hydrodynamics*, Prentice-Hall,
# 1962; standard textbook treatment: Bard, A. J.; Faulkner, L. R.
# *Electrochemical Methods: Fundamentals and Applications*, 2nd ed., Wiley,
# 2001, Ch. 9 ("Hydrodynamic Methods").
#
# At a fixed potential, the measured current density j has a kinetic
# (activation-controlled) and a mass-transport (diffusion-controlled)
# contribution in series:
#
#     1/j = 1/j_k + 1/(B * omega^0.5),   omega = 2*pi*rpm/60  (rad/s)
#
# so a plot of 1/j against omega^-0.5 at fixed potential is linear: the
# **intercept** gives the kinetic current density j_k, and the **slope**
# gives 1/B, where the Levich constant B is
#
#     B = 0.62 * n * F * D^(2/3) * nu^(-1/6) * C
#
# with n the electron-transfer number, F the Faraday constant, D the O2
# diffusion coefficient, nu the electrolyte's kinematic viscosity, and C the
# bulk O2 concentration -- so n follows once the electrolyte's transport
# parameters are known (see :data:`ELECTROLYTE_PRESETS`).


def angular_velocity(rpm) -> np.ndarray:
    """Rotation rate in rpm -> angular velocity omega (rad/s): omega = 2*pi*rpm/60."""
    return 2.0 * np.pi * np.asarray(rpm, dtype=float) / 60.0


# A K-L fit needs |j| well clear of zero: 1/j has a pole there, so at an
# analysis potential near the current's zero-crossing (just past the ORR
# onset) an ordinary noise-level current becomes an enormous 1/j that
# dominates the least-squares fit. On the bundled sample this produced
# n = 19.9 with R2 = 0.0007 at E = 0.70 V -- inside the app's own default
# 0.3-0.8 V window, so it was a value users actually saw. Points below this
# fraction of the largest |j| in the same fit are dropped.
MIN_KL_CURRENT_FRACTION = 0.05


@dataclass
class KoutieckyLevichFit:
    """One potential's Koutecký–Levich fit: 1/|j| vs omega^-0.5."""

    potential: float
    slope: float          # d(1/|j|) / d(omega^-0.5)
    intercept: float       # 1/|j_k|
    r_squared: float
    n_rotation_rates: int
    slope_stderr: float = float("nan")
    intercept_stderr: float = float("nan")

    @property
    def kinetic_current_density(self) -> float | None:
        """|j_k|, the mass-transport-free (kinetic) current density.

        Positive by construction: the fit runs on |j|, so j_k is a magnitude
        and does not inherit the cathodic sweep's negative sign. A fit whose
        intercept comes out **negative** has no j_k at all -- 1/j_k cannot be
        negative when j_k is a magnitude -- so this returns ``None`` rather
        than a negative kinetic current density. It happens on scattered
        Koutecký–Levich plots whose regression line crosses below the origin,
        exactly the fits :attr:`is_reliable` also rejects.
        """
        return None if self.intercept <= 0 else float(1.0 / self.intercept)

    @property
    def is_reliable(self) -> bool:
        """Whether this fit is worth quoting.

        A Koutecký–Levich plot with only three rotation rates and a poor R^2
        gives an ``n`` that is arithmetic, not evidence.
        """
        return (self.n_rotation_rates >= 3 and np.isfinite(self.r_squared)
                and self.r_squared >= 0.95 and self.intercept > 0)


def fit_koutecky_levich(rpm, j, min_points: int = 3,
                        min_current_fraction: float = MIN_KL_CURRENT_FRACTION,
                        ) -> KoutieckyLevichFit:
    """Fit ``1/|j|`` vs ``omega^-0.5`` at one potential across several
    rotation rates.

    ``rpm``/``j`` are parallel arrays (one entry per rotation rate).
    Non-finite, zero, and negligibly small ``|j|`` values are dropped (see
    :data:`MIN_KL_CURRENT_FRACTION`).

    The fit uses **|j|**, not the signed current. An ORR sweep records a
    cathodic (negative) current, which would otherwise make the slope,
    intercept and hence both ``j_k`` and ``n`` come out negative -- quantities
    the literature always reports as positive magnitudes, and which the
    caller then has to remember to ``abs()`` at every use. Taking the
    magnitude once, here, also keeps the fit meaningful for a data set whose
    sign convention is inconsistent between rotation rates.
    """
    rpm_arr = np.asarray(rpm, dtype=float)
    j_arr = np.abs(np.asarray(j, dtype=float))
    mask = np.isfinite(rpm_arr) & np.isfinite(j_arr) & (rpm_arr > 0) & (j_arr > 0)
    if mask.any():
        floor = float(min_current_fraction) * float(np.max(j_arr[mask]))
        mask &= j_arr > floor
    rpm_arr, j_arr = rpm_arr[mask], j_arr[mask]
    # The same rotation rate entered twice contributes two points at one x.
    # That is not an independent lever on the slope: three rates recorded as
    # 1600/1600/1600 would otherwise "fit" a line through a single x, where
    # sxx = 0 and the slope is whatever the residual noise dictates. Average
    # repeats instead, and let the min_points check below see the true count.
    if len(rpm_arr) and len(np.unique(rpm_arr)) != len(rpm_arr):
        uniq, inverse = np.unique(rpm_arr, return_inverse=True)
        j_arr = np.bincount(inverse, weights=j_arr) / np.bincount(inverse)
        rpm_arr = uniq
    if len(rpm_arr) < min_points:
        raise ValueError(
            f"Need at least {min_points} rotation rates for a Koutecký–Levich fit; "
            f"got {len(rpm_arr)}."
        )
    x = 1.0 / np.sqrt(angular_velocity(rpm_arr))
    y = 1.0 / j_arr
    k = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    sxx = float(np.sum((x - np.mean(x)) ** 2))
    if k > 2 and sxx > 0:
        resid_var = ss_res / (k - 2)
        slope_se = float(np.sqrt(resid_var / sxx))
        intercept_se = float(np.sqrt(resid_var * (1.0 / k + np.mean(x) ** 2 / sxx)))
    else:
        slope_se = intercept_se = float("nan")

    return KoutieckyLevichFit(
        potential=float("nan"), slope=float(slope), intercept=float(intercept),
        r_squared=r_squared, n_rotation_rates=k,
        slope_stderr=slope_se, intercept_stderr=intercept_se,
    )


def _validate_transport(diffusion_coeff_cm2_s: float,
                        kinematic_viscosity_cm2_s: float,
                        bulk_concentration_mol_cm3: float) -> None:
    """Reject non-physical O2 transport parameters before they reach the
    Levich prefactor.

    ``D**(2/3)`` and ``nu**(-1/6)`` are fractional powers: on a negative
    value Python raises a bare ``ValueError`` deep inside the arithmetic
    (or numpy returns ``nan``), neither of which tells the user that the
    diffusion coefficient they typed was the problem. A zero concentration
    or viscosity is equally meaningless, and zero viscosity divides by zero.
    """
    for name, value in (("Diffusion coefficient D", diffusion_coeff_cm2_s),
                        ("Kinematic viscosity nu", kinematic_viscosity_cm2_s),
                        ("Bulk O2 concentration C", bulk_concentration_mol_cm3)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number; got {value!r}.")


def levich_slope_to_n(kl_slope: float, diffusion_coeff_cm2_s: float,
                      kinematic_viscosity_cm2_s: float,
                      bulk_concentration_mol_cm3: float,
                      faraday_c_per_mol: float = FARADAY_C_PER_MOL) -> float | None:
    """Electron-transfer number ``n`` from a Koutecký–Levich slope
    (``1/B``) and the electrolyte's O2 transport parameters:
    ``n = B / (0.62 * F * D^(2/3) * nu^(-1/6) * C)``."""
    _validate_transport(diffusion_coeff_cm2_s, kinematic_viscosity_cm2_s,
                        bulk_concentration_mol_cm3)
    if kl_slope == 0 or not np.isfinite(kl_slope):
        return None
    b = 1.0 / kl_slope
    denom = (0.62 * faraday_c_per_mol * diffusion_coeff_cm2_s ** (2 / 3)
             * kinematic_viscosity_cm2_s ** (-1 / 6) * bulk_concentration_mol_cm3)
    # Magnitude: fit_koutecky_levich already works on |j|, so a negative n
    # here could only come from a caller passing a signed slope. n is an
    # electron count and is always reported positive.
    return None if denom == 0 else float(abs(b / denom))


def levich_current_density(n_electrons: float, rpm: float,
                           diffusion_coeff_cm2_s: float,
                           kinematic_viscosity_cm2_s: float,
                           bulk_concentration_mol_cm3: float,
                           faraday_c_per_mol: float = FARADAY_C_PER_MOL) -> float:
    """Ideal Levich (fully mass-transport-limited) current density in A/cm^2:
    ``j_lim = B * omega^0.5`` with ``B = 0.62 n F D^(2/3) nu^(-1/6) C``.

    The quickest sanity check on an RDE data set: if the measured plateau at
    1600 rpm is nowhere near the 4-electron value, the discrepancy is in the
    units, the electrode area, or the electrolyte parameters -- not in the fit.
    """
    _validate_transport(diffusion_coeff_cm2_s, kinematic_viscosity_cm2_s,
                        bulk_concentration_mol_cm3)
    b = (0.62 * float(n_electrons) * faraday_c_per_mol
         * diffusion_coeff_cm2_s ** (2 / 3)
         * kinematic_viscosity_cm2_s ** (-1 / 6) * bulk_concentration_mol_cm3)
    return float(b * np.sqrt(angular_velocity(rpm)))
