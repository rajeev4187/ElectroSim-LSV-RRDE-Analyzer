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


def electron_number(disk_current, ring_current,
                    collection_efficiency: float) -> np.ndarray:
    """RRDE electron-transfer number ``n = 4|Id| / (|Id| + |Ir|/N)``, clipped
    to its physically meaningful range [0, 4] (the formula is already
    bounded there analytically; the clip only guards floating-point edge
    cases at the extremes)."""
    id_ = np.abs(np.asarray(disk_current, dtype=float))
    ir_n = np.abs(np.asarray(ring_current, dtype=float)) / collection_efficiency
    denom = id_ + ir_n
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(denom > 0, 4.0 * id_ / denom, np.nan)
    return np.clip(n, 0.0, 4.0)


def peroxide_percent(disk_current, ring_current,
                     collection_efficiency: float) -> np.ndarray:
    """Peroxide yield ``%H2O2 = 200(|Ir|/N) / (|Id| + |Ir|/N)``, clipped to
    [0, 100] — near-zero disk current (e.g. close to open-circuit, before
    the ORR onset) can otherwise push the raw ratio above the physically
    meaningful 0-100 % range."""
    id_ = np.abs(np.asarray(disk_current, dtype=float))
    ir_n = np.abs(np.asarray(ring_current, dtype=float)) / collection_efficiency
    denom = id_ + ir_n
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(denom > 0, 200.0 * ir_n / denom, np.nan)
    return np.clip(pct, 0.0, 100.0)


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
    return np.gradient(cur_smooth, pot)


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
    pot = np.asarray(potential, dtype=float)
    cur = np.asarray(disk_current, dtype=float)
    if abs(cur[-1]) < abs(cur[0]):
        pot, cur = pot[::-1], cur[::-1]
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
    pot = np.asarray(potential, dtype=float)
    cur = np.asarray(disk_current, dtype=float)
    n = len(pot)
    if n < 5:
        raise ValueError("Need at least 5 points to locate E1/2.")
    if abs(cur[-1]) < abs(cur[0]):
        pot, cur = pot[::-1], cur[::-1]
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
    pot = np.asarray(potential, dtype=float)
    cur = np.asarray(disk_current, dtype=float)
    if abs(cur[-1]) < abs(cur[0]):
        pot, cur = pot[::-1], cur[::-1]
    deriv = _smoothed_derivative(pot, cur, smooth_window)
    second_deriv = np.gradient(deriv, pot)
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
    pot = np.asarray(potential, dtype=float)
    cur = np.asarray(disk_current, dtype=float)
    n = len(pot)
    if n < 5:
        raise ValueError("Need at least 5 points to locate onset/E1/2.")

    # Orient the arrays so index 0 is the rest-potential end and -1 is deep
    # into the mass-transport plateau (largest |current|), regardless of the
    # sweep direction the file was recorded in.
    if abs(cur[-1]) < abs(cur[0]):
        pot, cur = pot[::-1], cur[::-1]

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


def mass_transport_corrected_current(disk_current, limiting_current: float):
    """Kinetic current density ``j_k = j * j_d / (j_d - j)``.

    Removes the mass-transport contribution from a mixed kinetic-diffusion
    RDE curve so the low-overpotential (kinetic) region can be Tafel-fit;
    ``limiting_current`` is the plateau value from :func:`onset_and_half_wave`.
    """
    j = np.asarray(disk_current, dtype=float)
    jd = float(limiting_current)
    denom = jd - j
    with np.errstate(divide="ignore", invalid="ignore"):
        jk = np.where(denom != 0, j * jd / denom, np.nan)
    return jk


# --------------------------------------------------------------------------- #
# Koutecky-Levich analysis (multi-rotation-rate RDE)                          #
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


@dataclass
class KoutieckyLevichFit:
    """One potential's Koutecky-Levich fit: 1/j vs omega^-0.5."""

    potential: float
    slope: float          # d(1/j) / d(omega^-0.5)
    intercept: float       # 1/j_k
    r_squared: float
    n_rotation_rates: int

    @property
    def kinetic_current_density(self) -> float | None:
        """j_k, the mass-transport-free (kinetic) current density."""
        return None if self.intercept == 0 else float(1.0 / self.intercept)


def fit_koutecky_levich(rpm, j, min_points: int = 3) -> KoutieckyLevichFit:
    """Fit ``1/j`` vs ``omega^-0.5`` at one potential across several rotation
    rates. ``rpm``/``j`` are parallel arrays (one entry per rotation rate);
    non-finite or zero ``j`` values are dropped."""
    rpm_arr = np.asarray(rpm, dtype=float)
    j_arr = np.asarray(j, dtype=float)
    mask = np.isfinite(rpm_arr) & np.isfinite(j_arr) & (rpm_arr > 0) & (j_arr != 0)
    rpm_arr, j_arr = rpm_arr[mask], j_arr[mask]
    if len(rpm_arr) < min_points:
        raise ValueError(
            f"Need at least {min_points} rotation rates for a Koutecky-Levich fit; "
            f"got {len(rpm_arr)}."
        )
    x = 1.0 / np.sqrt(angular_velocity(rpm_arr))
    y = 1.0 / j_arr
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return KoutieckyLevichFit(
        potential=float("nan"), slope=float(slope), intercept=float(intercept),
        r_squared=r_squared, n_rotation_rates=len(rpm_arr),
    )


def levich_slope_to_n(kl_slope: float, diffusion_coeff_cm2_s: float,
                      kinematic_viscosity_cm2_s: float,
                      bulk_concentration_mol_cm3: float,
                      faraday_c_per_mol: float = FARADAY_C_PER_MOL) -> float | None:
    """Electron-transfer number ``n`` from a Koutecky-Levich slope
    (``1/B``) and the electrolyte's O2 transport parameters:
    ``n = B / (0.62 * F * D^(2/3) * nu^(-1/6) * C)``."""
    if kl_slope == 0:
        return None
    b = 1.0 / kl_slope
    denom = (0.62 * faraday_c_per_mol * diffusion_coeff_cm2_s ** (2 / 3)
             * kinematic_viscosity_cm2_s ** (-1 / 6) * bulk_concentration_mol_cm3)
    return None if denom == 0 else float(b / denom)
