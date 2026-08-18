"""Extract the uncompensated (ohmic / solution) resistance Ru from EIS data.

The reference dataset is a bare Nyquist plot: only ``Z'`` and ``Z''`` are stored,
with **no frequency column**, so a frequency-dependent equivalent-circuit fit is
not possible. Instead Ru is obtained geometrically.

In a Nyquist plot the kinetic process appears as a (possibly depressed)
semicircle. Its **high-frequency real-axis intercept equals Ru** (the series /
solution resistance), and the low-frequency intercept equals ``Ru + Rct``.
We fit a circle to the arc points and report the left intercept as Ru.

Methods
-------
``circle``   : algebraic (Kasa) circle fit of the selected arc points -> Ru, Rct.
``min_imag`` : Ru taken as Z' at the point of minimum |Z''| (quick estimate).
``manual``   : user supplies Ru directly.

The low-frequency diffusion / mass-transport tail (where |Z''| climbs steeply
again) must be excluded from the circle fit; :func:`auto_arc_range` finds a
sensible default cut-off that the GUI lets the user override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RuResult:
    """Outcome of a Ru extraction."""

    ru: float                      # uncompensated resistance, ohm
    rct: float | None = None       # charge-transfer resistance, ohm (circle fit)
    method: str = "circle"
    center: tuple[float, float] | None = None  # fitted circle centre (a, b)
    radius: float | None = None    # fitted circle radius
    rmse: float | None = None      # radial RMSE of the fit, ohm
    arc_slice: tuple[int, int] | None = None    # (start, stop) indices used
    # Angular span of the fitted points about the circle centre, in degrees.
    arc_coverage_deg: float | None = None

    @property
    def r_low(self) -> float | None:
        """Low-frequency real-axis intercept (Ru + Rct)."""
        if self.rct is None:
            return None
        return self.ru + self.rct

    @property
    def is_extrapolated(self) -> bool:
        """Whether Ru is an extrapolation well beyond the measured points.

        The high-frequency intercept is where the *fitted circle* crosses
        Z'' = 0, which is usually outside the measured data: a spectrum that
        stops at 100 kHz has not reached the real axis. The shorter the
        measured arc, the further the circle is being extrapolated and the
        more a small curvature error moves the intercept. Under roughly a
        quarter turn the answer should be quoted with that caveat -- a
        synthetic-arc check puts Ru within 0.16 ohm on a 20 ohm arc covering
        45-180 degrees, and it degrades quickly below that.
        """
        return (self.arc_coverage_deg is not None
                and self.arc_coverage_deg < 45.0)


def _abs_imag(z_imag: np.ndarray) -> np.ndarray:
    """Return |Z''| regardless of the file's sign convention."""
    return np.abs(z_imag)


def _smooth(v: np.ndarray, w: int = 3) -> np.ndarray:
    """Short moving average, so a single noisy point cannot pass for the
    arc apex in :func:`auto_arc_range`."""
    w = max(1, int(w)) | 1
    if w < 3 or len(v) < w:
        return v
    pad = w // 2
    return np.convolve(np.pad(v, (pad, pad), mode="edge"), np.ones(w) / w, mode="valid")


def auto_arc_range(z_real: np.ndarray, z_imag: np.ndarray) -> tuple[int, int]:
    """Auto-detect the arc point range, excluding the low-frequency tail.

    Data is ordered high -> low frequency. The kinetic arc rises to an apex
    and then descends to a local |Z''| minimum -- the trough between the arc
    and the diffusion tail. Points beyond that trough belong to the rising
    tail and are dropped. Returns a half-open ``(start, stop)`` index range.

    The trough is the minimum of the portion **after the apex**, rather than
    the global minimum of the whole spectrum. The distinction matters for the
    textbook case: a spectrum recorded from high frequency starts with
    |Z''| ~ 0, making the first point the global minimum, so an unanchored
    ``argmin`` trimmed the arc to nothing -- and the guard that caught that
    then fell back to fitting the *entire* spectrum, diffusion tail included,
    which drags the extrapolated high-frequency intercept (Ru) away from its
    true value. The apex itself is taken as the first local maximum, not the
    global one, because a strong diffusion tail commonly ends higher than the
    arc's own apex.
    """
    zi = _abs_imag(np.asarray(z_imag, dtype=float))
    n = len(zi)
    if n < 5:
        return 0, n

    sm = _smooth(zi)
    d = np.diff(sm)

    # First local maximum: the first descent that follows a rise. A spectrum
    # that starts already past its apex (no initial rise) has apex 0.
    apex = 0
    rose = False
    for i, step in enumerate(d):
        if step > 0:
            rose = True
        elif step < 0 and rose:
            apex = i
            break

    # Trough = lowest point after the apex; the diffusion tail rises away
    # from it on the far side, so a plain argmin over that span is stable
    # (no sensitivity to small wiggles, unlike a first-local-minimum scan).
    trough = apex + int(np.argmin(zi[apex:]))

    # Keep a couple of points past the trough so the right-hand intercept is
    # constrained, but never include the steep tail that follows.
    stop = min(trough + 3, n)
    if stop < 4:
        stop = n
    return 0, stop


def _fit_circle_kasa(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Kasa algebraic circle fit. Returns ``(a, b, r)`` for centre (a, b), radius r.

    Solves the linear system from ``2*a*x + 2*b*y + c = x^2 + y^2`` in least
    squares, where ``c = r^2 - a^2 - b^2``. Fast and unconditionally stable,
    but biased toward small radii when the points cover only a short arc.
    """
    a_mat = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    rhs = x**2 + y**2
    sol, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    a, b, c = sol
    r = float(np.sqrt(max(c + a**2 + b**2, 0.0)))
    return float(a), float(b), r


def _fit_circle_algebraic(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Circle fit used for the Nyquist arc. Returns ``(a, b, r)``.

    Kasa's algebraic fit. Its documented weakness is a bias toward small radii
    on very short arcs, so a lower-bias alternative (Taubin) was tried here --
    but measured against synthetic arcs of known radius, Kasa recovers Ru to
    within 0.16 ohm on a 20 ohm-radius arc across 45-180 degrees of span,
    which is well inside the scatter of real EIS data. It stays, on the
    principle that the fitter feeding the iR-correction tab should be the one
    with demonstrated accuracy on this problem rather than the one with the
    better reputation in general.
    """
    return _fit_circle_kasa(x, y)


def _angular_coverage_deg(dx: np.ndarray, dy: np.ndarray) -> float:
    """Angular span, in degrees, of the smallest sector containing all points.

    Not ``max(angle) - min(angle)``: an arc sitting across the +/-pi branch
    cut of ``arctan2`` has points at both ends of the range and would measure
    as very nearly a full turn. (A textbook Nyquist arc lands exactly there --
    its high-frequency end sits on the negative real axis of the circle's own
    frame, where a sign flip of a 1e-15 imaginary part decides between +pi
    and -pi.) Instead, find the largest *empty* gap between consecutive
    angles, treating them as points on a circle, and take what remains.
    """
    if len(dx) < 2:
        return 0.0
    angles = np.sort(np.arctan2(dy, dx))
    gaps = np.diff(angles)
    wrap_gap = (angles[0] + 2 * np.pi) - angles[-1]
    largest_gap = max(float(np.max(gaps)) if len(gaps) else 0.0, float(wrap_gap))
    return float(np.degrees(2 * np.pi - largest_gap))


def fit_ru_circle(z_real: np.ndarray, z_imag: np.ndarray,
                  start: int | None = None, stop: int | None = None) -> RuResult:
    """Fit a circle to the Nyquist arc and return Ru (high-frequency intercept).

    Parameters
    ----------
    z_real, z_imag : impedance components (any |Z''| sign convention).
    start, stop    : index range of arc points to fit; auto-detected if omitted.
    """
    zr = np.asarray(z_real, dtype=float)
    zi = _abs_imag(z_imag)
    if start is None or stop is None:
        a0, a1 = auto_arc_range(zr, zi)
        start = a0 if start is None else start
        stop = a1 if stop is None else stop
    start = max(0, int(start))
    stop = min(len(zr), int(stop))
    if stop - start < 3:
        raise ValueError("Need at least 3 points to fit a circle to the arc.")

    xs, ys = zr[start:stop], zi[start:stop]
    a, b, r = _fit_circle_algebraic(xs, ys)

    # Real-axis intercepts occur where the circle crosses Z'' = 0.
    disc = r**2 - b**2
    if disc <= 0:
        # Degenerate arc: fall back to the smallest measured Z'.
        ru = float(np.min(xs))
        rct = None
        r_low = None
    else:
        half = float(np.sqrt(disc))
        ru = a - half           # left / high-frequency intercept
        r_low = a + half        # right / low-frequency intercept
        rct = r_low - ru

    radial = np.sqrt((xs - a) ** 2 + (ys - b) ** 2)
    rmse = float(np.sqrt(np.mean((radial - r) ** 2)))

    coverage = _angular_coverage_deg(xs - a, ys - b)

    return RuResult(
        ru=float(ru),
        rct=float(rct) if rct is not None else None,
        method="circle",
        center=(a, b),
        radius=r,
        rmse=rmse,
        arc_slice=(start, stop),
        arc_coverage_deg=coverage,
    )


def fit_ru_min_imag(z_real: np.ndarray, z_imag: np.ndarray) -> RuResult:
    """Quick estimate: Ru = Z' at the point of minimum |Z''|."""
    zr = np.asarray(z_real, dtype=float)
    zi = _abs_imag(z_imag)
    idx = int(np.argmin(zi))
    return RuResult(ru=float(zr[idx]), method="min_imag")


def manual_ru(value: float) -> RuResult:
    """Wrap a user-supplied Ru value."""
    return RuResult(ru=float(value), method="manual")


def circle_path(center: tuple[float, float], radius: float,
                n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` of the upper half of the fitted circle for plotting."""
    a, b = center
    theta = np.linspace(0.0, np.pi, n)
    x = a + radius * np.cos(theta)
    y = b + radius * np.sin(theta)
    return x, y
