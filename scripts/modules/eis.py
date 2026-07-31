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

    @property
    def r_low(self) -> float | None:
        """Low-frequency real-axis intercept (Ru + Rct)."""
        if self.rct is None:
            return None
        return self.ru + self.rct


def _abs_imag(z_imag: np.ndarray) -> np.ndarray:
    """Return |Z''| regardless of the file's sign convention."""
    return np.abs(z_imag)


def auto_arc_range(z_real: np.ndarray, z_imag: np.ndarray) -> tuple[int, int]:
    """Auto-detect the arc point range, excluding the low-frequency tail.

    Heuristic: data is ordered high -> low frequency. The kinetic arc descends to
    a local |Z''| minimum (the trough between the arc and the diffusion tail);
    points beyond that trough belong to the rising tail and are dropped. Returns
    a half-open ``(start, stop)`` index range.
    """
    zi = _abs_imag(z_imag)
    n = len(zi)
    if n < 4:
        return 0, n
    # Index of the global minimum of |Z''| marks the trough after the arc.
    trough = int(np.argmin(zi))
    # Keep a couple of points past the trough so the right intercept is constrained,
    # but never include the steep tail that follows.
    stop = min(trough + 3, n)
    if stop < 4:  # too few points -> fall back to using everything
        stop = n
    return 0, stop


def _fit_circle_algebraic(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Kasa algebraic circle fit. Returns ``(a, b, r)`` for centre (a, b), radius r.

    Solves the linear system from ``2*a*x + 2*b*y + c = x^2 + y^2`` in least
    squares, where ``c = r^2 - a^2 - b^2``.
    """
    a_mat = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    rhs = x**2 + y**2
    sol, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    a, b, c = sol
    r = float(np.sqrt(max(c + a**2 + b**2, 0.0)))
    return float(a), float(b), r


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

    return RuResult(
        ru=float(ru),
        rct=float(rct) if rct is not None else None,
        method="circle",
        center=(a, b),
        radius=r,
        rmse=rmse,
        arc_slice=(start, stop),
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
