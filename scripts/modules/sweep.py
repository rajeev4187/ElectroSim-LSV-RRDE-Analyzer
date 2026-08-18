"""Sweep geometry: cleaning a raw voltammogram before any analysis touches it.

Every other module here assumes a *single, monotonic* potential sweep. Real
instrument exports very often are not one, and the mismatch silently corrupts
results rather than raising:

* **Approach / vertex legs.** A potentiostat usually ramps from its rest
  potential up (or down) to the sweep's vertex before the real scan begins, so
  the file opens with a short leg running the *opposite* way. The bundled
  ORR sample does exactly this — 21 points climbing to +1.41 V vs RHE before
  the cathodic scan starts. ``numpy.gradient`` over that fold reports a
  ``dI/dE`` spike ~26x the true maximum (the potential step reverses sign
  while the current does not), and ``numpy.interp`` — which requires strictly
  increasing x — silently returns garbage where the two legs overlap.
* **Recording direction.** Whether index 0 is the rest-potential end or the
  mass-transport plateau is a per-instrument (often per-method) convention.
  Baseline-relative detectors such as
  :func:`scripts.modules.tafel._onset_index` read their baseline off the first
  points, so they need the sweep oriented, not merely sorted.
* **Repeated / non-strictly-monotonic potentials.** Duplicate x values make
  ``numpy.gradient`` divide by zero and make ``numpy.interp`` order-dependent.

The helpers here normalise all three. They are deliberately conservative: a
clean single sweep passes through unchanged (same array objects where
possible), so applying them costs nothing on well-formed data.
"""

from __future__ import annotations

import numpy as np

# A leg shorter than this fraction of the record is treated as an instrument
# artifact (approach ramp, vertex overshoot) rather than a real sweep.
_MIN_SEGMENT_FRACTION = 0.15

# Points used at each end to decide the sweep's orientation. Comparing medians
# of a short block instead of the two single endpoints keeps one noisy final
# sample from flipping the whole record (the plateau end of a noisy sweep can
# easily dip below the rest end's stray capacitive current on a single point).
_ORIENT_BLOCK = 5


def _blocks(n: int) -> int:
    return max(1, min(_ORIENT_BLOCK, n // 4)) if n >= 4 else 1


def monotonic_segments(potential) -> list[tuple[int, int]]:
    """Split ``potential`` into maximal runs of one sweep direction.

    Returns a list of half-open ``(start, stop)`` index ranges, in the order
    they appear. Flat steps (``dE == 0``) attach to the run in progress rather
    than starting a new one, so ordinary quantisation of the potential axis
    does not shred the sweep into fragments.
    """
    pot = np.asarray(potential, dtype=float)
    n = len(pot)
    if n < 3:
        return [(0, n)]

    d = np.diff(pot)
    sign = np.sign(d)
    # Carry the last non-zero direction across flat steps.
    nz = sign != 0
    if not nz.any():
        return [(0, n)]
    idx = np.where(nz, np.arange(len(sign)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = sign[idx]
    # A record that *opens* with flat steps has no earlier direction to carry
    # forward, so those entries stay 0 and would read as a direction change
    # against the first real step -- splitting one sweep in two. Back-fill
    # them with the first non-zero direction instead.
    first = int(np.flatnonzero(nz)[0])
    filled[:first] = sign[first]

    breaks = np.flatnonzero(np.diff(filled) != 0) + 1
    bounds = [0, *(int(b) + 1 for b in breaks), n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def main_sweep_indices(potential, min_fraction: float = _MIN_SEGMENT_FRACTION):
    """Indices of the dominant (longest) monotonic leg of ``potential``.

    This is what strips a potentiostat's initial approach ramp or a vertex
    overshoot. Returns ``None`` when the record is already a single sweep, or
    when no leg is long enough to be confidently called the main one (in which
    case the caller should use the data unchanged rather than guess).
    """
    pot = np.asarray(potential, dtype=float)
    n = len(pot)
    segments = monotonic_segments(pot)
    if len(segments) < 2:
        return None
    start, stop = max(segments, key=lambda s: s[1] - s[0])
    if (stop - start) < max(3, int(round(min_fraction * n))):
        return None
    if (start, stop) == (0, n):
        return None
    # Keep the shared vertex point so the retained leg still spans the full
    # potential range it actually measured.
    return np.arange(max(0, start - 1), stop)


def clean_sweep(potential, *arrays, strip_approach: bool = True):
    """Return ``(potential, *arrays)`` reduced to one clean, usable sweep.

    Drops any row that is non-finite in the potential or in any companion
    array, then — when ``strip_approach`` — keeps only the dominant monotonic
    leg (see :func:`main_sweep_indices`). Companion arrays are sliced in
    lockstep, so disk/ring/current arrays stay aligned with the potential.
    """
    pot = np.asarray(potential, dtype=float)
    others = [np.asarray(a, dtype=float) if a is not None else None for a in arrays]

    mask = np.isfinite(pot)
    for a in others:
        if a is not None and len(a) == len(pot):
            mask &= np.isfinite(a)
    if not mask.all():
        pot = pot[mask]
        others = [a[mask] if (a is not None and len(a) == len(mask)) else a
                  for a in others]

    if strip_approach and len(pot) >= 3:
        keep = main_sweep_indices(pot)
        if keep is not None:
            pot = pot[keep]
            others = [a[keep] if (a is not None and len(a) > keep[-1]) else a
                      for a in others]

    return (pot, *others) if others else pot


def sweep_direction(potential) -> int:
    """``+1`` if the sweep advances toward more positive potential, ``-1``
    otherwise. Uses the net change across the record (robust to noise in any
    individual step)."""
    pot = np.asarray(potential, dtype=float)
    if len(pot) < 2:
        return 1
    return 1 if pot[-1] >= pot[0] else -1


def orient_rest_first(potential, current, *extra):
    """Reverse the arrays if needed so index 0 is the **rest-potential** end
    and index -1 sits deepest in the mass-transport plateau.

    Decided by comparing the *median* ``|current|`` of a short block at each
    end rather than the two single endpoints, so one noisy sample cannot flip
    the record. Returns the arrays in the same order they were passed.
    """
    pot = np.asarray(potential, dtype=float)
    cur = np.asarray(current, dtype=float)
    rest = [np.asarray(a, dtype=float) if a is not None else None for a in extra]
    n = len(cur)
    if n < 2:
        return (pot, cur, *rest) if rest else (pot, cur)

    k = _blocks(n)
    head = float(np.median(np.abs(cur[:k])))
    tail = float(np.median(np.abs(cur[-k:])))
    if tail < head:
        pot, cur = pot[::-1], cur[::-1]
        rest = [a[::-1] if a is not None else None for a in rest]
    return (pot, cur, *rest) if rest else (pot, cur)


def ascending_xy(x, y):
    """Return ``(x, y)`` sorted by ascending ``x`` with duplicate ``x`` values
    collapsed (their ``y`` averaged) — the precondition ``numpy.interp`` and
    ``numpy.gradient`` both require but neither enforces.

    ``numpy.interp`` on unsorted or duplicated x silently returns nonsense
    instead of raising, and ``numpy.gradient`` divides by a zero spacing, so
    every interpolation/differentiation in this package routes through here.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) == 0:
        return xa, ya
    order = np.argsort(xa, kind="stable")
    xs, ys = xa[order], ya[order]
    # Collapse runs of equal x to their mean y.
    new_run = np.ones(len(xs), dtype=bool)
    new_run[1:] = xs[1:] != xs[:-1]
    if new_run.all():
        return xs, ys
    group = np.cumsum(new_run) - 1
    counts = np.bincount(group)
    sums = np.bincount(group, weights=ys)
    return xs[new_run], sums / counts


def interp_at(x_query, x, y):
    """``numpy.interp`` with its sortedness precondition actually enforced
    (see :func:`ascending_xy`). Returns ``nan`` when there is nothing to
    interpolate from."""
    xs, ys = ascending_xy(x, y)
    if len(xs) == 0:
        return np.full(np.shape(x_query), np.nan)
    if len(xs) == 1:
        return np.full(np.shape(x_query), float(ys[0]))
    return np.interp(x_query, xs, ys)


def safe_gradient(y, x):
    """``dy/dx`` that tolerates unsorted/duplicated ``x``.

    Differentiates on a cleaned, strictly-increasing copy and maps the result
    back onto the caller's original ``x`` ordering, so the returned array
    lines up with the input point-for-point.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2:
        return np.zeros_like(ya)
    xs, ys = ascending_xy(xa, ya)
    if len(xs) < 2:
        return np.zeros_like(ya)
    grad = np.gradient(ys, xs)
    return np.interp(xa, xs, grad)
