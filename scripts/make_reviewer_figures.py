"""Reviewer-response figures built from ``sample-data/Obed data``.

Four PNGs, all drawn through the *same* styling helpers the Streamlit app
uses (``app.apply_plot_style``, ``app._BOX_AXIS_STYLE``, ``app._PALETTES``,
``app._font_stack``) so they are visually identical to what the GUI exports
rather than a look-alike re-implementation:

1. ``fig1_CA_HER_1M-KOH.png``      -- 1 h HER chronoamperometry, LFO vs
   5 wt% ReS2-LFO, with the current retention (% of the post-transient
   current) as its own panel.
2. ``fig2_CA_OER_1M-KOH.png``      -- the same for OER.
3. ``fig3_methanol_poisoning.png`` -- 2x2: CA retention and absolute CA for
   the two LFO-ReS2 runs vs 40 % Pt/C, plus each LFO-ReS2 run's RRDE disk
   polarization curve before and after the methanol spike.
4. ``fig4_RRDE_40pct-PtC.png``     -- 2x2 RRDE for 40 % Pt/C in 0.1 M KOH:
   disk and ring currents at 400-2400 rpm, and n / %H2O2 at 1600 rpm.

Run from the repository root::

    python scripts/make_reviewer_figures.py
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402  -- imported for its styling helpers only
from scripts.modules import orr, sweep  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "sample-data", "Obed data")
OUT = os.path.join(ROOT, "figures", "reviewer-response")

# --- Experimental constants -------------------------------------------------
# 5 mm glassy-carbon RRDE disk, the app's own default.
AREA_CM2 = 0.19625
# Ring collection efficiency, the app's default (see the ORR tab).
N_COLLECTION = 0.222
# E(RHE) = E(Hg/HgO) + E0 + 0.0592 pH, exactly the app's formula mode with
# its "Hg/HgO, 1 M KOH" entry (0.098 V vs NHE).
# Direct calibration offsets the user supplied for this dataset's own
# Hg/HgO reference electrode (the app's "Direct calibration offset" RHE
# mode) -- NOT the formula E_HgHgO + 0.098 + 0.0592*pH, which gives very
# slightly different numbers (0.927 V / 0.868 V) that do not match how this
# electrode was actually calibrated. The user's own framing: "an Hg/HgO
# electrode served as the reference electrode (calibrated to -0.945 and
# -0.841 V vs RHE for 1 M KOH and 0.1 M KOH, respectively)" -- i.e. RHE
# reads that many volts *below* Hg/HgO, equivalently E(RHE) = E(Hg/HgO) +
# 0.945 / +0.841.
RHE_1M_KOH = 0.945    # 1 M KOH  (HER/OER chronoamperometry, EIS)
RHE_01M_KOH = 0.841   # 0.1 M KOH (ORR / RRDE / poisoning, ORR EIS)

# The chronoamperograms all open with a double-layer charging spike an order
# of magnitude larger than the faradaic current. Normalising the retention to
# t = 0 would therefore report the decay of that spike, not of the catalyst,
# so the reference current is taken after it has died away.
T_REF_S = 60.0
# Methanol injection, read directly off the step in the 40 % Pt/C
# chronoamperogram (current collapses from -0.26 to +0.03 mA/cm^2 between
# t = 50.6 and 51.1 s). The LFO-ReS2 runs show no comparable step -- that
# absence is the methanol-tolerance result being reported.
T_MEOH_S = 50.8
# Pre-injection baseline window for the poisoning retention plot -- a stable
# stretch that is clear of both the opening double-layer transient and the
# methanol spike.
POISON_BASELINE = (30.0, T_MEOH_S - 1.0)

STYLE = app._default_style()
PALETTE = STYLE["palette"]
FONT_SIZE = STYLE["font_size"]
FAMILY = app._font_stack(STYLE["font_family"])
AXIS_FONT = {"family": FAMILY, "size": FONT_SIZE}
TITLE_FONT = {"family": FAMILY, "size": FONT_SIZE + 8}
SMALL_FONT = {"family": FAMILY, "size": max(9, round(FONT_SIZE * 0.55))}
PANEL_W = app._SCREEN_CANVAS_W       # one journal column of app canvas
PANEL_H = 460                        # apply_plot_style's default height
LINE_W = 3                           # the width the ORR tab's traces use


# --- Loading ----------------------------------------------------------------
def _read_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1]


def _child(parent: str, prefix: str) -> str:
    """The one entry of ``parent`` starting with ``prefix``.

    ``glob`` is unusable across this tree: every experiment directory name
    contains ``[...]``, which glob reads as a character class.
    """
    hits = sorted(n for n in os.listdir(parent) if n.startswith(prefix))
    if len(hits) != 1:
        raise LookupError(f"{prefix!r} in {parent}: expected 1 match, got {hits}")
    return os.path.join(parent, hits[0])


def load_ca(expt_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """``(t /s, j /mA cm^-2)`` from one ``CA Experiment [...]`` directory."""
    t, i_a = _read_xy(os.path.join(expt_dir, "Chronoamperogram",
                                   "Measured Current.csv"))
    return t, i_a * 1e3 / AREA_CM2


def load_rrde(expt_dir: str, rhe_offset: float):
    """``(E /V vs RHE, j_disk /mA cm^-2, i_ring /uA, j_ring /mA cm^-2, rpm)``
    for one RRDE run.

    ``i_ring`` (uA) is for plotting the ring trace on its own axis;
    ``j_ring`` -- the ring current divided by the *disk* area, same units
    and same divisor as ``j_disk`` -- is what :func:`orr.electron_number`
    and :func:`orr.peroxide_percent` must be given. Passing them ``j_disk``
    against a raw-uA ``i_ring`` (a 1000x unit mismatch on top of the missing
    area division) silently produces a physically meaningless n and
    %H2O2 -- the ratio is scale-invariant only when both currents already
    share one unit and one divisor, exactly the app's own note in the ORR
    tab about the ring "current density" being disk-area-normalised.

    Reduced to the dominant monotonic leg by :func:`sweep.clean_sweep`, the
    same call the app's ORR tab makes, so the short anodic approach segment
    every one of these files opens with is stripped rather than drawn as a
    spur across the plot.
    """
    volt = os.path.join(expt_dir, "Voltammogram")
    disk_name = _child(volt, "Disk Current")
    ring_name = _child(volt, "Ring Current")
    rpm = float(re.search(r"\((\d+) RPM\)", os.path.basename(disk_name)).group(1))
    e_raw, i_disk = _read_xy(disk_name)
    _, i_ring = _read_xy(ring_name)
    e, i_disk, i_ring = sweep.clean_sweep(e_raw, i_disk, i_ring)
    j_disk = i_disk * 1e3 / AREA_CM2
    j_ring = i_ring * 1e3 / AREA_CM2
    return e + rhe_offset, j_disk, i_ring * 1e6, j_ring, rpm


# --- Styling ----------------------------------------------------------------
def style_panels(fig, titles: list[tuple[str, str]], *, rows: int, cols: int,
                 height: int, width: int) -> None:
    """Apply the app's journal look to every panel of a subplot figure.

    :func:`app.apply_plot_style` handles a single-axis figure; this is the
    multi-panel equivalent, using the same font stack, the same
    ``_BOX_AXIS_STYLE`` frame and the same ``_journal_margin`` derivation so
    a panel here is indistinguishable from a figure the app exports.
    """
    margin = app._journal_margin(FONT_SIZE, titled=True)
    fig.update_layout(
        template="plotly_white", height=height, width=width,
        font=AXIS_FONT, paper_bgcolor="white", plot_bgcolor="white",
        margin={"l": margin["l"], "r": margin["r"],
                "t": margin["t"], "b": margin["b"]},
        legend={"font": SMALL_FONT, "bgcolor": "rgba(255,255,255,0.7)",
                "tracegroupgap": 18},
    )
    fig.update_xaxes(**app._BOX_AXIS_STYLE, tickfont=AXIS_FONT)
    fig.update_yaxes(**app._BOX_AXIS_STYLE, tickfont=AXIS_FONT)
    for k, (xt, yt) in enumerate(titles):
        r, c = divmod(k, cols)
        fig.update_xaxes(title={"text": xt, "font": TITLE_FONT}, row=r + 1, col=c + 1)
        fig.update_yaxes(title={"text": yt, "font": TITLE_FONT}, row=r + 1, col=c + 1)
    for note in fig.layout.annotations:
        # make_subplots' own panel titles ("(a) ...") -- size them with the
        # axis titles rather than Plotly's fixed 16 px.
        if note.text.startswith("("):
            note.font = {"family": FAMILY, "size": FONT_SIZE + 2, "color": "black"}


def panel_legend(fig, row: int, col: int, cols: int, *, position="top-left"):
    """A per-panel legend, anchored inside that panel's own domain.

    Plotly has one legend per figure, so each panel that needs its own gets a
    separate ``legend2``/``legend3``/... entry positioned on the panel's
    paper-space domain -- the same anchor points as the app's
    ``_LEGEND_ANCHORS``, just scaled into the subplot.
    """
    idx = (row - 1) * cols + col
    name = "legend" if idx == 1 else f"legend{idx}"
    xd = fig.get_subplot(row, col).xaxis.domain
    yd = fig.get_subplot(row, col).yaxis.domain
    anchor = app._LEGEND_ANCHORS[position]
    x = xd[0] + anchor["x"] * (xd[1] - xd[0])
    y = yd[0] + anchor["y"] * (yd[1] - yd[0])
    fig.update_layout(**{name: {
        "x": x, "y": y, "xanchor": anchor["xanchor"], "yanchor": anchor["yanchor"],
        "font": SMALL_FONT, "bgcolor": "rgba(255,255,255,0.75)",
        "bordercolor": "#d9d9d9", "borderwidth": 1, "tracegroupgap": 8,
    }})
    return name


def note(fig, row, col, text, *, x=0.5, y=0.06, color="black", size_delta=-6):
    """A small in-panel annotation (read-off values, %-change labels)."""
    fig.add_annotation(
        x=x, y=y, xref="x domain", yref="y domain", row=row, col=col,
        text=text, showarrow=False, align="center",
        font={"family": FAMILY, "size": max(10, FONT_SIZE + size_delta),
              "color": color},
        bgcolor="rgba(255,255,255,0.8)",
    )


def render_png(fig, path: str, dpi: int = 300) -> None:
    """Write ``fig`` to ``path`` as a PNG at ``dpi``, sidestepping kaleido's
    own browser auto-discovery.

    Plain ``fig.write_image`` calls into kaleido's ``choreographer``
    backend, which -- to decide whether it needs to download its own
    Chrome -- queries CPU architecture via a WMI call
    (``platform._wmi_query('CPU', 'Architecture')``) that intermittently
    raises `KeyError: 'Architecture'` on this machine (a flaky WMI
    provider, apparently worse under concurrent load). This crashed every
    multi-figure batch run partway through, after whichever figure
    happened to be mid-render when it fired -- not a one-off. `app.py`
    solved exactly this (see its own `_kaleido_render`/`_browser_paths`):
    handing kaleido an explicit, already-known-good browser executable
    path skips the auto-discovery code path entirely. Reused verbatim here
    rather than re-derived.
    """
    import kaleido

    opts = {"format": "png", "width": fig.layout.width,
           "height": fig.layout.height, "scale": dpi / 96}
    try:
        fig.write_image(path, **{k: v for k, v in opts.items() if k != "format"},
                        format=opts["format"])
        return
    except Exception:
        pass
    for browser_path in app._browser_paths():
        try:
            img_bytes = kaleido.calc_fig_sync(fig, opts=opts,
                                              kopts={"path": browser_path})
            with open(path, "wb") as f:
                f.write(img_bytes)
            return
        except Exception:
            continue
    raise RuntimeError(f"kaleido render failed for {path}: no working browser found")


def save(fig, name: str, dpi: int = 300) -> None:
    """Write ``name`` at ``dpi``, using the app's own kaleido settings.

    ``scale = dpi/96`` is the app's convention (see ``_render_export``): the
    figure's CSS size fixes its physical size and ``scale`` only buys
    resolution inside it.
    """
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    render_png(fig, path, dpi=dpi)
    w = fig.layout.width / 96 * 2.54
    print(f"  wrote {os.path.relpath(path, ROOT)}  "
          f"({fig.layout.width}x{fig.layout.height} px canvas = "
          f"{w:.1f} cm wide at {dpi} dpi)")


def retention(t: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(t, 100 |j|/|j_ref|)`` from :data:`T_REF_S` onward.

    In magnitude terms, not signed: HER/OER current is a single-sign
    quantity (cathodic for HER, anodic for OER), so "retention" here means
    how much of the |current| remains, growing above 100% if it actually
    increases over the hold rather than decaying. Using the signed ratio
    instead would still give the same *number* here (both j and j_ref share
    one sign for a plain hold), but stating it as |j|/|j_ref| makes that
    explicit rather than leaving the reader to infer it from a sign that
    happens to cancel.
    """
    k = int(np.searchsorted(t, T_REF_S))
    j_ref = float(np.median(np.abs(j[k:k + 50])))
    return t[k:], 100.0 * np.abs(j[k:]) / j_ref


def retention_from_window(t: np.ndarray, j: np.ndarray,
                          t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """``(t, 100 j/j_ref)`` over the *whole* trace, ``j_ref`` the median
    current in the ``[t0, t1]`` window.

    Used for the poisoning CA, where the reference has to sit on the stable
    plateau *before* the methanol spike (:data:`T_MEOH_S`) rather than at a
    fixed post-transient time -- :func:`retention`'s ``T_REF_S`` (60 s) falls
    a few seconds *after* the spike on the 40 % Pt/C run, which is what
    produced a nonsensical >1000 % retention when tried here.
    """
    mask = (t >= t0) & (t <= t1)
    j_ref = float(np.median(j[mask]))
    return t, 100.0 * j / j_ref


# --- Figures 1 & 2: HER / OER chronoamperometry -----------------------------
CHRONO_DIR = os.path.join(DATA, "20260730", "Chronoamperometry HER-OER")
CHRONO_RUNS = {
    "HER": [
        ("LaFeO₃", "LFO HER Chrono [2026-JUL-31 1631 #1]", -1.500),
        ("5 wt% ReS₂–LaFeO₃", "5ReS2-LFO HER Chrono [2026-JUL-31 1031 #1]", -1.500),
    ],
    "OER": [
        ("LaFeO₃", "LFO OER Chrono [2026-JUL-31 1031 #1]", 0.900),
        ("5 wt% ReS₂–LaFeO₃", "5ReS2-LFO OER Chrono [2026-JUL-31 1031 #1]", 0.900),
    ],
}


def figure_chrono(reaction: str, filename: str) -> None:
    print(f"{reaction} chronoamperometry:")
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.16,
        subplot_titles=(f"(a) {reaction} chronoamperogram",
                        "(b) Current retention"),
    )
    for k, (label, folder, e_applied) in enumerate(CHRONO_RUNS[reaction]):
        color = PALETTE[k % len(PALETTE)]
        t, j = load_ca(_child(os.path.join(CHRONO_DIR, folder), "CA Experiment"))
        t_h = t / 3600.0
        fig.add_trace(go.Scatter(
            x=t_h, y=j, mode="lines", name=label, legendgroup=label,
            line={"color": color, "width": LINE_W},
            hovertemplate="%{x:.3f} h, %{y:.3f} mA cm⁻²<extra>" + label + "</extra>",
        ), row=1, col=1)

        tr, pct = retention(t, j)
        fig.add_trace(go.Scatter(
            x=tr / 3600.0, y=pct, mode="lines", name=label, legendgroup=label,
            showlegend=False, line={"color": color, "width": LINE_W},
            hovertemplate="%{x:.3f} h, %{y:.1f} %<extra>" + label + "</extra>",
        ), row=1, col=2)
        end = float(np.median(pct[-50:]))
        print(f"  {label:22s} E = {e_applied:+.3f} V vs Hg/HgO "
              f"({e_applied + RHE_1M_KOH:+.3f} V vs RHE); "
              f"j(60 s) = {j[int(np.searchsorted(t, T_REF_S))]:+.4f} → "
              f"j(1 h) = {j[-1]:+.4f} mA cm⁻²; retention {end:.0f} %")
        fig.add_annotation(
            x=1.0, y=end, xref="x2", yref="y2", text=f"<b>{end:.0f} %</b>",
            showarrow=False, xanchor="right",
            yshift=14 if end >= 100 else -14,
            font={"family": FAMILY, "size": FONT_SIZE - 4, "color": color},
            bgcolor="rgba(255,255,255,0.85)",
        )

    fig.add_hline(y=100, line={"color": "#7f7f7f", "width": 2, "dash": "dot"},
                  row=1, col=2)

    style_panels(
        fig,
        [("Time (h)", "Current density (mA cm⁻²)"),
         ("Time (h)", "Current retention (%)")],
        rows=1, cols=2, height=PANEL_H + 60, width=PANEL_W * 2,
    )
    lg = panel_legend(fig, 1, 1, 2, position="bottom-left"
                      if reaction == "HER" else "top-right")
    for tr_ in fig.data:
        tr_.legend = lg
    save(fig, filename)


# --- Figure 3: methanol poisoning -------------------------------------------
# Only one LFO-ReS2 run is used: "POISON TEST" (the other candidate) gives an
# ORR current that *increases* after methanol addition and shows a sharp,
# unphysical single-point spike in its post-poisoning RRDE sweep -- both are
# signs of a bad run, not of real chemistry, and the sibling "POISON TEST
# RES2" folder is a clean repeat of the same 10% ReS2-LaFeO3 sample showing
# the expected *decrease*. See PROGRESS.md for the numbers from the dropped
# run. There is no separate ReS2-only (LaFeO3-free) methanol-poisoning
# dataset anywhere in sample-data/Obed data -- both poisoning folders hold
# "ORR CA 10 WT % LFO-RES2" -- so no ReS2 curve is added here; flag this to
# the user rather than substituting something else.
POISON_CA = [
    ("40 wt% Pt/C", os.path.join(DATA, "PtC", "Archive [2024-APR-03 1614 #1]"), -0.250),
    ("10% ReS₂–LaFeO₃",
     os.path.join(DATA, "POISON TEST RES2", "ORR CA 10 WT % LFO-RES2"), -0.299),
]
# Only the sweep segment worth showing: 0.9 V vs RHE (just past onset, where
# Just past onset down to the sweep's own lower end -- excludes the flat
# double-layer region above onset, where the raw data turns to wild noise
# (single points jumping between 0.4 and 2.5 mA/cm^2 in the raw CSV) above
# +0.047 V vs Hg/HgO. Expressed in the *raw* Hg/HgO scale and converted at
# use time rather than hard-coded in RHE terms, so it stays correct if the
# calibration offset above is ever revised (it already has been once).
POISON_RRDE_RAW_RANGE_HG_HGO = (-0.79, 0.04)
POISON_RRDE_XRANGE = [POISON_RRDE_RAW_RANGE_HG_HGO[0] + RHE_01M_KOH,
                      POISON_RRDE_RAW_RANGE_HG_HGO[1] + RHE_01M_KOH]
POISON_LSV_LABEL = "(c) 10% ReS₂–LaFeO₃<br>RRDE: before / after CH₃OH"
POISON_LSV_BEFORE = os.path.join(DATA, "POISON TEST RES2", "ORR CA 10 WT % LFO-RES2",
                                 "RRDE Experiment [2025-JAN-16 1334 #20]")
POISON_LSV_AFTER = os.path.join(DATA, "POISON TEST RES2", "ORR CA 10 WT % LFO-RES2",
                                "RRDE Experiment [2025-JAN-16 1344 #24]")
# Only the first 120 s of the CA hold is shown (methanol goes in at ~51 s;
# the response has settled well before 120 s and the longer tail some of
# these files carry adds nothing).
POISON_T_MAX_S = 120.0


def _ca_dir(parent: str) -> str:
    """The poisoning CA inside ``parent`` -- the short (~2 min) run, not the
    1 h stability run some of these folders also hold."""
    hits = []
    for name in sorted(os.listdir(parent)):
        if not name.startswith("CA Experiment"):
            continue
        path = os.path.join(parent, name)
        t, _ = _read_xy(os.path.join(path, "Chronoamperogram",
                                     "Measured Current.csv"))
        if t[-1] < 600:
            hits.append((t[-1], path))
    if len(hits) != 1:
        raise LookupError(f"expected 1 short CA run under {parent}, got {hits}")
    return hits[0][1]


def figure_poisoning(filename: str) -> None:
    print("Methanol poisoning:")
    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.09,
        subplot_titles=("(a) Current retention on CH₃OH addition",
                        "(b) Absolute CA response", POISON_LSV_LABEL),
    )

    for k, (label, folder, e_hold) in enumerate(POISON_CA):
        color = PALETTE[k % len(PALETTE)]
        # Every one of these folders holds exactly one short CA run; the
        # 1 h stability runs that sit beside some of them are filtered out
        # by the <600 s test in _ca_dir.
        t, j = load_ca(_ca_dir(folder))
        keep = t <= POISON_T_MAX_S
        t, j = t[keep], j[keep]
        _, pct = retention_from_window(t, j, *POISON_BASELINE)
        fig.add_trace(go.Scatter(
            x=t, y=pct, mode="lines", name=label, legendgroup=label,
            line={"color": color, "width": LINE_W},
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=j, mode="lines", name=label, legendgroup=label,
            line={"color": color, "width": LINE_W},
        ), row=1, col=2)
        j_base = float(np.median(j[(t >= POISON_BASELINE[0]) & (t <= POISON_BASELINE[1])]))
        end = float(np.median(pct[-20:]))
        print(f"  {label:20s} E = {e_hold:+.3f} V vs Hg/HgO; "
              f"j(baseline) = {j_base:+.4f} → j(end) = {j[-1]:+.4f} "
              f"mA cm⁻²; retention {end:.0f} %")

    for c in (1, 2):
        fig.add_vline(x=T_MEOH_S, line={"color": "#404040", "width": 2,
                                        "dash": "dash"}, row=1, col=c)
    fig.add_annotation(
        x=T_MEOH_S, y=1.0, xref="x", yref="y domain", row=1, col=1,
        text="CH₃OH", showarrow=False, xanchor="left", xshift=6, yshift=-4,
        font={"family": FAMILY, "size": FONT_SIZE - 6, "color": "#404040"},
    )
    fig.add_hline(y=100, line={"color": "#7f7f7f", "width": 2, "dash": "dot"},
                  row=1, col=1)
    fig.add_hline(y=0, line={"color": "#7f7f7f", "width": 1.5}, row=1, col=2)

    # --- (c): disk polarization before and after the spike, 10% ReS2-LaFeO3
    j_before = j_after = None
    for m, (tag, path) in enumerate((("before CH₃OH", POISON_LSV_BEFORE),
                                     ("after CH₃OH", POISON_LSV_AFTER))):
        e, j, _i_ring, _j_ring, rpm = load_rrde(path, RHE_01M_KOH)
        keep = (e >= POISON_RRDE_XRANGE[0]) & (e <= POISON_RRDE_XRANGE[1])
        e, j = e[keep], j[keep]
        color = PALETTE[0] if m == 0 else PALETTE[1]
        fig.add_trace(go.Scatter(
            x=e, y=j, mode="lines", name=tag, legendgroup=tag,
            line={"color": color, "width": LINE_W,
                  "dash": "solid" if m == 0 else "dash"},
        ), row=1, col=3)
        lim = float(np.median(j[np.argsort(e)[:8]]))
        if m == 0:
            j_before = lim
        else:
            j_after = lim
    # Magnitude-based change: poisoning should *reduce* the diffusion-limited
    # ORR current, so the sign here reflects |j| shrinking, not the raw
    # (already-negative) signed difference -- a signed ratio flips sign
    # whenever the less-negative value is called "after", which reads
    # backwards (see the HER retention fix and PROGRESS.md).
    change = 100.0 * (abs(j_after) - abs(j_before)) / abs(j_before)
    print(f"  10% ReS2-LaFeO3 RRDE: |j_lim| {abs(j_before):.3f} -> "
          f"{abs(j_after):.3f} mA cm⁻² ({change:+.0f} %) at {rpm:.0f} rpm")
    note(fig, 1, 3, f"j near {POISON_RRDE_XRANGE[0]:.2f} V: {change:+.0f} %"
                    f"  ·  {rpm:.0f} rpm", x=0.5, y=0.06, size_delta=-8)

    style_panels(
        fig,
        [("Time (s)", "Retention (%)"),
         ("Time (s)", "Current density (mA cm⁻²)"),
         ("Potential (V vs RHE)", "Disk current density (mA cm⁻²)")],
        rows=1, cols=3, height=PANEL_H + 120, width=int(PANEL_W * 2.6),
    )
    # Panel (c)'s title wraps to two lines (sample name + description);
    # give it the extra headroom a single-line title's margin doesn't.
    fig.update_layout(margin={"t": 100})
    fig.update_xaxes(range=POISON_RRDE_XRANGE, row=1, col=3)
    for (r, c), pos in (((1, 1), "top-right"), ((1, 2), "bottom-right"),
                        ((1, 3), "top-left")):
        lg = panel_legend(fig, r, c, 3, position=pos)
        for tr_ in fig.select_traces(row=r, col=c):
            tr_.legend = lg
    save(fig, filename)


# --- Figure 4: RRDE, 40 % Pt/C ----------------------------------------------
PTC_RRDE_DIR = os.path.join(DATA, "40% PtC 0-1 M KOH ORR", "Archive [2024-Jul 1046 #1]")
PRIMARY_RPM = 1600.0
# The diffusion-limited window the app averages n and %H2O2 over.
PLATEAU_V_RHE = (0.2, 0.6)


def figure_rrde_ptc(filename: str) -> None:
    print("40 % Pt/C RRDE (0.1 M KOH):")
    runs = []
    for name in sorted(n for n in os.listdir(PTC_RRDE_DIR) if n.startswith("RRDE")):
        runs.append(load_rrde(os.path.join(PTC_RRDE_DIR, name), RHE_01M_KOH))
    runs.sort(key=lambda r: r[4])

    e_ring_rhe = 0.40 + RHE_01M_KOH
    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.16, vertical_spacing=0.18,
        subplot_titles=("(a) Disk polarization curves",
                        f"(b) Ring current (E_ring = 0.40 V vs Hg/HgO, "
                        f"{e_ring_rhe:.2f} V vs RHE)",
                        f"(c) Electron-transfer number, {PRIMARY_RPM:.0f} rpm",
                        f"(d) Peroxide yield, {PRIMARY_RPM:.0f} rpm"),
    )
    for k, (e, j, i_ring, _j_ring, rpm) in enumerate(runs):
        color = PALETTE[k % len(PALETTE)]
        label = f"{rpm:.0f} rpm"
        fig.add_trace(go.Scatter(x=e, y=j, mode="lines", name=label,
                                 legendgroup=label,
                                 line={"color": color, "width": LINE_W}),
                      row=1, col=1)
        # Ring current as a density (uA/cm^2, disk area) rather than a raw
        # current, matching the disk panel's own units and the app's ORR
        # tab convention (ring normalised by the only area it's given).
        # showlegend=True: panel (b) needs its own rpm legend, since the
        # figure's single shared legend sits over panel (a) and does not
        # visually belong to (b) as far as a reader looking at (b) alone is
        # concerned.
        i_ring_density = _j_ring * 1000.0
        fig.add_trace(go.Scatter(x=e, y=i_ring_density, mode="lines", name=label,
                                 legendgroup=f"ring-{label}", showlegend=True,
                                 line={"color": color, "width": LINE_W}),
                      row=1, col=2)
        print(f"  {rpm:5.0f} rpm  j_lim = "
              f"{float(np.median(j[np.argsort(e)[:8]])):.3f} mA cm⁻²  "
              f"i_ring(max) = {i_ring_density.max():.2f} µA cm⁻²")

    primary = min(runs, key=lambda r: abs(r[4] - PRIMARY_RPM))
    e, j, i_ring, j_ring, rpm = primary
    n_arr = orr.electron_number(j, j_ring, N_COLLECTION)
    pct_arr = orr.peroxide_percent(j, j_ring, N_COLLECTION)
    fig.add_trace(go.Scatter(x=e, y=n_arr, mode="lines", showlegend=False,
                             line={"color": PALETTE[0], "width": LINE_W}),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=e, y=pct_arr, mode="lines", showlegend=False,
                             line={"color": PALETTE[2], "width": LINE_W}),
                  row=2, col=2)
    fig.update_yaxes(range=[0, 4.2], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=2)
    fig.add_hline(y=4, line={"color": "#7f7f7f", "width": 2, "dash": "dot"},
                  row=2, col=1)
    # n and %H2O2 are undefined (masked NaN) above onset -- restrict the
    # view to where the curve actually has values, from just above onset
    # down to the sweep's own lower end, instead of showing blank axis
    # space out to the full 1.3 V disk-sweep range.
    NPCT_XRANGE = [0.05, 0.95]
    fig.update_xaxes(range=NPCT_XRANGE, row=2, col=1)
    fig.update_xaxes(range=NPCT_XRANGE, row=2, col=2)

    n_mean, pct_mean, n_pts = orr.ring_disk_average(
        e, j, j_ring, N_COLLECTION, window=PLATEAU_V_RHE)
    note(fig, 2, 1, f"⟨n⟩ = {n_mean:.2f}<br>({PLATEAU_V_RHE[0]}–"
                    f"{PLATEAU_V_RHE[1]} V, {n_pts} pts)", x=0.5, y=0.18)
    note(fig, 2, 2, f"⟨%H₂O₂⟩ = {pct_mean:.1f} %", x=0.5, y=0.85)

    res = orr.onset_and_half_wave(e, j)
    for r, c in ((1, 1),):
        fig.add_vline(x=res.half_wave_potential,
                      line={"color": "#404040", "width": 2, "dash": "dashdot"},
                      row=r, col=c)
    note(fig, 1, 1,
         f"{rpm:.0f} rpm read-off:  E<sub>onset</sub> = "
         f"{res.onset_potential:.3f} V,  E<sub>1/2</sub> = "
         f"{res.half_wave_potential:.3f} V vs RHE",
         x=0.5, y=0.12, size_delta=-8)
    print(f"  {rpm:.0f} rpm: E_onset = {res.onset_potential:.3f} V, "
          f"E_1/2 = {res.half_wave_potential:.3f} V vs RHE, "
          f"n = {n_mean:.2f}, %H2O2 = {pct_mean:.1f} %")

    style_panels(
        fig,
        [("Potential (V vs RHE)", "Disk current density (mA cm⁻²)"),
         ("Potential (V vs RHE)", "Ring current density (µA cm⁻²)"),
         ("Potential (V vs RHE)", "Electron-transfer number, n"),
         ("Potential (V vs RHE)", "Peroxide yield, %H₂O₂")],
        rows=2, cols=2, height=(PANEL_H + 60) * 2, width=PANEL_W * 2,
    )
    # Separate legends per panel (not one shared legend anchored only to
    # panel (a)) so each plot is self-contained -- "bottom-right" for (a):
    # by the right edge of the sweep every rpm's disk current has already
    # converged to ~0, so that corner is clear of the curves, which is not
    # true of the top-right corner the curves are all *reaching toward*.
    # "top-right" for (b): the ring current has decayed to ~0 by high
    # potential, so that corner is clear there instead.
    lg_a = panel_legend(fig, 1, 1, 2, position="bottom-right")
    for tr_ in fig.select_traces(row=1, col=1):
        tr_.legend = lg_a
    lg_b = panel_legend(fig, 1, 2, 2, position="top-right")
    for tr_ in fig.select_traces(row=1, col=2):
        tr_.legend = lg_b
    save(fig, filename)


def main() -> None:
    figure_chrono("HER", "fig1_CA_HER_1M-KOH.png")
    figure_chrono("OER", "fig2_CA_OER_1M-KOH.png")
    figure_poisoning("fig3_methanol_poisoning.png")
    figure_rrde_ptc("fig4_RRDE_40pct-PtC.png")


if __name__ == "__main__":
    main()
