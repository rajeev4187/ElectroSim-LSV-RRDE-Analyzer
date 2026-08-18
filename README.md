# ElectroSim-LSV-RRDE-Analyzer

**Publication-ready electrochemical analysis in the browser — upload a file,
get the numbers and the figure.**

Live app: <https://lsv-analysis-ir-compensation-tafel-slope.streamlit.app/>

A Streamlit application that takes raw potentiostat exports (Excel/CSV/ZIP)
and returns the quantities electrocatalysis papers actually report — Ru, iR-
corrected curves, onset potentials, overpotentials at benchmark current
densities, Tafel slopes, Koutecký–Levich electron numbers, half-wave
potentials, and RRDE peroxide yields — together with journal-styled plots and
export-ready tables. No scripting, no installation, no data leaving memory.

---

## Status

| Aspect | Current state |
| --- | --- |
| Release | **v1.1.0** — stable, actively maintained |
| Deployment | Public Streamlit Cloud app (link above); runs locally the same way |
| Scope | 5 independent analysis modules, ~5.5 k lines of Python |
| Data handling | In-memory only; optional password gate; one-click session reset |
| License | MIT |

Recent additions: batch ZIP loading of whole RRDE data folders, replicate-group
statistics (mean ± SD) for LSV fits, reaction-aware automatic Tafel-region
detection, stacked ring/disk multi-rotation-rate plots, and independent but
linked EIS and LSV iR-correction tabs.

---

## Capabilities

| Module | Input | Output |
| --- | --- | --- |
| **EIS / Ru Analysis** | Nyquist data (`Z'`, `Z''`) | Ru from a high-frequency circle fit, plus Rct and Ru+Rct; adjustable fit range or manual Ru |
| **LSV iR Correction** | Polarization curve + Ru | `E_corr = E - (f/100)·I·Ru` at any factor 5–100 % (85 % recommended), with an over-compensation "fold-back" guard |
| **LSV Analysis** | One or many `Potential, Current` files | Onset potential, η at benchmark j (e.g. 10 mA/cm²), auto-detected Tafel region, Tafel slope + R² + mechanistic benchmark, multi-sample overlays |
| **K-L Analysis** | RDE curves at 3+ rotation rates | Koutecký–Levich fit at several potentials → kinetic current density j_k and electron-transfer number n |
| **ORR / RRDE Analysis** | Disk-only or disk + ring, 1+ samples, 1+ rotation rates | Onset, half-wave potential E½, mass-transport-corrected Tafel slope, electron number n and %H₂O₂ from ring/disk current, merged multi-rpm ring/disk figures |

Cross-cutting features:

- **12 reactions supported** — HER, HOR, OER, ORR, CO₂RR, CORR, N₂RR, NO₃RR,
  MOR, EOR, UOR, or unspecified. Samples of different reactions can share one
  overlay, split into per-reaction legend groups; overpotentials are reported
  against the correct equilibrium potential where one is defined.
- **9 reference electrodes** — SHE/NHE, SCE, Ag/AgCl, Hg/HgO, Hg/Hg₂SO₄
  variants — converted to RHE with
  `E(RHE) = E_meas + E°(ref vs NHE) + 0.0592 × pH`.
- **Real unit handling** — A ↔ mA ↔ µA ↔ nA and their `/cm²` counterparts,
  auto-detected from column headers; the electrode area reconciles absolute
  current with current density and Ω with Ω·cm², so `I·Ru` always resolves to
  volts.
- **Replicate groups** — repeat scans sharing a group name get every fitted
  value reported as **mean ± SD** in its own results section.
- **Publication-styled export** — Arial fonts, selectable size, closed box
  axes, draggable legends and slope labels.
- **Figure styling, under your control.** A **Plot appearance** panel on every
  tab sets the font family and size, the colour palette (including an
  Okabe-Ito colourblind-safe set and a grayscale set for print-only figures),
  per-series colours, line width, marker size and symbol, fit-line dash,
  gridlines, box frame, tick density and legend placement. Exports render the
  same figure object you see, so the two cannot drift apart.
- **Dynamic export.** Every plot exports as TIFF, PNG, SVG, PDF or JPEG at
  150-1200 dpi with adjustable pixel dimensions; every table exports as a
  journal-styled figure (ruled header, zebra rows) or as CSV, plus interactive
  HTML for any plot.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. Choose **Use bundled sample** in the sidebar to
try the EIS/LSV tabs instantly, or load `sample-data/Example ORR 0-1 M KOH/`
for a full five-rotation-rate RRDE dataset.

Image export (TIFF/PNG/SVG/PDF/JPEG) needs headless Chrome via `kaleido`; if
it is missing, run `plotly_get_chrome` or use the HTML/CSV downloads, which
need no external renderer.

### Development

```bash
pip install -r requirements-dev.txt
pytest          # analysis modules + app smoke tests
flake8 app.py scripts tests
```

CI runs the same checks on Python 3.12, 3.13 and 3.14 (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Dependency floors in
`requirements.txt` are split by Python version, because a single low floor
resolves on 3.14 to a release with no matching wheel and forces a source
build.

---

## Loading data

- **EIS / LSV tabs** — one Excel workbook (Sheet 1 = EIS `Z'`, `Z''`;
  Sheet 2 = LSV `Potential`, `Current`) or two CSVs. Several samples can sit
  side-by-side as repeated column pairs; pick the active one in the sidebar.
- **LSV Analysis** — one `Potential, Current` file per sample (Excel or CSV),
  combined into a single overlay.
- **K-L and ORR/RRDE** — share one loader, in two styles:
  - **Per-sample uploaders** for raw per-electrode files exactly as
    instrument software exports them
    (`Disk Current vs Disk Potential (1600 RPM).csv`), or a compiled workbook
    with `Potential, Disk current, Potential, Ring current` already paired.
    Rotation rate and disk/ring role are guessed from the filename and can be
    corrected by hand.
  - **Batch ZIP upload** — zip a data folder (one subfolder per sample) and
    load everything at once, roles and rotation rates read from filenames.

Electrolyte O₂ transport parameters for K-L ship as presets for 0.1 M KOH,
0.1 M HClO₄, and 0.5 M H₂SO₄, or can be entered manually.

---

## Methods

```text
iR correction    E_corr = E - (f/100)·I·Ru
Koutecký–Levich  1/j = 1/j_k + 1/(B·w^0.5),  w = 2·pi·rpm/60,
                 B = 0.62 n F D^(2/3) v^(-1/6) C
RRDE             n = 4|Id| / (|Id| + |Ir|/N)
                 %H2O2 = 200(|Ir|/N) / (|Id| + |Ir|/N)
```

- Koutecký, J.; Levich, V. G. *Zh. Fiz. Khim.* **1958**, *32*, 1565.
- Bard, A. J.; Faulkner, L. R. *Electrochemical Methods*, 2nd ed.; Wiley,
  2001; Ch. 9 (Levich constant B, RRDE collection efficiency, `n` and %H₂O₂).

### What the analysis does for you

- **Sweep cleaning.** Instrument exports commonly open with an approach or
  vertex leg running opposite to the real scan. It is detected and removed
  before anything is measured — left in, it corrupts every derivative and
  every interpolation on the curve.
- **Direction independence.** Onset, E½, η@j and the auto Tafel window give
  the same answer whether the file was recorded from the rest potential or
  from the plateau.
- **Automatic reaction assignment.** Each LSV sample's reaction (HER / HOR /
  OER / ORR) is inferred from the sign of its faradaic current and where it
  flows on the RHE scale, with a stated confidence and reason; it drives the
  equilibrium potential used for η, the mechanistic benchmark, and the
  overpotential cap on the auto Tafel window. Override per sample at any time.
- **Uncertainties and honest flags.** Tafel slopes are reported with a 95 %
  confidence interval and the number of current decades fitted, and are
  flagged when the window is too short to be a Tafel region regardless of how
  good R² looks. Koutecký-Levich rows carry a reliability verdict, since a fit
  through near-zero currents yields an arithmetically valid but meaningless
  `n`. Peroxide yield and `n` are blanked before the reaction onset instead of
  pinning at 100 %.
- **E½ method is yours to choose** — the literature-standard j_lim/2
  interpolation (default), the steepest point, or the d²I/dE² inflection.

Full technical notes — data formats, security hardening, and using the Python
API without the GUI — are in the module docstrings under
[`scripts/modules/`](scripts/modules/).

---

## Citation

If you use this app (including the public Streamlit deployment) in your work,
please cite it. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

> Kumar, R. (2026). *ElectroSim-LSV-RRDE-Analyzer* (v1.1.0)
> [Computer software]. North Carolina Central University.
> DOI: [10.5281/zenodo.21997767](https://doi.org/10.5281/zenodo.21997767).
> <https://github.com/rajeev4187/LSV-Analysis-iR-compensation-Tafel-slope>

```bibtex
@software{kumar_electrosim_lsv_rrde_2026,
  author  = {Kumar, Rajeev},
  title   = {ElectroSim-LSV-RRDE-Analyzer},
  version = {1.1.0},
  year    = {2026},
  doi     = {10.5281/zenodo.21997767},
  url     = {https://github.com/rajeev4187/LSV-Analysis-iR-compensation-Tafel-slope}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
