# ElectroSim-LSV-RRDE-Analyzer

**🔗 Live app:** <https://lsv-analysis-ir-compensation-tafel-slope.streamlit.app/>

A Streamlit app for electrochemistry workflows, run from uploaded Excel/CSV
files — no coding required:

1. **iR compensation** — correct linear-sweep voltammetry (LSV) data for
   ohmic drop, using the uncompensated resistance **Ru** fitted from an EIS
   Nyquist arc.
2. **LSV analysis** — onset potential, overpotential at benchmark current
   densities (e.g. j = 10 mA/cm²), and the Tafel slope of a polarization
   curve, with RHE reference conversion and mechanistic interpretation (HER,
   HOR, OER, ORR, CO₂RR, N₂RR, NO₃RR, and more).
3. **Koutecky–Levich (K-L) analysis** — the classic multi-rotation-rate RDE
   fit for the kinetic current density and electron-transfer number.
4. **ORR / RRDE analysis** — onset/E½/Tafel/electron-number/peroxide-yield
   from ring-disk data at one rotation rate, plus a multi-rotation-rate
   ring/disk comparison.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`). In the
sidebar, pick **Use bundled sample** to try the EIS/LSV tabs immediately with
the example workbook in `sample-data/`.

TIFF downloads need a headless Chrome (via `kaleido`); if it's missing, run
`plotly_get_chrome`, or use the HTML/CSV download options instead.

---

## How to use it

### 1 · EIS / Ru Analysis and 2 · LSV iR Correction

1. **Upload** an Excel workbook (Sheet 1 = EIS: `Z'`, `Z''`; Sheet 2 = LSV:
   `Potential`, `Current`) or two CSV files. Several samples can sit
   side-by-side in one sheet as repeated column pairs; pick the active one
   from the sidebar.
2. **Confirm units.** Current unit and Ru unit are auto-detected from the
   column headers. If one is per-area (e.g. mA/cm²) and the other isn't,
   enter the **electrode area (cm²)** to reconcile them.
3. **EIS / Ru Analysis tab** — Ru is read off the high-frequency intercept of
   a circle fit to the Nyquist semicircle (adjust the fit range, or enter Ru
   manually).
4. **LSV iR Correction tab** — pick one or more compensation factors
   (5–100 %; **85 % recommended**):

   ```text
   E_corrected = E_measured − (factor / 100) · I · Ru
   ```

   View the corrected curve against the raw one, then download the result as
   CSV. An over-compensation ("fold-back") guard flags any factor that's
   unsafe and suggests the highest safe one.

### 3 · LSV Analysis

Independent of the tabs above — its own uploader, units, and reference
electrode.

1. **Upload** one or more Potential/Current files (Excel or CSV), one per
   sample; combine them into a single overlay plot.
2. **Convert to RHE** if your data isn't already on that scale: pick the
   reference electrode used (SCE, Ag/AgCl variants, Hg/HgO, …) and the
   electrolyte pH.
3. Read off the **onset potential** (where |current| first departs from the
   flat pre-onset baseline) and the **overpotential at one or more benchmark
   current densities** (e.g. η at j = 10 mA/cm², the standard OER/HER
   activity benchmark; j = 2 mA/cm² is also common) — reported as
   η = |E − E_eq| for reactions with a known equilibrium potential
   (HER/HOR/OER/ORR/NO₃RR/N₂RR/CO₂RR), or as the raw potential otherwise.
4. The **linear Tafel region is auto-detected** from the reaction onset;
   fine-tune it per sample by dragging the slider or box-selecting the region
   directly on the plot.
5. **Tag each sample's reaction** (HER, HOR, OER, ORR, CO₂RR, N₂RR, NO₃RR,
   and more) — samples of different reactions can share one overlay, split
   into per-reaction legend groups.
6. **Group repeat scans of one sample** by giving them the same **Replicate
   group** name (auto-filled when you upload the same file more than once) —
   every fitted value (Tafel slope, onset, η@j, …) is then also reported as
   a **mean ± SD across the group** in its own results section.
7. Read off the **Tafel slope, R², and nearest mechanistic benchmark**, then
   download the plots (TIFF/HTML) and plotted data (CSV) — publication-styled
   with Arial fonts and a selectable font size.

### 4 · K-L (Koutecky–Levich) Analysis

Independent of the tabs above. Needs an RDE polarization curve at **three or
more rotation rates** for the same sample (working/disk electrode current
only — no ring needed).

1. **Upload** one file per rotation rate (Excel or CSV, just
   `Potential, Current`), or a ZIP of a whole data folder for batch loading
   (see [Loading RRDE/RDE data](#loading-rrderde-data) below).
2. **Convert to RHE** and set the **electrode area** the same way as the
   other tabs.
3. Pick the **electrolyte's O₂ transport parameters** (diffusion coefficient,
   kinematic viscosity, bulk O₂ concentration) — presets for 0.1 M KOH,
   0.1 M HClO₄, and 0.5 M H₂SO₄ are built in, or enter your own.
4. At each of several **analysis potentials** (evenly spaced across the
   range every rotation rate shares), the app fits

   ```text
   1/j = 1/j_k + 1/(B · ω^0.5),   ω = 2π·rpm/60,   B = 0.62 n F D^(2/3) ν^(−1/6) C
   ```

   (Koutecký & Levich, *Zh. Fiz. Khim.* **1958**, *32*, 1565; Bard &
   Faulkner, *Electrochemical Methods*, 2nd ed., Wiley, 2001, Ch. 9) — the
   **intercept** gives the kinetic current density j_k, and the **slope**
   gives the electron-transfer number **n** via the Levich constant B.
5. Download the RDE-curve and K-L plots (TIFF/HTML) and the results table
   (TIFF/CSV).

### 5 · ORR / RRDE Analysis

Independent of the tabs above. Works with disk-only (RDE) or disk+ring
(RRDE) data, one or more samples, one or more rotation rates each.

1. **Upload** data for one or more samples (see
   [Loading RRDE/RDE data](#loading-rrderde-data) below) and pick which
   samples to compare.
2. **Convert to RHE**, set the **electrode area**, and — if ring current is
   available — the **ring collection efficiency N**.
3. Pick a **primary rotation rate** (conventionally 1600 rpm); each sample
   uses its own closest available rate if it doesn't have that exact one.
4. At that rotation rate, read off each sample's **onset potential**, **half-
   wave potential E½** (the steepest point of the disk curve), and — after
   removing the mass-transport contribution — the **Tafel slope** (each
   sample gets its own adjustable fit-range slider).
5. If ring current is available: **electron-transfer number n** and
   **peroxide yield %H₂O₂**, computed directly from disk + ring current (no
   multi-rotation-rate fit needed):

   ```text
   n = 4|I_d| / (|I_d| + |I_r|/N)
   %H2O2 = 200(|I_r|/N) / (|I_d| + |I_r|/N)
   ```

   (Bard & Faulkner, *Electrochemical Methods*, 2nd ed., Wiley, 2001.)
6. If a sample has **several rotation rates**, see them compared in one
   merged plot — ring current above zero, disk current below, sharing one
   potential axis (the usual published-RRDE-figure style).
7. Download every plot (TIFF/HTML) and the results/plotted data (TIFF/CSV).

#### Loading RRDE/RDE data

The K-L and ORR/RRDE tabs share the same loader, in two styles (usable
together):

- **Per-sample uploaders** — for each sample, upload either raw
  per-electrode files (one file per rotation rate *and* electrode, just
  `Potential, Current` — the layout most RDE/RRDE instrument software
  exports, e.g. `Disk Current vs Disk Potential (1600 RPM).csv`), or a
  compiled workbook with `Potential, Disk current, Potential, Ring current`
  already paired for one rotation rate. Rotation rate and disk/ring role are
  guessed from the filename and can be corrected by hand.
- **Batch ZIP upload** — zip a whole data folder (one subfolder per sample)
  and upload it in one go; role and rotation rate are read from filenames
  automatically, with no per-file correction UI. Use the per-sample
  uploaders instead for anything that needs fixing by hand.

---

## Good to know

- **LSV data.** Linear-sweep voltammetry records current as the electrode
  potential is swept linearly — every tab starts from a `Potential, Current`
  (or current-density) pair read from your file.
- **Potential → RHE conversion.** If your data is referenced to another
  electrode (SCE, Ag/AgCl, …), it's converted with
  `E(RHE) = E(measured) + E°(ref vs NHE) + 0.0592 V·pH⁻¹ × pH`. Skip this if
  your data is already vs RHE.
- **Current & current-density conversion.** Switching the current-unit
  dropdown (A ↔ mA ↔ µA ↔ nA, or their `/cm²` counterparts) instantly
  rescales the values within that family. Converting *between* an absolute
  current and a density needs the **electrode area**:
  `current density [A/cm²] = current [A] / area [cm²]`. The same area also
  reconciles `Ru` between Ω and Ω·cm² so `I·Ru` always resolves to volts.
- **Privacy.** Uploaded files are processed in memory only — nothing is
  written to disk — and a **Clear / reset** button wipes the session. An
  optional password gate can be set via a `app_password` secret.
- Full technical notes (data format, security hardening, using the Python
  API without the GUI) are in the module docstrings under
  [`scripts/modules/`](scripts/modules/).

---

## Citation

If you use this app (including the public Streamlit deployment) in your
work, please cite it. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); a plain-text form:

> Kumar, R. (2026). *ElectroSim-LSV-RRDE-Analyzer*
> (v1.1.0) [Computer software]. North Carolina Central University.
> <https://github.com/rajeev4187/LSV-Analysis-iR-compensation-Tafel-slope>

```bibtex
@software{kumar_electrosim_lsv_rrde_2026,
  author  = {Kumar, Rajeev},
  title   = {ElectroSim-LSV-RRDE-Analyzer},
  version = {1.1.0},
  year    = {2026},
  url      = {https://github.com/rajeev4187/LSV-Analysis-iR-compensation-Tafel-slope}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
