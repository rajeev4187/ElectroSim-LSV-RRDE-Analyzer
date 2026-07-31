# LSV Analysis: iR Compensation, Tafel Slope

**🔗 Live app:** <https://lsv-analysis-ir-compensation-tafel-slope.streamlit.app/>

A Streamlit app for two independent electrochemistry workflows, run from
uploaded Excel/CSV files — no coding required:

1. **iR compensation** — correct linear-sweep voltammetry (LSV) data for
   ohmic drop, using the uncompensated resistance **Ru** fitted from an EIS
   Nyquist arc.
2. **Tafel slope analysis** — fit the linear (activation-controlled) region
   of a polarization curve to extract the Tafel slope, with RHE reference
   conversion and mechanistic interpretation (HER, HOR, OER, ORR, CO₂RR,
   N₂RR, NO₃RR, and more).

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`). In the
sidebar, pick **Use bundled sample** to try the EIS/LSV tabs immediately with
the example workbook in `sample-data/`.

PNG downloads need a headless Chrome (via `kaleido`); if it's missing, run
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

### 3 · Tafel Slope Analysis

Independent of the tabs above — its own uploader, units, and reference
electrode.

1. **Upload** one or more Potential/Current files (Excel or CSV), one per
   sample; combine them into a single overlay plot.
2. **Convert to RHE** if your data isn't already on that scale: pick the
   reference electrode used (SCE, Ag/AgCl variants, Hg/HgO, …) and the
   electrolyte pH.
3. The **linear Tafel region is auto-detected** from the reaction onset;
   fine-tune it per sample by dragging the slider or box-selecting the region
   directly on the plot.
4. **Tag each sample's reaction** (HER, HOR, OER, ORR, CO₂RR, N₂RR, NO₃RR,
   and more) — samples of different reactions can share one overlay, split
   into per-reaction legend groups.
5. Read off the **Tafel slope, R², and nearest mechanistic benchmark**, then
   download the plots (PNG/HTML) and plotted data (CSV) — publication-styled
   with Arial fonts and a selectable font size.

---

## Good to know

- **LSV data.** Linear-sweep voltammetry records current as the electrode
  potential is swept linearly — every tab starts from a `Potential, Current`
  (or current-density) pair read from your file.
- **Potential → RHE conversion** (Tafel tab). If your data is referenced to
  another electrode (SCE, Ag/AgCl, …), it's converted with
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
  [`ir_compensation/`](ir_compensation/).

---

## Citation

If you use this app (including the public Streamlit deployment) in your
work, please cite it. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); a plain-text form:

> Kumar, R. (2026). *LSV Analysis: iR Compensation and Tafel Slope*
> (v1.1.0) [Computer software]. North Carolina Central University.
> <https://github.com/rajeev4187/LSV-Analysis-iR-compensation-Tafel-slope>

```bibtex
@software{kumar_lsv_analysis_2026,
  author  = {Kumar, Rajeev},
  title   = {LSV Analysis: iR Compensation and Tafel Slope},
  version = {1.1.0},
  year    = {2026},
  url      = {https://github.com/rajeev4187/LSV-Analysis-iR-compensation-Tafel-slope}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
