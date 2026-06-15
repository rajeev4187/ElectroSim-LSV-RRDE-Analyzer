# iR-compensation-calculation

A **Streamlit GUI** for automated ohmic-drop (*iR*) compensation of
linear-sweep voltammetry (LSV) data, using the uncompensated resistance
**Ru** estimated from electrochemical impedance spectroscopy (EIS).

**🔗 Live app:** <https://lsv-ir-compensation-calculation.streamlit.app/>

> Estimate `Ru` from an EIS Nyquist arc, reconcile current/resistance units
> (with electrode area when needed), then apply a user-selected
> iR-compensation factor (5 – 100 %; 85 % recommended) to your LSV curve and
> export the corrected data.

---

## Workflow

```text
 Excel / CSV ─►  EIS (Z', Z'')  ─►  circle fit  ─►  Ru  ┐
                                                        ├─►  E_corr = E − (f%)·I·Ru  ─►  CSV
                 LSV (Potential, Current)  ──────────────┘
```

1. **Load data** – an Excel workbook where **Sheet 1 = EIS** (`Z'`, `Z''`) and
   **Sheet 2 = LSV** (`Potential`, `Current`), or two separate CSV files.
   Each sheet may hold **several datasets side-by-side as repeated column
   pairs** (`Z'_1, Z''_1, Z'_2, Z''_2, …` and `E_1, I_1, E_2, I_2, …`). The app
   parses every pair and **links EIS sample *i* to LSV sample *i***; pick the
   active sample from the sidebar.
2. **Units & electrode area** – tell the app how the LSV current and the EIS
   resistance are reported so that `I·Ru` resolves to **volts**. Units are
   **auto-detected from the column headers** (e.g. `Current (mA/cm2)`,
   `Z' (Ohm)`) and shown in the sidebar:
   - *Absolute current* (mA, A, …) pairs with **Ru in Ω**.
   - *Current density* (mA/cm², …) pairs with **Ru in Ω·cm²**.
   - When one side is per-area and the other is not, an **electrode area
     (cm²)** reconciles them (`Ru[Ω·cm²] = Ru[Ω] · area`); when the units are
     already consistent the area input is bypassed.
3. **EIS / Ru Analysis tab** – the kinetic semicircle of the Nyquist plot is
   fitted with an algebraic circle fit. Its **high-frequency real-axis
   intercept is Ru**; the low-frequency intercept gives `Ru + Rct`. The
   low-frequency diffusion tail is auto-excluded and the fit range is fully
   adjustable. You can also enter Ru manually. When the current is a density,
   the impedance is reported **area-normalised in Ω·cm²** (with the original
   raw **Ω** value shown alongside). With multiple samples loaded, a **batch
   table** lists the fitted `Ru` for every sample.
4. **LSV iR Correction tab** – pick one *or several* compensation factors
   (5 – 100 %; 85 % recommended). The app computes

   ```text
   E_corrected = E_measured − (factor / 100) · I · Ru
   ```

   (with current scaled to its base SI unit and Ru reconciled via the area, so
   the drop is in volts). It shows a **with-vs-without comparison** (overlay or
   side-by-side), a results table with **Ru (Ω)**, **Ru effective (Ω·cm²)**,
   and the max iR drop, and lets you **download a CSV** that embeds `Ru`, the
   compensation %, and one column block per factor.
5. **Over-compensation guard** – each factor is checked for *fold-back* (the
   corrected potential reversing instead of advancing). Over-corrected factors
   are flagged with a ⚠ status, and the app reports the **highest safe factor**
   for that sample plus an interpretation of what a good compensation looks
   like.

### Why Ru comes from a circle fit (not an equivalent-circuit fit)

The reference dataset stores only `Z'` and `Z''` with **no frequency column**,
so a frequency-dependent equivalent-circuit (e.g. Randles) fit is not possible.
The geometric circle fit recovers Ru directly from the Nyquist arc and needs no
frequency data.

---

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `pandas`, `openpyxl`, `streamlit`, `plotly`.

## Running the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default `http://localhost:8501`).
Select **Use bundled sample** in the sidebar to try it immediately with
`sample-data/Book1-original data.xlsx` (current in mA/cm², EIS in Ω;
electrode area 0.04 cm²).

## Deploy publicly on Streamlit Community Cloud

This app is ready to deploy for free on
[Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repository to GitHub (it already contains `app.py` and
   `requirements.txt` at the root, which is all the platform needs).
2. Go to **share.streamlit.io → Create app**, pick this repo/branch, and set
   the **main file path** to `app.py`.
3. Click **Deploy**. The platform installs `requirements.txt` and serves the
   app at `https://<your-app-name>.streamlit.app`.

The sample workbook ships in `sample-data/`, so the deployed app works out of
the box; users can also upload their own Excel/CSV files.

### Keeping the source code private

Streamlit Community Cloud runs the app **directly from the GitHub repo it is
linked to**, so the code must live in that repo — you cannot `.gitignore`
`app.py` and still deploy it (the platform would not have the file). To publish
a working app while keeping the source hidden from the public, **make the
GitHub repository private**: Community Cloud can deploy from private repos
(authorize repo access when creating the app). The deployed app stays publicly
reachable at its `*.streamlit.app` URL, while the code is not visible to anyone
browsing GitHub. (You can additionally restrict app *viewers* in the app's
**Settings → Sharing** if you also want to limit who can use it.)

---

## Project layout

| Path | Purpose |
| ---- | ------- |
| `app.py` | Streamlit GUI (EIS/Ru analysis + LSV iR-correction tabs). |
| `ir_compensation/data_io.py` | Load EIS / LSV from Excel or CSV (fuzzy column matching). |
| `ir_compensation/eis.py` | Circle fit of the Nyquist arc → `Ru`, `Rct`. |
| `ir_compensation/correction.py` | Apply the iR correction (5–100 % factor; current/Ru unit & area reconciliation). |
| `sample-data/Book1-original data.xlsx` | Example data: `EIS` sheet (Z′, Z″ in Ω), `LSV` sheet (Potential V, Current mA/cm²). |

## Using the core API without the GUI

```python
from ir_compensation import data_io, eis, correction

eis_d = data_io.load_eis("sample-data/Book1-original data.xlsx", sheet=0)
lsv_d = data_io.load_lsv("sample-data/Book1-original data.xlsx", sheet=1)

ru = eis.fit_ru_circle(eis_d.z_real, eis_d.z_imag).ru     # ≈ 27.5 Ω

# Current density (mA/cm²) + Ru in Ω → reconcile with the electrode area.
result = correction.apply_ir_correction(
    lsv_d.potential, lsv_d.current, ru,
    factor_percent=85, current_unit="mA/cm²",
    ru_unit="Ω", area_cm2=0.04,
)
result.potential_corrected   # iR-corrected potentials (V)
result.ru_effective          # Ru reconciled to Ω·cm² (= ru · area)

# Absolute current (mA) + Ru in Ω needs no area:
correction.apply_ir_correction(
    lsv_d.potential, lsv_d.current, ru,
    factor_percent=100, current_unit="mA",  # 100 % = full correction
)
```

## Notes

- **Unit consistency.** Absolute-current units (`A`, `mA`, `µA`, `nA`) pair
  with **Ru in Ω**; current-density units (`A/cm²`, `mA/cm²`, …) pair with
  **Ru in Ω·cm²**. The current is scaled to its base SI unit (e.g. mA → A
  divides by 1000) and, when only one side is area-normalised, the **electrode
  area** converts Ω ↔ Ω·cm² — so `I·Ru` always resolves to volts. Units are
  guessed from the column headers and can be overridden in the sidebar.
- **Compensation factor** is selectable from **5 % to 100 %**. 100 % is the
  full ohmic-drop correction; **85 % is the recommended safe default**, since
  full positive feedback can over-correct/oscillate when `Ru` is uncertain.
  The fold-back guard flags any factor that over-corrects.

---

## Security

Because the app accepts arbitrary uploaded files on a public URL, several
safeguards are built in:

- **Upload limits.** `.streamlit/config.toml` caps upload size (15 MB) and
  message size, keeps XSRF protection on, disables usage telemetry, and hides
  Python tracebacks from end users (`showErrorDetails = false`).
- **Malicious-file defenses** (`ir_compensation/data_io.py`): Excel archives
  are checked for **decompression-bomb** expansion before parsing; tables are
  rejected above row/column caps (memory-exhaustion / DoS guard); the number of
  parsed column-pairs is capped; and header-derived labels are stripped of
  control characters / newlines before display or CSV output (prevents
  CSV/markdown injection).
- **No persisted data.** Uploads are processed in memory only; nothing is
  written to disk. The sample is **not preloaded**, and a **Clear / reset**
  button wipes session data.
- **Optional password gate.** Set an `app_password` secret (see
  `.streamlit/secrets.toml.example`) to require a password; comparison is
  constant-time. Leave it unset for a fully public app. `secrets.toml` is
  git-ignored, so credentials are never committed.

To restrict *who* can open the app entirely, you can also limit viewers under
the app's **Settings → Sharing** on Streamlit Community Cloud.

---

## Citation

If you use this app (including a public Streamlit deployment) in your work,
please cite it. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); a plain-text form:

> Kumar, R. (2026). *Automated iR Compensation: EIS fitting and LSV
> correction* (v1.0.0) [Computer software]. North Carolina Central University.
> <https://github.com/rajeev4187/LSV-iR-compensation-calculation>

```bibtex
@software{kumar_ir_compensation_2026,
  author  = {Kumar, Rajeev},
  title   = {Automated iR Compensation: EIS fitting and LSV correction},
  version = {1.0.0},
  year    = {2026},
  url      = {https://github.com/rajeev4187/LSV-iR-compensation-calculation}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
