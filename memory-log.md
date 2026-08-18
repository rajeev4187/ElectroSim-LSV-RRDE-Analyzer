# Review Log — ElectroSim-LSV-RRDE Analyzer

Four independent reviewer agents audited the app (science, numerics, app/export
robustness, and journal-style figures/tables) against the reference EIS app
(`Working-Apps/EIS/ElectroSim-EIS.py`). No CRITICAL (wrong-science or
crash/security) defects were found in the computational core. Findings below
are the remaining IMPORTANT/MINOR items, prioritized for the fix pass.

Legend: 🔴 IMPORTANT, 🟡 MINOR, ✅ confirmed-correct (no action).

---

## 1. Scientific accuracy (Reviewer 1)

### 🔴 RHE default inconsistency (17 mV mode-toggle shift)
- `app.py` ~L2763 (K-L tab), ~L3559 (ORR tab) default to **"Hg/HgO, 1 M NaOH"
  (0.140 V) + "0.1 M KOH" (pH 13)** → offset 0.909 V.
- `app.py` ~L1871 (direct-calibration placeholder) cites **0.926 V** for
  Hg/HgO in 1 M KOH.
- Toggling modes shifts all potentials by 17 mV with no data change.
- **Fix:** default K-L/ORR to `"Hg/HgO, 1 M KOH" (0.098 V)` + pH 13 (→ 0.867 V),
  or make the calibration placeholder match the formula mode.

### 🔴 `infer_reaction` classification branch ordering
- `tafel.py` ~L485–495: anodic branch checks `e_mid >= 1.23 → OER`, then
  `e_hi >= 1.35 → OER`, **then** `e_lo <= 0.25 → HOR`. An HOR sweep recorded
  from ~0 V up past 1.4 V is misclassified **OER**, silently setting E_eq=1.23 V.
- **Fix:** test the HOR criterion (`e_lo <= 0.25`) before the `e_hi >= 1.35` OER
  branch (or require `e_mid > 1.23` for OER).
- Also: cathodic `e_mid > 0.95` returns "just below 1.23 V" wording that is
  wrong when `e_mid` is actually above 1.23 V (branch the wording).
- Also: poor-ORR cathodic sweep with `e_mid` just below 0.15 V gets **high HER**
  confidence with no ORR alternative flagged — extend the medium-confidence
  message to the 0.0–0.15 V band.

### 🟡 Minor
- `correction.py` ~L154 docstring says factor "clamped to 5..85" but code clamps
  to [5, 100].
- `tafel.exchange_current` is only physically i0 when the fit runs on
  overpotential (never displayed in GUI — latent footgun only).
- `app.py` ~L3576–3584: ring current divided by **disk** area and labeled
  "Ring current (mA/cm²)" — misleading (ring has its own area). n/%H2O2 are
  unaffected (ratio). Relabel.
- E1/2 search window default 0.4–0.8 V can silently narrow for poor catalysts;
  add a user-visible warning when E1/2 lands on the window edge.
- `eis.fit_ru_circle` Ru is an extrapolation (arc coverage dependent); add a
  visible caveat. `_abs_imag` includes high-frequency inductive artifacts.
- `app.py` ~L1899 caption hardcodes "0.0592" while code uses 0.05916.

### ✅ Confirmed correct
iR sign/units/Ru-reconciliation; Tafel slope magnitude + regression + E_eq;
RHE conversion + reference values + Nernst slope; K-L n/B/Levich/presets (5.7
mA/cm² @ 1600 rpm); ring-disk n & %H2O2 + masking; mass-transport correction
guards; EIS circle intercept `a − √(r²−b²)`.

---

## 2. Numerical / algorithmic (Reviewer 2)

### 🔴 `monotonic_segments` off-by-one (latent)
- `sweep.py` ~L77–79: `bounds = [0, *(int(b)+1 for b in breaks), n]` double-counts
  the boundary; every segment after the first starts one index too late.
- Currently masked because `main_sweep_indices` (~L98) re-adds the vertex via
  `np.arange(max(0, start-1), stop)`. Fragile silent coupling.
- **Fix:** emit genuinely maximal/shared-vertex segments and drop the `-1`
  (or document the coupling).

### 🟡 Minor
- `fit_tafel` auto-range path never uses onset detection (no `current`/`e_eq`
  forwarded) — always the coarse `_best_r2_window` fallback.
- `fit_slice` reports pre-filter window, not the fitted (non-finite-dropped)
  points.
- `_stats_from_sums` one-pass variance is cancellation-prone at large offsets
  (fine at current scales); could subtract a reference before cumsum.
- `orr.KoutieckyLevichFit.kinetic_current_density` can return negative when
  `intercept < 0`; `is_reliable` checks `!= 0` not `> 0`.
- `fit_koutecky_levich` does not reject duplicate rotation rates (degenerate
  polyfit).
- `_ring_disk_terms`: all-zero disk current → `peroxide_percent` returns a
  confident 100 % (mask `peak <= 0` as all-invalid).
- `levich_slope_to_n` / `levich_current_density`: no positivity validation for
  D/ν/C (fractional powers of negative values raise).
- `sweep.ascending_xy` does not drop NaN x (public `interp_at`).
- `_grow_from_onset` overpotential cap assumes onset sits on the faradaic side
  of e_eq (silently ignored otherwise).
- `sweep._blocks` degenerates to 1 for tiny sweeps (single-endpoint robustness).

### ✅ Confirmed correct
Prefix-sum OLS (slope/intercept/R²) == polyfit; standard errors; O(1) per window
(no O(n²)); `auto_tafel_range` reversed-index mapping `(n-stop, n-onset_idx)`;
`ascending_xy` duplicate collapsing; `clean_sweep` lockstep masking; Kasa circle
fit + `disc<=0` fallback; onset/half-wave direction-independence + approach-leg
handling.

---

## 3. App / export robustness (Reviewer 3)

### 🔴 Widget-key collisions from user data
- `app.py` ~L3158–3162: sheet selectbox keyed by raw filename → DuplicateWidgetID
  when two uploads share a name (common in RRDE workflows). Key on upload index.
- `app.py` L2708, L3505, ~L4178: `_orr_data_loader` does not de-duplicate sample
  labels (Tafel loader does). `dict(samples)[label]` silently collapses
  same-named samples; `orr_rrde_rpms_{label}` key collides.
- `app.py` ~L2649: bar-chart key from `f"{target_j:g}"` — distinct floats like
  1000000 and 1000000.1 both format `"1e+06"` → DuplicateWidgetID.

### 🔴 Export cache ignores format/dpi/size
- `app.py` L287–320: cached export invalidated only by figure JSON signature;
  changing format/dpi/width/height still serves stale bytes with old ext/mime.
  Store `fmt/dpi/out_w/out_h` in the cache and compare.

### 🟡 Minor
- HTML export built on every rerun (expander bodies always execute) — contradicts
  docstring.
- Table canvas width can exceed the 4000 px `number_input` max → clamp defaults.
- `_zero_anchored_range` returns `[0,0]` for all-zero data → degenerate axis.
- Index-keyed per-sample widgets (`tafel_reaction_{i}`, etc.) mis-associate when
  selection changes — key on stable label slug.
- User strings (file/folder/ZIP names, "Sample name"/"Legend name") not sanitized
  before entering HTML export (XSS-in-a-file).
- Vector export size "in points" comment wrong (kaleido uses CSS px).
- Client-side legend/annotation drag edits are not captured in server-side export.
- ZIP loaders bound decompressed size but not member count.
- `np.interp` on duplicate ring potential in `_orr_merge_entries`/
  `_orr_table_to_entry` — route through `sweep.ascending_xy`.

### ✅ Confirmed correct
Password gate (hmac.compare_digest, constant-time); zip-bomb guard + row/col caps;
no eval/exec/subprocess/pickle; cache_data keyed on bytes/str; dpi→scale math;
MIME types; export signature laziness; cross-tab isolation.

---

## 4. Journal-style figures & tables (Reviewer 4)

### 🔴 HIGH
- **10 px margins clip tick labels/axis titles** in every export
  (`apply_plot_style` L1367–1372; hard-coded `margin` dicts in Tafel/LSV tabs).
  Adopt reference margins `l=110, r=90, b=90/100`.
- **Square 520×520 px default export** = 1.73 in @300 dpi (below journal minimum).
  Add single/double-column width presets (8.6 cm / 17.8 cm).
- **Ticks grey & thin**: `_axis_style`/`_BOX_AXIS_STYLE` set `ticks='outside'`
  but no `tickwidth=2, ticklen=7, tickcolor='black'`; axis `linewidth` 1.5 → 2.

### 🟡 MEDIUM
- No "nice" tick rounding helper (port `_nice_axis`/`_square_nyquist` from EIS app).
- PNG/JPEG lack dpi metadata (route through Pillow like TIFF).
- Tables left-align numbers and draw interior vertical rules (should right-align
  numeric columns; drop cell rules).
- Axis title reuses tick size (should be larger, e.g. size+8).
- `show_title` not honored by Tafel/LSV hard-coded titles.
- Hard-coded 22 pt Tafel slope annotation.

### ✅ Already equivalent to EIS reference
White bg / no grid by default; TIFF LZW + 300 dpi tagging; legend scaling;
HTML + CSV fallback; figure-change signature check.
