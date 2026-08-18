# Review Log — ElectroSim-LSV-RRDE Analyzer

Four independent reviewer agents audited the app (science, numerics, app/export
robustness, and journal-style figures/tables) against the reference EIS app
(`Working-Apps/EIS/ElectroSim-EIS.py`). No CRITICAL (wrong-science or
crash/security) defects were found in the computational core. Findings below
are the remaining IMPORTANT/MINOR items, prioritized for the fix pass.

Legend: 🔴 IMPORTANT, 🟡 MINOR, ✅ confirmed-correct (no action),
☑️ **FIXED** (2026-08-18 pass — see "Fix pass" at the bottom).

---

## 1. Scientific accuracy (Reviewer 1)

### ☑️ ~~RHE default inconsistency (17 mV mode-toggle shift)~~ — FIXED
- `app.py` ~L2763 (K-L tab), ~L3559 (ORR tab) default to **"Hg/HgO, 1 M NaOH"
  (0.140 V) + "0.1 M KOH" (pH 13)** → offset 0.909 V.
- `app.py` ~L1871 (direct-calibration placeholder) cites **0.926 V** for
  Hg/HgO in 1 M KOH.
- Toggling modes shifts all potentials by 17 mV with no data change.
- **Fix:** default K-L/ORR to `"Hg/HgO, 1 M KOH" (0.098 V)` + pH 13 (→ 0.867 V),
  or make the calibration placeholder match the formula mode.

### ☑️ ~~`infer_reaction` classification branch ordering~~ — FIXED
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

### ☑️ ~~`monotonic_segments` off-by-one (latent)~~ — FIXED
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

### ☑️ ~~Widget-key collisions from user data~~ — FIXED
- `app.py` ~L3158–3162: sheet selectbox keyed by raw filename → DuplicateWidgetID
  when two uploads share a name (common in RRDE workflows). Key on upload index.
- `app.py` L2708, L3505, ~L4178: `_orr_data_loader` does not de-duplicate sample
  labels (Tafel loader does). `dict(samples)[label]` silently collapses
  same-named samples; `orr_rrde_rpms_{label}` key collides.
- `app.py` ~L2649: bar-chart key from `f"{target_j:g}"` — distinct floats like
  1000000 and 1000000.1 both format `"1e+06"` → DuplicateWidgetID.

### ☑️ ~~Export cache ignores format/dpi/size~~ — FIXED
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

### ☑️ HIGH — all three FIXED
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

---

## Fix pass — 2026-08-18

Every 🔴 IMPORTANT finding above is now fixed, plus the 🟡 items listed
here. 133 tests pass, flake8 clean.

### Science
- RHE: K-L and ORR tabs now default to `Hg/HgO, 1 M KOH` (0.098 V) + pH 13, so
  the formula and calibration modes agree. The 0.926 V placeholder is kept but
  its help text now says it is the pH-14 value and gives the pH-13 one (0.867 V).
  The caption reads the Nernst slope from `tafel.NERNST_SLOPE_V_PER_PH` instead
  of a hardcoded 0.0592.
- `infer_reaction`: HOR is tested **before** the "reaches 1.35 V" OER branch, so
  an HOR sweep run up past 1.4 V is no longer called OER. Cathodic wording
  branches on whether `e_mid` is actually above 1.23 V. The 0–0.15 V band now
  returns HER at *medium* confidence and names ORR as the alternative.
- `correction.py` docstring now points at `MIN/MAX_FACTOR_PERCENT` rather than
  restating a stale "5..85".
- Ring current normalised by the **disk** area is labelled `mA/cm²(disk)` and
  carries a caption saying it is not the ring's own current density (and that
  n / %H₂O₂ are unaffected because the area cancels).
- E½ that lands within 5 mV of either edge of its search window now raises a
  visible warning naming the window.
- `eis.RuResult` gained `arc_coverage_deg` + `is_extrapolated`; the EIS tab
  states the arc coverage and warns below 45°. Coverage is the smallest sector
  containing the points (largest-gap method) — a plain max-minus-min angle reads
  a textbook 45° arc as 359°, because its high-frequency end sits exactly on
  the ±π branch cut.

### Numerics
- `monotonic_segments` now emits genuinely maximal, vertex-sharing segments;
  `main_sweep_indices` dropped its compensating `start - 1`. The two are no
  longer silently coupled.
- `ascending_xy` drops non-finite x (NaN sorted to the end and became the
  interpolation's upper bound).
- `_ring_disk_terms` masks everything when peak |Id| <= 0 (all-zero disk used to
  give a confident 100 % H₂O₂).
- `KoutieckyLevichFit.kinetic_current_density` returns `None` for a negative
  intercept; `is_reliable` requires `intercept > 0`.
- `fit_koutecky_levich` averages duplicate rotation rates before fitting.
- `levich_slope_to_n` / `levich_current_density` validate D, ν, C as positive
  and finite (fractional powers of a negative raise deep in the arithmetic).
- `fit_tafel` accepts and forwards `current`/`e_eq` to `auto_tafel_range`, so
  its auto path can use onset detection instead of always falling back.
- `fit_slice` semantics documented (requested window; `n_points`/`decades` are
  the post-filter truth).

### App / export
- Sheet picker keyed on the upload's slot, not its filename.
- ORR sample labels de-duplicated via `_dedup_label`, shared across the ZIP and
  per-sample paths (fixes `dict(samples)` collapsing and `orr_rrde_rpms_{label}`).
- `target_js` de-duplicated on the **formatted** value, and bar-chart widgets
  keyed on the loop index.
- Export cache identity now includes format/dpi/width/height.
- ZIP member count capped at `MAX_ZIP_MEMBERS = 500`.
- Ring alignment and E½ lookups route through `sweep.ascending_xy` /
  `sweep.interp_at` instead of raw `np.interp` on possibly-duplicated x.
- `_zero_anchored_range` returns [-1, 1] for all-zero data instead of [0, 0].
- HTML export: plotly 6.9 already escapes `<`, `>` and `/` in the JSON payload,
  so the XSS-in-a-file concern does not reproduce. A regression test now pins it.

### Journal style
- `_journal_margin()` replaces every hard-coded 10 px margin dict (the reference
  EIS app's l=110/r=90/b=90, scaled with the font); `apply_plot_style` uses the
  same values.
- Ticks: `tickwidth=2, ticklen=7, tickcolor="black"`, axis `linewidth` 1.5 → 2.
- Axis titles render at tick size + 8.
- Journal column-width export presets (8.6 / 12.0 / 17.8 cm) with a live
  size caption; the pixel inputs are keyed on the preset so changing it
  actually moves them. Width-pinned figures (tables, the two-panel ring/disk
  plot) default to "As shown" so they are not re-flowed.
- Margins are now *derived* from the text they hold (rotated axis title +
  tick label + tick length), not copied from the reference app's fixed
  110/90 — those were sized for its ~930 px canvas and swallowed 62 % of an
  8.6 cm single-column figure at any font size. `apply_plot_style` and the
  hand-built figures share one derivation.
- When the margins still take >55 % of a canvas (28 pt type in a single
  column genuinely does), the download panel now says so and points at the
  font-size control, instead of silently exporting a strip of plot inside a
  wide white border.
- All raster exports go through Pillow so PNG and JPEG carry a dpi tag, not
  only TIFF. Format label is now "TIFF (LZW)" since the dpi is selectable.
- Tables: numeric columns right-aligned, interior cell rules removed (the
  comment already claimed both), canvas width clamped to the input's range.
- `show_title` is honoured by the Tafel, LSV and benchmark-bar titles, and the
  top margin follows it. The Tafel slope annotation scales with the chosen font
  instead of a fixed 22 pt.

### Export verified end-to-end (kaleido)

☑️ **kaleido 1.3.0 installed; the export path is exercised end-to-end for
the first time** (it needs no separate `plotly_get_chrome` step). All 20
format × dpi combinations render; every raster lands within 0.01 cm of the
requested 8.6 cm and carries the correct dpi tag. Rendered figures were
inspected: nothing is clipped at 10 pt / 8.6 cm or 28 pt / 17.8 cm.

Doing this immediately caught a bug in the then-new preset code: **kaleido's
width/height are CSS pixels (1/96 in) which it then multiplies by
`scale = dpi/96`**, so converting cm → px *at dpi* double-counted the
resolution and a 150 dpi single-column figure came out 13.4 cm instead of 8.6.
Physical size follows from the CSS size alone; dpi only buys resolution within
it. A parametrised end-to-end test now covers 5 formats × 3 dpi, skipped
automatically where kaleido cannot render.

That in turn exposed the margin problem: margins are absolute pixels, so the
reference app's fixed l=110/r=90 (sized for its ~930 px canvas) ate 62 % of an
8.6 cm figure at *any* font size. Margins are now derived from the text they
hold.

### Second fix pass — 2026-08-18 (all remaining MINOR items)

Everything left on the list above is now addressed. Where a finding did not
reproduce, or could not be fixed as stated, that is recorded rather than
silently dropped.

**Fixed**

- **Per-sample widgets were keyed on selection position** (`tafel_reaction_0`,
  `tafel_name_1`, ...). De-selecting the first of three samples shifted every
  later one down a slot and handed it the previous occupant's stored reaction,
  fit range, legend name and colour — attributing one sample's settings to
  another. Now keyed on `_widget_slug(label)` (labels are already unique via
  `_dedup_label`), with a hash tail so labels differing only in punctuation
  stay distinct.
- **`_abs_imag` folded inductive points onto the capacitive side.** New
  `eis.inductive_lead_count()` identifies a high-frequency inductive lead by
  the *sign* of Z'' before the magnitude is taken — the dominant sign is the
  capacitive one, so the test works under either file convention — and
  `auto_arc_range` starts after it. On a synthetic arc with a realistic
  inductive branch, **Ru error 1.108 → 0.000 ohm**. The bundled sample is
  unchanged at 27.5336 (no false positive). This needed a second fix: the
  first version passed `auto_arc_range` the already-absolute values, so the
  sign test had nothing to read and detected nothing.
- **`_stats_from_sums` cancellation.** The prefix sums are now accumulated on
  mean-centred copies and the intercept shifted back. Slope error against
  `polyfit` at a 1e6 axis offset: **1.4e-5 → 4.7e-13** (~3e7x); at 1e8 the old
  form was 100 % wrong (0.12 error on a 0.12 slope).
- **`_grow_from_onset` inapplicable overpotential cap.** When the onset was
  already past the cap (mis-assigned reaction, or e_eq from another couple),
  the cap pinned the window to its minimum width and quietly reported a slope
  fitted to five points. An inapplicable cap now drops out entirely and R^2
  alone decides.
- **`tafel.exchange_current` returned a wrong i0 on a potential axis.** It
  extrapolates to zero on whatever y-axis it was given; on V vs RHE that is
  the current at 0 V vs RHE, orders of magnitude from i0. Nothing in the
  numbers distinguishes the two axes, so `TafelResult` now carries
  `fitted_on_overpotential` (set by `fit_tafel(..., overpotential=True)`) and
  the property returns `None` unless it is declared.
- **Nice tick rounding**, ported from the reference EIS app as `_nice_axis`
  (1/2/2.5/5 x 10^n steps) plus `_square_nyquist`.
- **HTML export rebuilt on every rerun.** Measured: `to_html` is ~65 ms on a
  4-sample/4000-point figure against ~7 ms for the signature that says it was
  not needed. The signature is now computed once per figure per rerun and
  shared by the image and HTML caches, so an unchanged figure costs ~7 ms
  instead of ~72 ms — and expander bodies run even while collapsed, so this
  was being paid for every figure on the page on every widget interaction.

**Could not be fixed as stated — mitigated instead**

- **Client-side legend/annotation drags in the server-side export.** Not
  possible: `st.plotly_chart` surfaces selection events but has no API for
  relayout ones, so a dragged legend cannot be read back into the Python
  figure. Mitigated on both sides: the chart's camera button (the one path
  that *does* keep drags) now emits a 4x-scale PNG (~300 dpi at double-column
  width) instead of a screen-resolution one, and the download panel says
  plainly that drags are not in the server-rendered files and points at the
  legend-position control, which is.
- **`sweep._blocks` degenerating to 1 on tiny sweeps.** No fix exists — a
  6-point sweep does not contain enough of an end to take a median of.
  Documented as a best guess rather than a robust one.

**Judgement call**

- Squaring the Nyquist axes on the *whole* spectrum (as the reference app
  does) crushed the kinetic arc into the corner on the bundled sample: the
  diffusion tail reaches 500 ohm against a 60 ohm arc. Since that tab exists
  to judge the arc, the axes are framed on the fitted arc plus the fitted
  circle, with a checkbox for the full spectrum and a caption naming how many
  points are off-view.

### Still open

Nothing from the review remains. Two notes for future work:

- The 28/36 pt journal font sizes assume a large canvas; at 8.6 cm single
  column the margins necessarily take >55 % of the figure. The export panel
  now warns and points at the font control, but a smaller default font set
  for single-column work would be a genuine improvement.
- `_orr_data_loader`'s per-slot widgets (`orr_sample_name_{i}`) are still
  index-keyed. That one is correct: the slot exists before the user has typed
  a name, so there is no stable label to key on.

---

## Third pass — 2026-08-18 (independent re-review)

The two earlier passes were re-verified against the code (all their FIXED
claims hold) and the suite was re-run: **158 tests passed, no skips**, kaleido
exports included. Two process notes: `pytest` is not installed in `.venv`, so
the "133 tests pass" line above was not reproducible as written; and an
interrupted kaleido run leaves orphan `chrome.exe` processes that make
`test_export_renders_at_the_requested_physical_size` fail spuriously with
"Couldn't close or kill browser subprocess" — clear them and re-run.

Then a fresh review found one substantive defect the earlier passes missed.

### 🔴 The auto Tafel window optimised something it was not graded on — FIXED

`TafelResult.quality_warnings` grades a fit by **decades of current covered**
(>= 1 decade is the stated convention). Neither window search targeted that:
`_grow_from_onset` maximised R^2 greedily, `_best_r2_window` scored
`R^2 x point-count`. The auto-detector was therefore optimising a different
quantity from the one it was judged by, and it showed on both bundled samples:

| sample | before | after |
|---|---|---|
| LSV (OER) | 5 pts, 0.02 dec, 227 mV/dec, R2 0.845 | 300 pts, **1.33 dec, 191 mV/dec, R2 0.982** |
| ORR (j_k) | 18 pts, 0.18 dec, **491 mV/dec** | 86 pts, **2.65 dec, 130 mV/dec** |

491 mV/dec is physically impossible for ORR; ~130 mV/dec over 2.6 decades is
the textbook first-electron-transfer value. Both old windows were flagged
unreliable by the app, so no silently wrong number was ever displayed — but
the feature failed on the project's own sample data.

Underneath sat a concrete bug: **the `patience` counter was armed from the
first candidate**, so a window whose R^2 was still *climbing* out of the noise
of its own minimum width aborted growth. On the LSV sample R^2 ran 0.845 ->
0.906 -> 0.938 -> 0.957 — four consecutive "misses" against the 0.99
threshold, so it broke — when R^2 = 0.992 lay eleven points further on. A
rising R^2 means the window is improving, which is the opposite of the
curvature the counter existed to detect.

Fixes, all in `tafel.py`:

- New `MIN_TAFEL_DECADES = 1.0` and an R^2 ladder (threshold, then 0.98 /
  0.97 / 0.95). `_pick_by_decades` returns the first floor's widest-in-decades
  window that reaches a decade; if no floor gets there the *strictest* floor's
  answer stands, so fit quality is never traded for width the data cannot
  support.
- `_grow_from_onset` scores every candidate in one vectorised pass instead of
  growing greedily. `patience` is kept for signature compatibility and is
  documented as unused — there is no longer any growth to abandon.
- `_best_r2_window` uses the same decade-first score; `R^2 x point-count`
  rewarded windows wide in *points*, which on a densely-sampled plateau is not
  wide in current at all.
- `auto_tafel_range` accepted the onset-anchored window on **point count**
  alone, so a 5-point, 0.02-decade window was returned as confident and the
  global scan was never consulted. It now accepts only a window that reaches a
  decade, and otherwise lets the global scan compete (the onset-anchored
  window still wins ties, being anchored to physics rather than to whichever
  stretch scored best).

### ⚡ Efficiency

`_grow_from_onset` was an O(n) Python loop of scalar arithmetic over prefix
sums — 22 ms on a 16 000-point sweep, re-paid per sample on every Streamlit
rerun. Vectorised: **22.0 -> 1.8 ms at 16k (12x), 5.6 -> 0.2 ms at 4k**.
`_window_r2` is pinned to the scalar form to 1e-12 by test.

Trade-off recorded honestly: `_best_r2_window` now allocates per-start arrays
and calls `np.maximum.accumulate` over the tail for each grid start, so on
*very* large sweeps it is slower than the old scalar scan — 2.2 -> 6.4 ms at
16k, 3.6 -> 16.8 ms at 50k. It is faster at the sizes real files actually have
(1.3 ms vs 2.0 ms at 4k), it is only the fallback path, and 17 ms is
imperceptible in a Streamlit rerun, so this was left alone rather than
carrying a block-decomposition for it.

### 🟡 Also fixed

- **`orr.ring_disk_average` was tested but never called by the app.** The ORR
  tab reported n and %H2O2 interpolated at the single potential E1/2 — the
  quantity `orr.py`'s own docstring calls inferior to the plateau average the
  literature quotes. E1/2 is method-dependent (the three methods disagree by
  65 mV on the bundled file, moving n by ~0.05 and the peroxide yield by ~3
  points for no change in the data), and a single-point read-off inherits that
  spread in full. The tab now has a plateau-averaging window (default
  0.2-0.6 V vs RHE) and reports the average, the point count, and the E1/2
  value beside it for cross-checking. Non-finite results render blank instead
  of `nan`, and an empty window warns.
- **The Levich 4-electron prefactor was retyped inline twice** in `app.py`
  instead of calling `orr.levich_current_density`. Values agreed exactly; it
  was a divergence risk, now removed.
- `app.py` E128 continuation-line lint error in `_legend_dict`. flake8 clean.

### Still open (unchanged from the second pass)

- Smaller default journal font set for single-column work.
- `_orr_data_loader`'s index-keyed per-slot widgets — correct as they are.

**Test count after this pass: 165 (7 added), all passing, flake8 clean.**
