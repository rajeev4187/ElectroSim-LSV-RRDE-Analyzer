# Instrument sample files

Drop real instrument-export files here to enable/verify their parser in
`scripts/modules/instrument_formats.py`:

- **Biologic** — `.mpr` or `.mpt` (supported, via `galvani`).
- **Pine Research AfterMath** — `.paax` (supported — turns out to be plain
  XML, parsed with the stdlib, no extra dependency; verified against the
  chrono/ORR/EIS examples dropped here).
- **CH Instruments** — `.bin` (not yet supported; needs a real file here
  before it can be trusted for real data).

Everything in this folder is git-ignored (see `.gitignore`) — it is local
verification data, not published with the repo.
