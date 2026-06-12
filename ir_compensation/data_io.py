"""Data loading for EIS and LSV datasets.

The reference workbook (``sample-data/Book1.xlsx``) stores:
  * Sheet 1 -> EIS data with columns ``Z'`` (real) and ``Z''`` (imaginary)
  * Sheet 2 (named "LSV") -> ``Potential`` and ``Current``

Loaders are tolerant: they match columns by fuzzy header names and fall back to
positional columns (first = x, second = y) when headers are unrecognised.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---- safety limits for untrusted uploads -----------------------------------
# A public deployment ingests arbitrary user files. These caps bound the work
# done per upload so a crafted file cannot exhaust memory/CPU (denial of
# service) or smuggle in pathological structure.
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024  # xlsx is a zip: guard zip bombs
MAX_ROWS = 500_000                          # reject absurdly tall tables
MAX_COLS = 1_000                            # reject absurdly wide tables
MAX_DATASETS = 100                          # cap parsed column-pairs
_LABEL_MAXLEN = 60


class DataValidationError(ValueError):
    """Raised when an input file violates a safety limit or is malformed."""


def _safe_label(value) -> str:
    """Sanitise a header-derived label before it is displayed or written.

    Strips control characters and newlines (which could break CSV/markdown
    output or inject rows) and truncates length.
    """
    text = "".join(c for c in str(value) if c.isprintable())
    text = text.replace("\r", " ").replace("\n", " ").strip()
    return text[:_LABEL_MAXLEN] if text else "?"


def _guard_excel_archive(src) -> None:
    """Reject Excel files that decompress to an unsafe size (zip bombs).

    .xlsx/.xlsm are ZIP archives; the upload size says nothing about how much
    they expand in memory. Sum the declared uncompressed sizes and refuse the
    file if it exceeds the cap. Leaves the stream position reset for re-reading.
    """
    seekable = hasattr(src, "seek")
    if seekable:
        src.seek(0)
    try:
        with zipfile.ZipFile(src) as zf:
            total = 0
            for info in zf.infolist():
                total += info.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise DataValidationError(
                        "Excel file expands too large; rejected as a possible "
                        "decompression bomb."
                    )
    except zipfile.BadZipFile as exc:
        raise DataValidationError("Not a valid .xlsx workbook.") from exc
    finally:
        if seekable:
            src.seek(0)


# ---- column name hints -----------------------------------------------------
_ZREAL_HINTS = ("z'", "zre", "z_re", "real", "zr")
_ZIMAG_HINTS = ('z"', "z''", "zim", "z_im", "imag", "zi")
_POT_HINTS = ("potential", "voltage", "e ", "e(", "ewe", "volt", "e/v", "e vs")
_CUR_HINTS = ("current", "i ", "i(", "i/", "amp", "j ", "j/", "i vs")


@dataclass
class EISData:
    """Nyquist-plot impedance data."""

    z_real: np.ndarray  # ohm
    z_imag: np.ndarray  # ohm (as stored in the file; sign handled at fit time)
    label: str = ""     # source column-pair label (multi-dataset sheets)

    def __len__(self) -> int:
        return len(self.z_real)


@dataclass
class LSVData:
    """Linear-sweep voltammetry data."""

    potential: np.ndarray  # V
    current: np.ndarray     # native file units (see current_unit in GUI)
    label: str = ""         # source column-pair label (multi-dataset sheets)

    def __len__(self) -> int:
        return len(self.potential)


def list_sheets(path: str | Path) -> list[str]:
    """Return the sheet names of an Excel workbook."""
    _guard_excel_archive(path)
    return pd.ExcelFile(path).sheet_names


def _match_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    lowered = {c: str(c).strip().lower() for c in columns}
    for col, name in lowered.items():
        for hint in hints:
            if name.startswith(hint) or hint in name:
                return col
    return None


def _check_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Reject tables whose size exceeds the safety caps (DoS guard)."""
    if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
        raise DataValidationError(
            f"Input too large ({df.shape[0]} rows x {df.shape[1]} cols); "
            f"limit is {MAX_ROWS} rows x {MAX_COLS} cols."
        )
    return df


def _read_table(path, sheet: str | int | None) -> pd.DataFrame:
    # In-memory buffers (e.g. Streamlit uploads) have no path/suffix; route them
    # by sheet: a sheet argument implies Excel, otherwise CSV.
    if not isinstance(path, (str, Path)):
        if sheet is None:
            return _check_shape(pd.read_csv(path))
        _guard_excel_archive(path)
        return _check_shape(pd.read_excel(path, sheet_name=sheet))
    path = Path(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return _check_shape(pd.read_csv(path))
    _guard_excel_archive(path)
    return _check_shape(
        pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    )


def _two_numeric_columns(
    df: pd.DataFrame, x_hints: tuple[str, ...], y_hints: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    cols = list(df.columns)
    x_col = _match_column(cols, x_hints)
    y_col = _match_column(cols, y_hints)
    # Fall back to the first two columns if headers are not recognised.
    if x_col is None or y_col is None or x_col == y_col:
        if len(cols) < 2:
            raise ValueError("Expected at least two columns of data.")
        x_col, y_col = cols[0], cols[1]
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        raise ValueError("No numeric data rows found.")
    return x[mask], y[mask]


def load_eis(path: str | Path, sheet: str | int | None = 0) -> EISData:
    """Load EIS data (Z', Z'') from a workbook sheet or CSV file."""
    df = _read_table(path, sheet)
    zr, zi = _two_numeric_columns(df, _ZREAL_HINTS, _ZIMAG_HINTS)
    return EISData(z_real=zr, z_imag=zi)


def load_lsv(path: str | Path, sheet: str | int | None = 1) -> LSVData:
    """Load LSV data (Potential, Current) from a workbook sheet or CSV file."""
    df = _read_table(path, sheet)
    pot, cur = _two_numeric_columns(df, _POT_HINTS, _CUR_HINTS)
    return LSVData(potential=pot, current=cur)


# --- multiple datasets stored as repeated column pairs ---------------------- #
def _iter_column_pairs(df: pd.DataFrame):
    """Yield ``(label, x, y)`` for each consecutive pair of columns.

    A sheet may hold several datasets side by side, e.g.
    ``Z'_1, Z''_1, Z'_2, Z''_2, ...`` (EIS) or
    ``Potential_1, Current_1, Potential_2, Current_2, ...`` (LSV). Columns are
    taken two at a time; each pair is cleaned of non-numeric / NaN rows
    independently (datasets may differ in length). The label is built from the
    pair's headers.
    """
    cols = list(df.columns)
    emitted = 0
    for k in range(0, len(cols) - 1, 2):
        if emitted >= MAX_DATASETS:
            break
        x_col, y_col = cols[k], cols[k + 1]
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        label = f"{_safe_label(x_col)} / {_safe_label(y_col)}"
        emitted += 1
        yield label, x[mask], y[mask]


def load_eis_datasets(path: str | Path, sheet: str | int | None = 0) -> list[EISData]:
    """Load one or more EIS datasets (one per ``Z', Z''`` column pair)."""
    df = _read_table(path, sheet)
    out = [EISData(z_real=x, z_imag=y, label=lab)
           for lab, x, y in _iter_column_pairs(df)]
    if not out:
        raise ValueError("No EIS column pairs (Z', Z'') found in the sheet.")
    return out


def load_lsv_datasets(path: str | Path, sheet: str | int | None = 1) -> list[LSVData]:
    """Load one or more LSV datasets (one per ``Potential, Current`` pair)."""
    df = _read_table(path, sheet)
    out = [LSVData(potential=x, current=y, label=lab)
           for lab, x, y in _iter_column_pairs(df)]
    if not out:
        raise ValueError("No LSV column pairs (Potential, Current) found.")
    return out
