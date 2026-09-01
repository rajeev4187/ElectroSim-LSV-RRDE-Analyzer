"""Raw instrument-export file formats, auto-detected by file extension.

Every other loader in this project (:mod:`scripts.modules.data_io`) reads
CSV/Excel — the format every instrument's own software can export to, but
not always the format a user actually has on hand. This module adds a second
path: parsing an instrument's *native* export directly, so a file never has
to make a round trip through the vendor software first.

Supported today
----------------
- **Biologic EC-Lab** ``.mpr`` (binary) and ``.mpt`` (ASCII export), via the
  third-party ``galvani`` library (MIT-licensed, actively maintained).
- **Pine Research AfterMath** ``.paax`` project files. These are plain XML
  (verified against real exports — see ``sample-data/instrument-samples/``):
  a ``<DAB_node type="...">`` tree holding, among many GUI/metadata nodes,
  ``<stream>`` elements that each carry one measured quantity's data as a
  ``<data>`` block with ``X_units``/``Y_units`` (each tagging its physical
  quantity via ``qty_kind``, e.g. ``"time"``, ``"potential"``, ``"current"``,
  ``"resistance"``) and one or more ``<points>`` pages. AfterMath paginates
  a long trace into many small (<=100-point) ``<points>`` blocks instead of
  one — confirmed against a real chronopotentiometry export, where
  consecutive pages' time values are exactly contiguous (page *n*'s last
  time equals page *n+1*'s first) — so a stream's full series is every page
  under its ``<data>`` concatenated in document order. Every quantity is
  stored in raw SI base units regardless of its display label (a
  ``qty_kind="current"`` stream's ``qty_base`` is always ``"[I]"``, i.e.
  Amperes, never mA) — :func:`_paax_streams` converts current streams to mA
  once here so the rest of this project never has to special-case the unit.

Both formats are converted to a plain :class:`pandas.DataFrame` shaped like
Biologic's own field names (``time/s``, ``Ewe/V``, ``I/mA``, ``Re(Z)/Ohm``,
``-Im(Z)/Ohm``, …) via :func:`load_table` — the same header vocabulary the
rest of this project's column-matching (fuzzy header hints in ``data_io``
and the ORR/Chrono tabs) already understands, so neither format is a special
case downstream.

Not yet supported
------------------
CH Instruments ``.bin`` has no maintained open-source parser to build on,
and reverse-engineering an undocumented binary layout without a real file to
check the result against risks silently-wrong scientific values — worse
than refusing to load the file. :func:`load_table` raises
:class:`UnsupportedFormatError` naming the format, rather than guessing. See
``sample-data/instrument-samples/README.md`` for how to add support once a
real export file is available to verify against.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

# Extensions this module recognises as *some* instrument format, even for
# the ones with no parser yet -- so a caller can name the format in its
# error message instead of just saying "unrecognised file type".
KNOWN_EXTENSIONS: dict[str, str] = {
    ".mpr": "Biologic EC-Lab (binary export)",
    ".mpt": "Biologic EC-Lab (ASCII export)",
    ".paax": "Pine Research AfterMath",
    ".bin": "CH Instruments",
}

# Extensions with an actual parser below.
_SUPPORTED_EXTENSIONS = frozenset({".mpr", ".mpt", ".paax"})


class UnsupportedFormatError(ValueError):
    """A file extension was recognised as an instrument format, but this
    module has no parser for it yet."""


def known_format_name(filename: str) -> str | None:
    """The recognised format name for ``filename``'s extension, or ``None``
    if the extension isn't a known instrument format at all (in which case a
    caller should fall back to the ordinary CSV/Excel path)."""
    return KNOWN_EXTENSIONS.get(Path(filename).suffix.lower())


def is_supported(filename: str) -> bool:
    """Whether :func:`load_table` can actually parse ``filename``."""
    return Path(filename).suffix.lower() in _SUPPORTED_EXTENSIONS


# --------------------------------------------------------------------------- #
# Biologic EC-Lab (.mpr / .mpt), via galvani                                  #
# --------------------------------------------------------------------------- #
def _require_galvani():
    try:
        from galvani import BioLogic
    except ImportError as exc:  # pragma: no cover - exercised only when the
        # optional dependency is missing, which the requirements.txt entry
        # should prevent in a normal install.
        raise UnsupportedFormatError(
            "Reading Biologic .mpr/.mpt files needs the 'galvani' package "
            "(add it with `pip install galvani`)."
        ) from exc
    return BioLogic


def _biologic_table(file_or_path, filename: str) -> pd.DataFrame:
    BioLogic = _require_galvani()
    suffix = Path(filename).suffix.lower()
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    try:
        if suffix == ".mpr":
            record_array = BioLogic.MPRfile(file_or_path).data
        else:  # .mpt
            record_array, _comments = BioLogic.MPTfile(file_or_path)
    except Exception as exc:
        raise ValueError(f"{filename}: could not parse as a Biologic file ({exc}).") from exc
    return pd.DataFrame(record_array)


# --------------------------------------------------------------------------- #
# Pine Research AfterMath (.paax) — plain XML, parsed with the stdlib         #
# --------------------------------------------------------------------------- #
# A current-kind stream's raw values are always in Amperes (qty_base="[I]"),
# regardless of qty_kind's display label -- confirmed against a real export.
# This project's own convention (matching Biologic's "I/mA" field) is
# milliamps, so every current-kind stream is scaled once here.
_PAAX_SCALE_TO_PROJECT_UNITS = {"current": 1000.0}  # A -> mA; everything
# else (time in s, potential in V, resistance in Ohm) already matches.


def _paax_root(file_or_path) -> ET.Element:
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    return ET.parse(file_or_path).getroot()


def _paax_stream_xy(data_el: ET.Element) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate every ``<points>`` page under one stream's ``<data>``, in
    document order (see the module docstring on AfterMath's pagination).
    ``None`` if the stream carries no actual points (many GUI/derived-plot
    streams reference their data elsewhere rather than embedding it)."""
    x_parts, y_parts = [], []
    for pts in data_el.findall("points"):
        x_el, y_el = pts.find("X_data"), pts.find("Y_data")
        if x_el is None or y_el is None or not x_el.text or not y_el.text:
            continue
        x_parts.append(np.fromstring(x_el.text, sep=","))
        y_parts.append(np.fromstring(y_el.text, sep=","))
    if not x_parts:
        return None
    return np.concatenate(x_parts), np.concatenate(y_parts)


def _paax_streams(root: ET.Element):
    """Yield one dict per ``<stream>`` that carries real point data:
    ``identifier`` (Pine's short tag, e.g. ``"current"``), ``display_name``
    (e.g. ``"Measured Current"``), ``x_kind``/``y_kind`` (Pine's physical-
    quantity tags, e.g. ``"time"``/``"current"``) and the concatenated,
    unit-normalised ``x``/``y`` arrays (see
    ``_PAAX_SCALE_TO_PROJECT_UNITS``)."""
    for stream_el in root.iter("stream"):
        ident_el = stream_el.find("identifier")
        data_el = stream_el.find("data")
        if ident_el is None or data_el is None:
            continue
        xy = _paax_stream_xy(data_el)
        if xy is None:
            continue
        x, y = xy
        name_el = stream_el.find("name/ascii")
        x_units, y_units = data_el.find("X_units"), data_el.find("Y_units")
        x_kind = x_units.get("qty_kind") if x_units is not None else None
        y_kind = y_units.get("qty_kind") if y_units is not None else None
        x = x * _PAAX_SCALE_TO_PROJECT_UNITS.get(x_kind, 1.0)
        y = y * _PAAX_SCALE_TO_PROJECT_UNITS.get(y_kind, 1.0)
        yield {
            "identifier": ident_el.get("name", ""),
            "display_name": (name_el.text or "").strip() if name_el is not None else "",
            "x_kind": x_kind, "y_kind": y_kind,
            "x": x, "y": y,
        }


def _paax_find(root: ET.Element, *, identifier: str | None = None,
               x_kind: str | None = None, y_kind: str | None = None,
               name_contains: str | None = None) -> dict | None:
    """First stream (in document order) matching every given criterion, or
    ``None``. A file may hold several experiments' worth of streams; taking
    the first is a deliberate, documented simplification (see
    :func:`_paax_table`'s docstring) rather than silently merging or
    guessing which experiment the caller wants."""
    for s in _paax_streams(root):
        if identifier is not None and s["identifier"] != identifier:
            continue
        if x_kind is not None and s["x_kind"] != x_kind:
            continue
        if y_kind is not None and s["y_kind"] != y_kind:
            continue
        if name_contains is not None and name_contains.lower() not in s["display_name"].lower():
            continue
        return s
    return None


def _paax_table(file_or_path, filename: str) -> pd.DataFrame:
    """A Biologic-shaped DataFrame (``time/s``, ``Ewe/V``, ``I/mA``) built
    from an AfterMath file's *first* chronoamperometry/chronopotentiometry
    experiment — its ``Measured Potential`` and ``Measured Current`` streams
    (recorded on the same time grid, verified against a real export) rather
    than the "Applied" setpoint streams. A ``.paax`` file may contain many
    studies/experiments; only the first with time-series potential/current
    is used — for a specific one, split it in AfterMath first and export
    just that study.
    """
    root = _paax_root(file_or_path)
    pot = _paax_find(root, identifier="potential", x_kind="time")
    cur = _paax_find(root, identifier="current", x_kind="time")
    if pot is None and cur is None:
        raise ValueError(
            f"{filename}: no time-series Potential/Current stream found in "
            "this AfterMath file — is it a chronoamperometry/"
            "chronopotentiometry experiment?"
        )
    columns: dict[str, np.ndarray] = {}
    if pot is not None:
        columns["time/s"] = pot["x"]
        columns["Ewe/V"] = pot["y"]
    if cur is not None:
        columns.setdefault("time/s", cur["x"])
        columns["I/mA"] = cur["y"]
    return pd.DataFrame(columns)


def _paax_lsv(file_or_path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """``(potential, current)`` from an AfterMath file's first "Current vs
    Potential" curve (an LSV/CV/RDE polarization sweep)."""
    root = _paax_root(file_or_path)
    s = _paax_find(root, identifier="current_vs_potential")
    if s is None:
        raise ValueError(
            f"{filename}: no 'Current vs Potential' curve found in this "
            "AfterMath file — is it an LSV/CV/RDE polarization experiment?"
        )
    return s["x"], s["y"]


def _paax_eis(file_or_path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """``(z_real, z_imag)`` from an AfterMath file's first Nyquist
    (-Zi vs Zr) curve."""
    root = _paax_root(file_or_path)
    s = _paax_find(root, x_kind="resistance", y_kind="resistance",
                   name_contains="nyquist")
    if s is None:
        raise ValueError(
            f"{filename}: no Nyquist (impedance) curve found in this "
            "AfterMath file — is it an EIS experiment?"
        )
    return s["x"], s["y"]


# --------------------------------------------------------------------------- #
# Public, format-dispatching API                                              #
# --------------------------------------------------------------------------- #
def load_table(file_or_path, filename: str) -> pd.DataFrame:
    """Parse a raw instrument export into a Biologic-shaped DataFrame
    (``time/s``, ``Ewe/V``, ``I/mA``, …).

    ``file_or_path`` is a file-like object (e.g. ``io.BytesIO`` from a
    Streamlit upload) or a path string; ``filename`` supplies the extension
    that selects the parser.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        format_name = known_format_name(filename)
        if format_name is not None:
            raise UnsupportedFormatError(
                f"{filename}: {format_name} files aren't supported yet. "
                "See sample-data/instrument-samples/README.md."
            )
        raise UnsupportedFormatError(
            f"{filename}: not a recognised instrument export format."
        )

    if suffix == ".paax":
        df = _paax_table(file_or_path, filename)
    else:
        df = _biologic_table(file_or_path, filename)

    if df.empty or df.shape[1] == 0:
        raise ValueError(f"{filename}: no data columns found in the file.")
    return df


# --- convenience field lookups for the Biologic-shaped table ---------------- #
_POTENTIAL_FIELDS = ("Ewe/V", "<Ewe>/V", "Ecell/V", "<Ewe/V>")
_CURRENT_FIELDS = ("I/mA", "<I>/mA")
_ZREAL_FIELDS = ("Re(Z)/Ohm",)
_ZIMAG_FIELDS = ("-Im(Z)/Ohm",)


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> np.ndarray | None:
    for name in candidates:
        if name in df.columns:
            return df[name].to_numpy(dtype=float)
    return None


def load_eis(file_or_path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """``(z_real, z_imag)`` from a supported instrument file's impedance
    data (Biologic's Re(Z)/-Im(Z) columns, or a Pine AfterMath Nyquist
    curve)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".paax":
        return _paax_eis(file_or_path, filename)
    df = load_table(file_or_path, filename)
    z_real = _first_present(df, _ZREAL_FIELDS)
    z_imag = _first_present(df, _ZIMAG_FIELDS)
    if z_real is None or z_imag is None:
        raise ValueError(
            f"{filename}: no impedance columns (Re(Z)/Ohm, -Im(Z)/Ohm) — "
            "is this an EIS technique export?"
        )
    return z_real, z_imag


def load_lsv(file_or_path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """``(potential, current)`` from a supported instrument file's
    polarization-curve data (Biologic's Ewe/I columns, or a Pine AfterMath
    "Current vs Potential" curve)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".paax":
        return _paax_lsv(file_or_path, filename)
    df = load_table(file_or_path, filename)
    potential = _first_present(df, _POTENTIAL_FIELDS)
    current = _first_present(df, _CURRENT_FIELDS)
    if potential is None or current is None:
        raise ValueError(
            f"{filename}: no potential/current columns (Ewe/V, I/mA)."
        )
    return potential, current
