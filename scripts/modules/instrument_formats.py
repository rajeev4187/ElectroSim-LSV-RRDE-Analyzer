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

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

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


def _paax_stream_from_element(stream_el: ET.Element) -> dict | None:
    """Parse one ``<stream>`` element into ``identifier`` (Pine's short tag,
    e.g. ``"current"``), ``display_name`` (e.g. ``"Measured Current"``),
    ``x_kind``/``y_kind`` (Pine's physical-quantity tags, e.g.
    ``"time"``/``"current"``) and the concatenated, unit-normalised ``x``/
    ``y`` arrays (see ``_PAAX_SCALE_TO_PROJECT_UNITS``). ``None`` if the
    stream carries no identifier or no actual points (many GUI/derived-plot
    streams reference their data elsewhere rather than embedding it)."""
    ident_el = stream_el.find("identifier")
    data_el = stream_el.find("data")
    if ident_el is None or data_el is None:
        return None
    xy = _paax_stream_xy(data_el)
    if xy is None:
        return None
    x, y = xy
    name_el = stream_el.find("name/ascii")
    x_units, y_units = data_el.find("X_units"), data_el.find("Y_units")
    x_kind = x_units.get("qty_kind") if x_units is not None else None
    y_kind = y_units.get("qty_kind") if y_units is not None else None
    x = x * _PAAX_SCALE_TO_PROJECT_UNITS.get(x_kind, 1.0)
    y = y * _PAAX_SCALE_TO_PROJECT_UNITS.get(y_kind, 1.0)
    return {
        "identifier": ident_el.get("name", ""),
        "display_name": (name_el.text or "").strip() if name_el is not None else "",
        "x_kind": x_kind, "y_kind": y_kind,
        "x": x, "y": y,
    }


def _paax_streams(root: ET.Element):
    """Yield one dict per ``<stream>`` in the whole document that carries
    real point data (see :func:`_paax_stream_from_element`) — every
    experiment's streams flattened together, in document order."""
    for stream_el in root.iter("stream"):
        s = _paax_stream_from_element(stream_el)
        if s is not None:
            yield s


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


# --------------------------------------------------------------------------- #
# Per-experiment candidates (a .paax file usually bundles several)            #
# --------------------------------------------------------------------------- #
# AfterMath's DAB tree nests one "experiment" node inside each named "study"
# node, which itself holds the "X_results" node the actual data streams live
# under (study -> experiment -> X_results -> streams -> stream). Confirmed
# across every sample .paax on hand: the *experiment* node's own <name> is
# always the literal placeholder text "experiment" -- never a real name --
# while the *study* node one level up carries AfterMath's own generated
# label, e.g. "RRDE Experiment[ 5b]2026-JUL-08 1513[ 23]20]" (technique +
# timestamp, with some internal bracket noise from AfterMath's own numbering
# left in deliberately: some real exports hold several same-technique,
# same-minute studies distinguished only by that trailing number). A study's
# own <name> is direct URL-encoded text (unlike a <stream>'s
# <name><ascii>...</ascii></name>), so it needs unquoting.
def _paax_node_index(root: ET.Element) -> dict[str, ET.Element]:
    """``DAB_node`` index -> element, for every node in the file."""
    return {el.get("index"): el for el in root.iter("DAB_node") if el.get("index")}


def _paax_node_name(el: ET.Element | None) -> str | None:
    """A DAB_node's own decoded ``<name>``, or ``None`` if it has none. The
    raw text is URL-encoded, so trailing whitespace (e.g. a literal ``%20``)
    only shows up *after* unquoting — stripped there, not before."""
    if el is None:
        return None
    name_el = el.find("name")
    if name_el is None or not name_el.text:
        return None
    decoded = unquote(name_el.text).strip()
    return decoded or None


def _paax_experiment_label(nodes: dict[str, ET.Element], experiment_idx: str | None) -> str:
    """Human-readable label for one experiment — its parent study's own
    name (technique + timestamp), or a bare index if even that is missing
    (has happened for an aborted run that never got far enough to be
    named)."""
    exp_el = nodes.get(experiment_idx) if experiment_idx else None
    study_idx = exp_el.get("parent") if exp_el is not None else None
    name = _paax_node_name(nodes.get(study_idx)) if study_idx else None
    return name or (f"Experiment {experiment_idx}" if experiment_idx else "Experiment")


def _paax_matches_per_experiment(root: ET.Element, *,
                                 identifier: str | tuple[str, ...] | None = None,
                                 x_kind: str | None = None, y_kind: str | None = None,
                                 name_contains: str | None = None):
    """Yield ``(experiment_index, label, stream)`` for the first stream
    within *each* experiment (``X_results`` block) matching every given
    criterion, in document order — one candidate per experiment that has a
    match, rather than :func:`_paax_find`'s single first-in-the-whole-file
    match. This is what lets a caller offer "which experiment?" instead of
    silently taking whichever happens to come first — including, in a real
    multi-technique export, a failed or aborted one that merely happens to
    sort earliest.

    ``identifier`` accepts a tuple of Pine's own stream-identifier tags, not
    just one: a technique that produces the same *shape* of curve can still
    tag it differently -- a real export on hand carries a DPV (differential
    pulse voltammetry) curve as ``"differential_current_vs_potential"``,
    distinct from CV/LSV's plain ``"current_vs_potential"``, and both are
    genuine potential-vs-current sweeps a caller may want to accept. This is
    deliberately *not* done by matching on ``x_kind``/``y_kind`` alone
    (``"potential"``/``"current"``) instead of an identifier allow-list: the
    same real export also carries hundreds of EIS Lissajous-loop streams in
    that exact shape, which are not a polarization curve at all and would
    flood a picker meant for one.
    """
    identifiers = (
        {identifier} if isinstance(identifier, str)
        else set(identifier) if identifier is not None else None
    )
    nodes = _paax_node_index(root)
    for xr in root.iter("DAB_node"):
        if xr.get("type") != "X_results":
            continue
        exp_idx = xr.get("parent")
        for stream_el in xr.iter("stream"):
            s = _paax_stream_from_element(stream_el)
            if s is None:
                continue
            if identifiers is not None and s["identifier"] not in identifiers:
                continue
            if x_kind is not None and s["x_kind"] != x_kind:
                continue
            if y_kind is not None and s["y_kind"] != y_kind:
                continue
            if name_contains is not None and name_contains.lower() not in s["display_name"].lower():
                continue
            yield exp_idx, _paax_experiment_label(nodes, exp_idx), s
            break  # first matching stream within this experiment is enough


def _paax_experiment_count(root: ET.Element) -> int:
    """How many experiments (``X_results`` blocks) this file has at all --
    for an honest "searched N experiments, found none" message rather than
    a bare "not found" that leaves a reader unsure whether the search was
    thorough."""
    return sum(1 for xr in root.iter("DAB_node") if xr.get("type") == "X_results")


# Pine's own stream-identifier tags for a potential-vs-current sweep. Not
# just "current_vs_potential" (CV/LSV/RDE/RRDE) -- see
# :func:`_paax_matches_per_experiment`'s docstring for why DPV's own
# ``differential_current_vs_potential`` belongs here too, and why this is an
# identifier allow-list rather than a shape (x_kind/y_kind) match.
_LSV_STREAM_IDENTIFIERS = ("current_vs_potential", "differential_current_vs_potential")


def list_eis_candidates(file_or_path, filename: str) -> list[dict]:
    """``[{"label": ..., "z_real": ..., "z_imag": ...}, ...]`` — one entry
    per usable experiment. A Biologic file always yields exactly one entry
    (one technique per file); a Pine AfterMath file yields one per
    experiment with a Nyquist (-Zi vs Zr) curve, labelled by its study's own
    name — build a picker from this when there is more than one.
    """
    suffix = Path(filename).suffix.lower()
    if suffix != ".paax":
        z_real, z_imag = load_eis(file_or_path, filename)
        return [{"label": filename, "z_real": z_real, "z_imag": z_imag}]
    root = _paax_root(file_or_path)
    out = [
        {"label": label, "z_real": s["x"], "z_imag": s["y"]}
        for _, label, s in _paax_matches_per_experiment(
            root, x_kind="resistance", y_kind="resistance", name_contains="nyquist")
    ]
    if not out:
        n_exp = _paax_experiment_count(root)
        raise ValueError(
            f"{filename}: searched {n_exp} experiment(s) in this AfterMath "
            "file and found no Nyquist (impedance) curve in any of them — "
            "is it an EIS experiment?"
        )
    return out


def list_lsv_candidates(file_or_path, filename: str) -> list[dict]:
    """``[{"label": ..., "potential": ..., "current": ...}, ...]`` — one
    entry per usable experiment. A Biologic file always yields exactly one
    entry; a Pine AfterMath file yields one per experiment with a
    potential-vs-current sweep curve (LSV/CV/RDE/RRDE's
    ``"current_vs_potential"`` or DPV's own
    ``"differential_current_vs_potential"`` — see
    :data:`_LSV_STREAM_IDENTIFIERS`), labelled by its study's own name.
    """
    suffix = Path(filename).suffix.lower()
    if suffix != ".paax":
        potential, current = load_lsv(file_or_path, filename)
        return [{"label": filename, "potential": potential, "current": current}]
    root = _paax_root(file_or_path)
    out = [
        {"label": label, "potential": s["x"], "current": s["y"]}
        for _, label, s in _paax_matches_per_experiment(
            root, identifier=_LSV_STREAM_IDENTIFIERS)
    ]
    if not out:
        n_exp = _paax_experiment_count(root)
        raise ValueError(
            f"{filename}: searched {n_exp} experiment(s) in this AfterMath "
            "file and found no Current-vs-Potential (or Differential "
            "Current-vs-Potential) curve in any of them — is it an "
            "LSV/CV/DPV/RDE/RRDE polarization experiment?"
        )
    return out


_RPM_IN_NAME = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*rpm\s*\)?", re.IGNORECASE)


def _paax_rpm_from_name(name: str) -> float | None:
    """A rotation rate parsed out of a stream's own display name, e.g.
    ``"Disk Current vs Disk Potential (2000 RPM)"`` -> ``2000.0`` — Pine
    bakes it directly into the name for an RRDE/RDE experiment, so it does
    not have to be guessed from a filename or typed by hand."""
    m = _RPM_IN_NAME.search(name)
    return float(m.group(1)) if m else None


def _paax_rrde_streams_per_experiment(root: ET.Element):
    """Yield ``(experiment_index, label, disk_stream, ring_stream_or_None,
    rpm_or_None)`` for every experiment with at least one potential-vs-
    current stream.

    A real RRDE experiment records **two** streams under the identical
    ``current_vs_potential`` identifier within the same experiment — Pine
    names them ``"Disk Current vs Disk Potential (...)"`` and ``"Ring
    Current vs Disk Potential (...)"`` (confirmed against a real RRDE
    export; the ring stream shares the disk's own potential axis exactly,
    not a separate one). :func:`_paax_matches_per_experiment` stops at the
    first match per experiment, which is exactly right for a plain LSV/CV
    file (there is only one) but silently drops the ring current entirely
    for an RRDE one — this collects every matching stream per experiment
    instead and tells disk from ring by each stream's own name.
    """
    nodes = _paax_node_index(root)
    identifiers = set(_LSV_STREAM_IDENTIFIERS)
    for xr in root.iter("DAB_node"):
        if xr.get("type") != "X_results":
            continue
        exp_idx = xr.get("parent")
        matches = []
        for stream_el in xr.iter("stream"):
            s = _paax_stream_from_element(stream_el)
            if s is not None and s["identifier"] in identifiers:
                matches.append(s)
        if not matches:
            continue
        ring = next((s for s in matches if "ring" in s["display_name"].lower()), None)
        disk = next((s for s in matches if s is not ring), matches[0])
        rpm = _paax_rpm_from_name(disk["display_name"])
        if rpm is None and ring is not None:
            rpm = _paax_rpm_from_name(ring["display_name"])
        yield exp_idx, _paax_experiment_label(nodes, exp_idx), disk, ring, rpm


def list_rrde_candidates(file_or_path, filename: str) -> list[dict]:
    """``[{"label": ..., "potential": ..., "disk_current": ...,
    "ring_current": ... or None, "rpm": ... or None}, ...]`` — one entry
    per usable experiment, keeping **both** channels when an RRDE
    experiment records both (unlike :func:`list_lsv_candidates`, which
    keeps only the first potential-vs-current stream per experiment and so
    silently drops the ring current). Also recovers the rotation rate when
    Pine's own stream name states it, e.g. ``"...( 2000 RPM)"``.

    A Biologic file always yields exactly one entry with ``ring_current``
    and ``rpm`` both ``None`` — Biologic's own RRDE recording isn't
    disambiguated the same way here and is treated as disk-only, matching
    how the rest of this project already handles a raw two-column
    instrument file elsewhere.
    """
    suffix = Path(filename).suffix.lower()
    if suffix != ".paax":
        potential, current = load_lsv(file_or_path, filename)
        return [{"label": filename, "potential": potential, "disk_current": current,
                 "ring_current": None, "rpm": None}]
    root = _paax_root(file_or_path)
    out = [
        {
            "label": label, "potential": disk["x"], "disk_current": disk["y"],
            "ring_current": ring["y"] if ring is not None else None,
            "rpm": rpm,
        }
        for _, label, disk, ring, rpm in _paax_rrde_streams_per_experiment(root)
    ]
    if not out:
        n_exp = _paax_experiment_count(root)
        raise ValueError(
            f"{filename}: searched {n_exp} experiment(s) in this AfterMath "
            "file and found no Current-vs-Potential (or Differential "
            "Current-vs-Potential) curve in any of them — is it an "
            "LSV/CV/DPV/RDE/RRDE polarization experiment?"
        )
    return out


def list_chrono_candidates(file_or_path, filename: str) -> list[dict]:
    """``[{"label": ..., "time_s": ..., "potential": ... or None,
    "current": ... or None}, ...]`` — one entry per usable experiment,
    carrying whichever of the time-series Potential/Current streams that
    experiment actually has (both, for a typical chrono run; the caller
    picks whichever the chosen technique — CA or CP — needs). A Biologic
    file always yields exactly one entry.
    """
    suffix = Path(filename).suffix.lower()
    if suffix != ".paax":
        df = load_table(file_or_path, filename)
        time_s = df["time/s"].to_numpy(dtype=float) if "time/s" in df.columns else None
        potential = df["Ewe/V"].to_numpy(dtype=float) if "Ewe/V" in df.columns else None
        current = df["I/mA"].to_numpy(dtype=float) if "I/mA" in df.columns else None
        if time_s is None or (potential is None and current is None):
            raise ValueError(f"{filename}: no time-series potential/current data found.")
        return [{"label": filename, "time_s": time_s, "potential": potential, "current": current}]

    root = _paax_root(file_or_path)
    pot_by_exp = {
        idx: (label, s) for idx, label, s in
        _paax_matches_per_experiment(root, identifier="potential", x_kind="time")
    }
    cur_by_exp = {
        idx: (label, s) for idx, label, s in
        _paax_matches_per_experiment(root, identifier="current", x_kind="time")
    }
    exp_indices = sorted(set(pot_by_exp) | set(cur_by_exp), key=lambda i: int(i))
    out = []
    for idx in exp_indices:
        pot_entry, cur_entry = pot_by_exp.get(idx), cur_by_exp.get(idx)
        label = (pot_entry or cur_entry)[0]
        out.append({
            "label": label,
            "time_s": (pot_entry or cur_entry)[1]["x"],
            "potential": pot_entry[1]["y"] if pot_entry else None,
            "current": cur_entry[1]["y"] if cur_entry else None,
        })
    if not out:
        n_exp = _paax_experiment_count(root)
        raise ValueError(
            f"{filename}: searched {n_exp} experiment(s) in this AfterMath "
            "file and found no time-series Potential/Current stream in any "
            "of them — is it a chronoamperometry/chronopotentiometry "
            "experiment?"
        )
    return out


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
