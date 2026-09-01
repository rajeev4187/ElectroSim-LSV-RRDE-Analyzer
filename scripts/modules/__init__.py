"""Automated iR compensation, Tafel-slope, and ORR/RRDE analysis from EIS,
LSV, and RRDE data.

Public API
----------
- data_io   : load EIS (Z', Z'') and LSV/polarization (Potential, Current) from Excel/CSV
- eis       : fit the EIS Nyquist arc and extract the uncompensated resistance Ru
- correction: apply ohmic-drop (iR) correction to LSV data
- tafel     : fit the Tafel slope from a polarization curve
- orr       : RRDE oxygen-reduction analysis (onset/E1/2, electron number, peroxide yield)
- chrono    : chronoamperometry/chronopotentiometry retention & stability metrics
- formatting: chemical-formula subscript/superscript text formatting
- instrument_formats: raw instrument export files (currently Biologic .mpr/.mpt)
"""

from . import (
    chrono, correction, data_io, eis, formatting, instrument_formats, orr, tafel,
)

__all__ = [
    "correction", "data_io", "eis", "orr", "tafel", "chrono", "formatting",
    "instrument_formats",
]
__version__ = "1.2.0"
