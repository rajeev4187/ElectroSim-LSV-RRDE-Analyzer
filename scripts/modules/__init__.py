"""Automated iR compensation and Tafel-slope analysis from EIS and LSV data.

Public API
----------
- data_io   : load EIS (Z', Z'') and LSV/polarization (Potential, Current) from Excel/CSV
- eis       : fit the EIS Nyquist arc and extract the uncompensated resistance Ru
- correction: apply ohmic-drop (iR) correction to LSV data
- tafel     : fit the Tafel slope from a polarization curve
"""

from . import data_io, eis, correction, tafel

__all__ = ["data_io", "eis", "correction", "tafel"]
__version__ = "1.0.0"
