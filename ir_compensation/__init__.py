"""Automated iR compensation from EIS fitting and LSV data.

Public API
----------
- data_io   : load EIS (Z', Z'') and LSV (Potential, Current) from Excel/CSV
- eis       : fit the EIS Nyquist arc and extract the uncompensated resistance Ru
- correction: apply ohmic-drop (iR) correction to LSV data
"""

from . import data_io, eis, correction

__all__ = ["data_io", "eis", "correction"]
__version__ = "1.0.0"
