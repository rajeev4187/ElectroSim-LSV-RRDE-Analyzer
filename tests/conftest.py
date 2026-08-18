"""Make the repository root importable so ``scripts.modules`` resolves.

The project is a Streamlit app run from its checkout rather than an installed
package, so there is no console entry point or site-packages install for
pytest to pick up.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
