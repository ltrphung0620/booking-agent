"""Reuse the existing demo without copying its model, tools or agent loop."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
