# -*- coding: utf-8 -*-
"""Imported first by every script: makes `cdkit`, the sibling scripts and the
external companion code importable regardless of the working directory.
All locations are configured through environment variables (see cdkit/paths.py)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from cdkit.paths import add_sys_paths  # noqa: E402

add_sys_paths()
