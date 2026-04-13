"""Helpers for importing legacy modules without scattering sys.path hacks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def ensure_project_root_on_path(root: Path | None = None) -> Path:
    """Ensure the legacy project root (with src/…) is in sys.path."""
    root = root or Path(__file__).resolve().parent.parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def import_legacy_module(module_path: str) -> Any:
    """Import a legacy module after adding project root to sys.path."""
    ensure_project_root_on_path()
    return importlib.import_module(module_path)
