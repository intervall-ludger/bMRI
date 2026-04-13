"""Imports all bmri modules so that mypy checks them via the test suite.

This avoids the duplicate-module-name problem between ``src.bmri`` and
``bmri`` while still ensuring that mypy sees the full public surface of
the package when run with ``python -m mypy``.
"""

from __future__ import annotations

import bmri  # noqa: F401
from bmri import config, exceptions, logger, types, validators  # noqa: F401
from bmri.cli import main as cli_main  # noqa: F401
from bmri.cli.commands import fit as cli_fit  # noqa: F401
from bmri.cli.commands import view as cli_view  # noqa: F401
from bmri.io import readers  # noqa: F401

