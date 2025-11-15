from __future__ import annotations

import pytest
from typer.testing import CliRunner

from bmri.cli import main


def test_cli_help_shows_description() -> None:
    runner = CliRunner()
    result = runner.invoke(main.app, ["--help"])
    assert result.exit_code == 0
    assert "Bio-sensitive MRI Analysis Framework" in result.stdout


def test_cli_entry_point_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyApp:
        def __call__(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(main, "app", DummyApp())

    with pytest.raises(SystemExit) as excinfo:
        main.cli_entry_point()

    assert excinfo.value.code == 130


def test_cli_entry_point_handles_generic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyApp:
        def __call__(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(main, "app", DummyApp())

    with pytest.raises(SystemExit) as excinfo:
        main.cli_entry_point()

    assert excinfo.value.code == 1
