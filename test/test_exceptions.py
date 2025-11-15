from __future__ import annotations

from bmri import exceptions


def test_exception_str_includes_details() -> None:
    err = exceptions.BMRIError("boom", details="extra")
    assert "extra" in str(err)

    validation = exceptions.ValidationError("bad input", details="explanation")
    assert "bad input" in str(validation)
