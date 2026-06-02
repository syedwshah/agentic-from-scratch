"""Tests for the tool layer. Run with:

    uv run pytest -q
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tools import dispatch_tool, get_nanotime


def test_get_nanotime_returns_recent_ns() -> None:
    result = get_nanotime()
    assert "nanotime" in result
    assert isinstance(result["nanotime"], int)
    # ns since 1970 is currently > 1.7e18; this is a sanity floor.
    assert result["nanotime"] > 10**18


def test_dispatch_unknown_tool_raises_key_error() -> None:
    with pytest.raises(KeyError):
        dispatch_tool("does_not_exist", {})


# ---------------------------------------------------------------------------
# TODO (live): once you implement day_of_week_from_nanotime() and translate(),
# uncomment these tests. They double as a TDD checkpoint.
# ---------------------------------------------------------------------------

def test_day_of_week_from_nanotime_is_tuesday_on_2026_06_02() -> None:
    # 2026-06-02 (UTC) is a Tuesday
    seconds = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc).timestamp()
    ns = int(seconds * 1_000_000_000)
    out = dispatch_tool("day_of_week_from_nanotime", {"nanotime": ns})
    assert out == {"day_of_week": "Tuesday"}


def test_translate_tuesday_to_spanish() -> None:
    out = dispatch_tool(
        "translate",
        {"text": "Tuesday", "target_language": "spanish"},
    )
    assert out["translated"] == "martes"


def test_translate_unsupported_language_returns_error_payload() -> None:
    out = dispatch_tool(
        "translate",
        {"text": "Tuesday", "target_language": "klingon"},
    )
    assert "error" in out
    assert "supported" in out
