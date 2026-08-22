"""Tests for the temporal range parser and recall arm."""

from datetime import datetime, timedelta, timezone

from galaxymem.temporal_parse import parse_temporal_range


NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)


def test_iso_date_single_day():
    r = parse_temporal_range("what was true on 2026-07-15", now=NOW)
    assert r is not None
    start, end = r
    assert start.day == 15 and start.month == 7 and start.year == 2026
    assert (end - start).days == 1


def test_iso_month_expands_to_month():
    r = parse_temporal_range("what did i say in 2026-07", now=NOW)
    assert r is not None
    start, end = r
    assert start.day == 1 and start.month == 7
    assert end.month == 8 and end.day == 1


def test_named_month():
    r = parse_temporal_range("what was true in july", now=NOW)
    assert r is not None
    start, end = r
    assert start.month == 7 and start.day == 1
    assert end.month == 8


def test_named_month_with_year():
    r = parse_temporal_range("memories from march 2025", now=NOW)
    assert r is not None
    start, end = r
    assert (start.year, start.month) == (2025, 3)
    assert (end.year, end.month) == (2025, 4)


def test_last_month_relative():
    r = parse_temporal_range("what was true last month", now=NOW)
    assert r is not None
    start, end = r
    # August now → July 1..Aug 1
    assert (start.year, start.month) == (2026, 7)
    assert (end.year, end.month) == (2026, 8)


def test_last_week_relative():
    r = parse_temporal_range("what happened last week", now=NOW)
    assert r is not None
    start, end = r
    assert (NOW - start).days == 7
    assert end == NOW


def test_yesterday():
    r = parse_temporal_range("what did we do yesterday", now=NOW)
    assert r is not None
    start, end = r
    assert start.day == 21 and end.day == 22


def test_no_temporal_content_returns_none():
    assert parse_temporal_range("what do you know about the api", now=NOW) is None
    assert parse_temporal_range("remember the deploy deadline", now=NOW) is None


if __name__ == "__main__":
    print("run via pytest")
