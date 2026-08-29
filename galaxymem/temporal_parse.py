"""Temporal range extraction from natural-language recall queries.

Regex-only v1 (no LLM call) — recognizes ISO dates/months, named months
with optional years, and a small set of relative expressions. Temporal
retrieval arm pattern inspired by hindsight: date-scoped queries get a
dedicated retrieval signal instead of fighting recency decay.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

Range = Tuple[datetime, datetime]

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ISO_MONTH = re.compile(r"\b(\d{4})-(\d{2})\b")
_NAMED_MONTH_YEAR = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE
)
_NAMED_MONTH = re.compile(r"\bin\s+(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)
_LAST_MONTH = re.compile(r"\blast month\b", re.IGNORECASE)
_LAST_WEEK = re.compile(r"\blast week\b", re.IGNORECASE)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)


def _month_range(year: int, month: int) -> Range:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return start, end


def parse_temporal_range(
    query: str, now: Optional[datetime] = None
) -> Optional[Range]:
    """Extract an explicit [start, end) window from a query, if any.

    Returns (start, end) as timezone-aware UTC datetimes, or None when the
    query carries no recognizable time expression.
    """
    if not query:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    m = _ISO_DATE.search(query)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            start = datetime(y, mo, d, tzinfo=timezone.utc)
        except ValueError:
            return None
        return start, start + timedelta(days=1)

    m = _ISO_MONTH.search(query)
    if m and not _ISO_DATE.search(query):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return _month_range(y, mo)

    m = _NAMED_MONTH_YEAR.search(query)
    if m:
        return _month_range(int(m.group(2)), _MONTHS[m.group(1).lower()])

    m = _NAMED_MONTH.search(query)
    if m:
        month = _MONTHS[m.group(1).lower()]
        year = now.year
        # "in july" said in January most likely means last July
        start_candidate = datetime(year, month, 1, tzinfo=timezone.utc)
        if start_candidate > now:
            year -= 1
        return _month_range(year, month)

    if _LAST_MONTH.search(query):
        first_of_this = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        prev_year, prev_month = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        return datetime(prev_year, prev_month, 1, tzinfo=timezone.utc), first_of_this

    if _LAST_WEEK.search(query):
        return now - timedelta(days=7), now

    if _YESTERDAY.search(query):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)

    return None
