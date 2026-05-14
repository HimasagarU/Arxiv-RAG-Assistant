"""Normalize paper dates and IDs for artifacts and retrieval."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Optional


def normalize_published(value: Any) -> Optional[str]:
    """Return ISO date string YYYY-MM-DD or None if unknown."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    s = str(value).strip()
    if not s:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r"^(\d{4})-(\d{2})\b", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"
    m = re.match(r"^(\d{4})\b", s)
    if m:
        return f"{m.group(1)}-01-01"
    return None
