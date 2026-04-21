"""Confidence decay: unreinforced concepts fade over time."""

from __future__ import annotations

import math
from datetime import datetime, timezone


DEFAULT_HALF_LIFE_DAYS = 62.0
STALE_CONFIDENCE = 0.3
STALE_AGE_DAYS = 90.0


def age_days(iso: str | None) -> float:
    """Days since the given ISO 8601 timestamp, or a large sentinel when None."""
    if not iso:
        return 365 * 10
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 365 * 10
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max((now - ts).total_seconds() / 86400.0, 0.0)


def adjusted_confidence(
    base: float,
    last_reinforced_at: str | None,
    source_count: int,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Return a decayed confidence in [0, 1]."""
    age = age_days(last_reinforced_at)
    decay = math.exp(-age / half_life_days)
    reinforce = min(1.0, math.log2(1 + max(source_count, 0)) / 4)
    factor = 0.5 + 0.5 * max(decay, reinforce)
    return round(max(0.0, min(1.0, base * factor)), 3)


def is_stale(adjusted: float, age_days_value: float) -> bool:
    """Stale = low adjusted confidence AND long since reinforcement."""
    return adjusted < STALE_CONFIDENCE and age_days_value > STALE_AGE_DAYS
