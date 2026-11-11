"""Fund classifications shared by batch jobs and the lightweight web reader."""

import re
from calendar import monthrange
from datetime import date

MIN_FUND_AGE_MONTHS = 6


def fund_maturity_date(inception_date: str) -> str:
    """Six calendar months after inception, with month-end dates clamped."""
    founded = date.fromisoformat(inception_date)
    year, month = divmod(founded.year * 12 + founded.month - 1 + MIN_FUND_AGE_MONTHS, 12)
    month += 1
    return date(year, month, min(founded.day, monthrange(year, month)[1])).isoformat()


HOLD_PERIOD_RE = re.compile(
    r"(\d+|[一二两三四五六七八九十]+)\s*(年|个月|月|天)(滚动)?持有"
    r"|持有期|定开|定期开放|封闭式|封闭运作|滚动持有"
    r"|(?<![-\d])(\d+|[一二两三四五六七八九十]+)\s*(年|个月|月|天)\s*"
    r"(混合|股票|债券|债)")


def is_locked_fund(name) -> bool:
    """Funds whose redemption restrictions prevent the strategy's stop-loss."""
    return HOLD_PERIOD_RE.search(str(name)) is not None


def is_fof(fund_type, name=None) -> bool:
    """Use type first; some FOF funds omit the label from their short name."""
    return any("FOF" in str(value).upper() or "基金中基金" in str(value)
               for value in (fund_type, name))
