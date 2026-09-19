"""Turning raw daily rows into a weekly table the model can learn from.

Why weekly: the app answers "what should I order for the next two weeks", so the
target is a two-week total. Forecasting a two-week total directly is far steadier
than forecasting 14 separate days and adding them up, and it matches how a small
kitchen actually places orders.

The feature set encodes the two signals the brief asked for:
  recent trend        lags and rolling means over the last 4 and 13 weeks
  the year before     the same week last year, and the mean of the surrounding
                      five weeks, which stands in for "that month last year"
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

HORIZON_WEEKS = 2
HORIZON_DAYS = HORIZON_WEEKS * 7

TARGET = "target_next_2w"

FEATURE_COLUMNS = [
    "lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "roll4_mean", "roll8_mean", "roll13_mean", "roll4_std",
    "trend_ratio", "momentum",
    "ly_same_week", "ly_month_mean", "ly_ratio",
    "waste_rate_4w", "sell_through_4w",
    "price_rel", "promo_rate_4w",
    "week_of_year", "week_sin", "week_cos", "month",
    "holiday_weeks_ahead", "weeks_of_history",
    "shelf_life_days", "unit_cost_cat",
    "sku", "category",
]

CATEGORICAL = ["sku", "category"]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth given weekday of a month. n = -1 means the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    next_month = date(year + (month == 12), (month % 12) + 1, 1)
    last = next_month - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm. Easter shifts food demand a lot."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def holiday_dates(year: int) -> list[date]:
    """Dates that reliably move food volume in a US small business."""
    return [
        date(year, 1, 1),                        # New Year's Day
        date(year, 2, 14),                       # Valentine's Day
        _easter(year),
        _nth_weekday(year, 5, 6, 2),             # Mother's Day, 2nd Sunday in May
        _nth_weekday(year, 5, 0, -1),            # Memorial Day
        _nth_weekday(year, 6, 6, 3),             # Father's Day
        date(year, 7, 4),                        # Independence Day
        _nth_weekday(year, 9, 0, 1),             # Labor Day
        date(year, 10, 31),                      # Halloween
        _nth_weekday(year, 11, 3, 4),            # Thanksgiving
        date(year, 12, 24),
        date(year, 12, 25),
        date(year, 12, 31),
    ]


def week_start(values: pd.Series) -> pd.Series:
    """Monday of the week each date falls in."""
    dates = pd.to_datetime(values)
    return dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")


def holiday_weeks(years: list[int]) -> set[pd.Timestamp]:
    """Monday of every week containing a holiday."""
    out: set[pd.Timestamp] = set()
    for year in years:
        for day in holiday_dates(year):
            stamp = pd.Timestamp(day)
            out.add(stamp - pd.Timedelta(days=stamp.dayofweek))
    return out


def to_weekly(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows into one row per item per week."""
    if history.empty:
        return pd.DataFrame(columns=["sku", "week", "units_sold", "units_bought",
                                     "units_wasted", "unit_cost", "is_promo",
                                     "closing_stock", "days_recorded"])

    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["week"] = week_start(frame["date"])
    frame["is_promo"] = frame["is_promo"].astype(float)

    weekly = frame.groupby(["sku", "week"], as_index=False).agg(
        units_sold=("units_sold", "sum"),
        units_bought=("units_bought", "sum"),
        units_wasted=("units_wasted", "sum"),
        unit_cost=("unit_cost", "mean"),
        is_promo=("is_promo", "mean"),
        closing_stock=("closing_stock", "last"),
        days_recorded=("date", "nunique"),
    )
    return weekly.sort_values(["sku", "week"]).reset_index(drop=True)


def _fill_week_gaps(weekly: pd.DataFrame) -> pd.DataFrame:
    """Insert missing weeks as zero-sales so lags line up with real time.

    Without this, a shop that closed for two weeks would have lag_1w pointing at
    a week a month earlier, and every seasonal feature would drift.
    """
    if weekly.empty:
        return weekly
    filled = []
    for sku, group in weekly.groupby("sku", sort=False):
        span = pd.date_range(group["week"].min(), group["week"].max(), freq="7D")
        group = group.set_index("week").reindex(span)
        group["sku"] = sku
        group["units_sold"] = group["units_sold"].fillna(0.0)
        group["units_bought"] = group["units_bought"].fillna(0.0)
        group["units_wasted"] = group["units_wasted"].fillna(0.0)
        group["is_promo"] = group["is_promo"].fillna(0.0)
        group["days_recorded"] = group["days_recorded"].fillna(0)
        group["unit_cost"] = group["unit_cost"].ffill().bfill()
        group["closing_stock"] = group["closing_stock"].ffill()
        filled.append(group.rename_axis("week").reset_index())
    return pd.concat(filled, ignore_index=True).sort_values(["sku", "week"]).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, fill: float = 1.0) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def build_features(history: pd.DataFrame, catalog: pd.DataFrame,
                   *, with_target: bool = True) -> pd.DataFrame:
    """Build the weekly modelling table.

    Every feature on a row uses only data from that week or earlier, so a row can
    be used to predict the two weeks that follow it without leaking the answer.
    """
    weekly = _fill_week_gaps(to_weekly(history))
    if weekly.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET, "week", "sku"])

    rows = []
    for sku, group in weekly.groupby("sku", sort=False):
        group = group.sort_values("week").reset_index(drop=True)
        sold = group["units_sold"]

        for lag in (1, 2, 3, 4):
            group[f"lag_{lag}w"] = sold.shift(lag)

        group["roll4_mean"] = sold.shift(1).rolling(4, min_periods=1).mean()
        group["roll8_mean"] = sold.shift(1).rolling(8, min_periods=1).mean()
        group["roll13_mean"] = sold.shift(1).rolling(13, min_periods=1).mean()
        group["roll4_std"] = sold.shift(1).rolling(4, min_periods=2).std()

        # Recent level against the 3-month level: above 1 means demand is climbing.
        group["trend_ratio"] = _safe_ratio(group["roll4_mean"], group["roll13_mean"])
        group["momentum"] = _safe_ratio(group["roll4_mean"], group["roll8_mean"])

        # Same week a year ago, and the month around it.
        group["ly_same_week"] = sold.shift(52)
        surrounding = pd.concat([sold.shift(s) for s in (50, 51, 52, 53, 54)], axis=1)
        group["ly_month_mean"] = surrounding.mean(axis=1)
        group["ly_ratio"] = _safe_ratio(group["ly_same_week"], group["ly_month_mean"])

        bought4 = group["units_bought"].shift(1).rolling(4, min_periods=1).sum()
        wasted4 = group["units_wasted"].shift(1).rolling(4, min_periods=1).sum()
        sold4 = sold.shift(1).rolling(4, min_periods=1).sum()
        group["waste_rate_4w"] = _safe_ratio(wasted4, bought4, fill=0.0).clip(0, 1)
        group["sell_through_4w"] = _safe_ratio(sold4, bought4, fill=1.0).clip(0, 3)

        cost = group["unit_cost"].astype(float)
        group["price_rel"] = _safe_ratio(cost, cost.shift(1).rolling(13, min_periods=1).mean())
        group["promo_rate_4w"] = group["is_promo"].shift(1).rolling(4, min_periods=1).mean()

        group["weeks_of_history"] = np.arange(len(group)) + 1

        if with_target:
            group[TARGET] = sum(sold.shift(-h) for h in range(1, HORIZON_WEEKS + 1))

        rows.append(group)

    table = pd.concat(rows, ignore_index=True)

    iso = table["week"].dt.isocalendar()
    table["week_of_year"] = iso["week"].astype(int)
    table["week_sin"] = np.sin(2 * np.pi * table["week_of_year"] / 52.0)
    table["week_cos"] = np.cos(2 * np.pi * table["week_of_year"] / 52.0)
    table["month"] = table["week"].dt.month

    years = sorted({int(y) for y in table["week"].dt.year.unique()})
    holidays = holiday_weeks(years + [max(years) + 1])
    ahead = pd.Series(0, index=table.index, dtype=int)
    for step in range(1, HORIZON_WEEKS + 1):
        future = table["week"] + pd.Timedelta(days=7 * step)
        ahead += future.isin(holidays).astype(int)
    table["holiday_weeks_ahead"] = ahead

    cat = catalog.copy() if not catalog.empty else pd.DataFrame(columns=["sku"])
    keep = [c for c in ("sku", "category", "shelf_life_days", "unit_cost") if c in cat.columns]
    cat = cat[keep].rename(columns={"unit_cost": "unit_cost_cat"})
    table = table.merge(cat, on="sku", how="left")

    if "category" not in table.columns:
        table["category"] = "Unknown"
    table["category"] = table["category"].fillna("Unknown").astype(str)
    table["shelf_life_days"] = pd.to_numeric(
        table.get("shelf_life_days"), errors="coerce").fillna(7.0)
    table["unit_cost_cat"] = pd.to_numeric(
        table.get("unit_cost_cat"), errors="coerce").fillna(table["unit_cost"])
    table["unit_cost_cat"] = table["unit_cost_cat"].fillna(0.0)

    for col in CATEGORICAL:
        table[col] = table[col].astype(str).astype("category")

    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce")

    return table


def training_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for fitting: a known target and at least one real lag."""
    if table.empty or TARGET not in table.columns:
        return table.iloc[0:0]
    usable = table[TARGET].notna() & table["lag_1w"].notna()
    return table[usable].copy()


def latest_rows(table: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """The one row per item that the next two weeks are predicted from."""
    if table.empty:
        return table
    working = table if as_of is None else table[table["week"] <= pd.Timestamp(as_of)]
    if working.empty:
        return table.iloc[0:0]
    return (working.sort_values("week")
            .groupby("sku", as_index=False, observed=True)
            .tail(1)
            .reset_index(drop=True))
