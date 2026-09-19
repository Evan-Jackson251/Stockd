"""Generates two years of plausible stock history for a small cafe.

Real historical data for one specific business will not exist at a hackathon, and
a flat random series makes a forecasting demo look pointless. This simulates the
things that actually make ordering hard:

  weekday rhythm      Saturday brunch outsells Tuesday
  annual seasonality  soup in January, salad in July, each item with its own phase
  holidays            Thanksgiving week spikes, the week after collapses
  promotions          occasional discounts that pull demand forward
  slow growth         the shop is getting busier
  spoilage            stock is tracked in batches, so short shelf life really wastes

Ordering is simulated with a deliberately naive weekly rule, which is what the
model then has to beat.
"""

from __future__ import annotations

from collections import deque
from datetime import date, timedelta

import numpy as np
import pandas as pd

from foodstock.core.features import holiday_dates

ITEMS = [
    # sku, name, category, unit, cost, price, shelf life, pack, base/day, season amp, phase
    ("TOM-RMA", "Roma tomatoes",      "Produce",   "kg",   2.40,  5.50,  6,  5, 14, 0.30, 0.55),
    ("LET-ROM", "Romaine lettuce",    "Produce",   "each", 1.15,  3.20,  5, 12, 22, 0.42, 0.50),
    ("AVO-HAS", "Hass avocados",      "Produce",   "each", 0.95,  2.75,  7, 24, 30, 0.22, 0.45),
    ("BER-STR", "Strawberries",       "Produce",   "tray", 3.10,  7.00,  4,  8, 12, 0.60, 0.42),
    ("MLK-WHL", "Whole milk",         "Dairy",     "gal",  3.60,  0.00,  12, 4, 26, 0.12, 0.10),
    ("CHE-CHD", "Cheddar block",      "Dairy",     "kg",   7.80, 16.00,  30, 2,  7, 0.15, 0.88),
    ("BUT-UNS", "Unsalted butter",    "Dairy",     "kg",   6.20,  0.00,  45, 6,  5, 0.20, 0.92),
    ("BRD-SDG", "Sourdough loaves",   "Bakery",    "each", 1.80,  5.00,  3, 10, 34, 0.18, 0.30),
    ("CRO-PLN", "Plain croissants",   "Bakery",    "each", 0.70,  3.25,  2, 24, 46, 0.20, 0.35),
    ("CHK-BRS", "Chicken breast",     "Meat",      "kg",   6.40, 14.00,  4,  5, 18, 0.14, 0.60),
    ("BAC-STR", "Streaky bacon",      "Meat",      "kg",   7.10, 15.50,  9,  5, 11, 0.16, 0.05),
    ("COF-BLD", "House coffee beans", "Beverage",  "kg",  14.50, 38.00, 180, 4,  9, 0.10, 0.02),
    ("OAT-MLK", "Oat milk",           "Beverage",  "case", 18.00,  0.00,  90, 1,  6, 0.14, 0.20),
    ("FLR-APP", "All purpose flour",  "Dry goods", "kg",   1.05,  0.00, 300, 10, 8, 0.08, 0.95),
    ("SUG-GRN", "Granulated sugar",   "Dry goods", "kg",   0.95,  0.00, 365, 10, 6, 0.09, 0.90),
    ("PMP-PUR", "Pumpkin puree",      "Dry goods", "case", 22.00,  0.00, 400, 1,  2, 0.95, 0.80),
]

WEEKDAY_FACTOR = np.array([0.82, 0.78, 0.86, 0.98, 1.22, 1.48, 1.26])  # Mon..Sun


def build_catalog() -> pd.DataFrame:
    rows = []
    for sku, name, cat, unit, cost, price, life, pack, *_ in ITEMS:
        rows.append({
            "sku": sku, "item_name": name, "category": cat, "unit": unit,
            "unit_cost": cost,
            "unit_price": price if price > 0 else np.nan,
            "shelf_life_days": life, "pack_size": pack,
            "min_order_qty": pack, "lead_time_days": 2 if cat in ("Produce", "Bakery") else 4,
        })
    return pd.DataFrame(rows)


def _demand_curve(item, days: list[date], holidays: set[date], rng: np.random.Generator):
    _, _, _, _, _, _, _, _, base, amp, phase = item
    n = len(days)
    day_of_year = np.array([d.timetuple().tm_yday for d in days])
    weekday = np.array([d.weekday() for d in days])

    seasonal = 1.0 + amp * np.sin(2 * np.pi * (day_of_year / 365.25 - phase))
    weekly = WEEKDAY_FACTOR[weekday]
    growth = np.linspace(1.0, 1.18, n)

    holiday_boost = np.ones(n)
    for i, d in enumerate(days):
        if d in holidays:
            holiday_boost[i] = 1.9
        elif any(abs((d - h).days) <= 2 for h in holidays if abs((d - h).days) <= 2):
            holiday_boost[i] = 1.35

    # Promotions: a handful of week-long discounts per year.
    promo = np.zeros(n, dtype=bool)
    for start in rng.choice(np.arange(n - 7), size=max(1, n // 120), replace=False):
        promo[start:start + 7] = True
    promo_lift = np.where(promo, 1.45, 1.0)

    mean = base * seasonal * weekly * growth * holiday_boost * promo_lift
    demand = rng.poisson(np.clip(mean, 0.1, None)).astype(float)
    return demand, promo


def _simulate(item, days, demand, promo, rng) -> list[dict]:
    """Run FIFO batches so waste and closing stock are internally consistent."""
    sku, _, _, _, cost, _, shelf_life, pack, base, *_ = item
    batches: deque[list[float]] = deque()  # [units, days_left]
    rows = []

    on_order: dict[int, float] = {}
    lead = 2
    target_cover = shelf_life if shelf_life < 14 else 14

    for i, day in enumerate(days):
        received = on_order.pop(i, 0.0)
        if received:
            batches.append([received, float(shelf_life)])

        want = demand[i]
        sold = 0.0
        while want > 0 and batches:
            batch = batches[0]
            take = min(batch[0], want)
            batch[0] -= take
            sold += take
            want -= take
            if batch[0] <= 1e-9:
                batches.popleft()

        for batch in batches:
            batch[1] -= 1
        wasted = sum(b[0] for b in batches if b[1] <= 0)
        while batches and batches[0][1] <= 0:
            batches.popleft()
        batches = deque([b for b in batches if b[1] > 0])

        stock = sum(b[0] for b in batches)

        # Naive weekly reorder: cover recent average demand, with noisy judgement.
        if day.weekday() == 0:
            recent = demand[max(0, i - 14):i + 1]
            avg = recent.mean() if len(recent) else base
            need = avg * target_cover - stock
            need *= rng.normal(1.0, 0.16)
            qty = max(0.0, np.ceil(need / pack) * pack)
            if qty > 0:
                on_order[i + lead] = on_order.get(i + lead, 0.0) + qty

        day_cost = cost * float(rng.normal(1.0, 0.04)) * (1 + 0.05 * np.sin(i / 90))

        rows.append({
            "date": day, "sku": sku,
            "units_sold": round(sold, 2),
            "units_bought": round(received, 2),
            "units_wasted": round(wasted, 2),
            "closing_stock": round(stock, 2),
            "unit_cost": round(max(0.05, day_cost), 3),
            "is_promo": int(promo[i]),
        })

    return rows


def generate(days: int = 760, end: date | None = None, seed: int = 7
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return catalog, history and a current stock snapshot."""
    rng = np.random.default_rng(seed)
    end = end or date.today()
    start = end - timedelta(days=days - 1)
    dates = [start + timedelta(days=i) for i in range(days)]

    years = sorted({d.year for d in dates})
    holidays = {h for y in years for h in holiday_dates(y)}

    all_rows: list[dict] = []
    for item in ITEMS:
        demand, promo = _demand_curve(item, dates, holidays, rng)
        all_rows.extend(_simulate(item, dates, demand, promo, rng))

    history = pd.DataFrame(all_rows)
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values(["sku", "date"]).reset_index(drop=True)

    catalog = build_catalog()

    last = history.groupby("sku").tail(1).set_index("sku")
    snapshot = []
    for sku, _, _, _, _, _, life, *_ in ITEMS:
        on_hand = float(last.loc[sku, "closing_stock"])
        snapshot.append({
            "sku": sku,
            "on_hand": round(on_hand, 2),
            "days_to_expiry": int(max(1, round(life * rng.uniform(0.25, 0.8)))),
        })

    return catalog, history, pd.DataFrame(snapshot)


def write_demo_data(store, days: int = 760, seed: int = 7) -> dict[str, int]:
    """Populate a DataStore with demo data. Returns row counts."""
    catalog, history, snapshot = generate(days=days, seed=seed)
    store.replace("catalog", catalog)
    store.replace("history", history)
    store.replace("snapshot", snapshot)
    return {"catalog": len(catalog), "history": len(history), "snapshot": len(snapshot)}
