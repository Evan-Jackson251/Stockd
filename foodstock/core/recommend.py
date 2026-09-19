"""Turning a demand forecast into an order sheet.

The forecast says how much will sell over two weeks. It does not say how much to
buy, because four other things get in the way:

  stock already on hand   some of which will spoil before it can be sold
  shelf life              a two week forecast does not justify a two week order
                          of something that dies in three days
  pack size and minimums  suppliers sell cases, not units
  budget                  when the money runs out, something has to give

Shelf life is handled by splitting the answer in two. The two-week total is what
the owner is budgeting for and is what the price estimate covers. The delivery
quantity is what actually goes on the next order form, capped at what can be sold
before it spoils, with a reorder interval attached. Croissants get 140 units
across seven drops of 20, not one drop of 140 that half ends up binned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from foodstock.core.features import HORIZON_DAYS
from foodstock.core.model import Forecast

# What the service level slider means in plain terms.
SERVICE_LEVEL_LABELS = [
    (0.20, "Lean", "Order tight. Least waste, expect some items to run out."),
    (0.40, "Careful", "A little under the middle estimate. Waste stays low."),
    (0.60, "Balanced", "Order to the middle estimate. Roughly even odds either way."),
    (0.78, "Comfortable", "Keep a buffer. Fewer gaps, a little more waste."),
    (0.90, "Never run out", "Order high. Shelves stay full, waste goes up."),
]


def service_level_label(level: float) -> tuple[str, str]:
    """The slider's name and one-line consequence at a given level."""
    for threshold, name, blurb in SERVICE_LEVEL_LABELS:
        if level <= threshold:
            return name, blurb
    return SERVICE_LEVEL_LABELS[-1][1], SERVICE_LEVEL_LABELS[-1][2]


@dataclass
class OrderPlan:
    """A complete order recommendation, ready to show or export."""

    lines: pd.DataFrame
    horizon_days: int
    service_level: float
    budget: float | None
    as_of_week: pd.Timestamp
    notes: list[str] = field(default_factory=list)

    @property
    def ordered(self) -> pd.DataFrame:
        return self.lines[self.lines["horizon_qty"] > 0]

    @property
    def total_cost(self) -> float:
        """Estimated spend across the whole horizon."""
        return float(self.lines["horizon_cost"].sum())

    @property
    def first_delivery_cost(self) -> float:
        return float(self.lines["delivery_cost"].sum())

    @property
    def projected_waste_units(self) -> float:
        return float(self.lines["projected_waste_units"].sum())

    @property
    def projected_waste_cost(self) -> float:
        return float(self.lines["projected_waste_cost"].sum())

    @property
    def flagged(self) -> pd.DataFrame:
        return self.lines[self.lines["flags"].str.len() > 0]

    def summary(self) -> dict[str, object]:
        lines = self.lines
        waste_share = (100.0 * self.projected_waste_cost / self.total_cost
                       if self.total_cost > 0 else 0.0)
        return {
            "items_ordered": int((lines["horizon_qty"] > 0).sum()),
            "items_skipped": int((lines["horizon_qty"] == 0).sum()),
            "total_cost": self.total_cost,
            "first_delivery_cost": self.first_delivery_cost,
            "budget": self.budget,
            "budget_left": None if self.budget is None else self.budget - self.total_cost,
            "projected_waste_units": self.projected_waste_units,
            "projected_waste_cost": self.projected_waste_cost,
            "waste_from_order_cost": float(lines["waste_from_order_cost"].sum()),
            "waste_from_stock_units": float(lines["waste_from_stock_units"].sum()),
            "waste_share_pct": waste_share,
            "stockout_risks": int(lines["flags"].str.contains("Stockout risk").sum()),
            "order_today": int(lines["flags"].str.contains("Order today").sum()),
            "expiring_now": float(lines["expiring_before_use"].sum()),
            "deferred": int(lines["flags"].str.contains("Held back").sum()),
        }

    def order_sheet(self) -> pd.DataFrame:
        """The columns a supplier order actually needs."""
        out = self.ordered[[
            "sku", "item_name", "unit", "delivery_qty", "packs_per_delivery",
            "pack_size", "deliveries", "reorder_every_days",
            "horizon_qty", "unit_cost", "horizon_cost", "reason",
        ]].copy()
        return out.sort_values("horizon_cost", ascending=False).reset_index(drop=True)


def _as_float(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _round_to_packs(quantity, pack_size: np.ndarray,
                    min_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round up to whole packs and honour supplier minimums."""
    quantity = np.asarray(quantity, dtype=float)
    packs = np.ceil(np.maximum(quantity, 0) / pack_size)
    qty = packs * pack_size
    lift = (qty > 0) & (qty < min_order)
    qty = np.where(lift, np.ceil(min_order / pack_size) * pack_size, qty)
    packs = np.ceil(np.divide(qty, pack_size, out=np.zeros_like(qty), where=pack_size > 0))
    return qty, packs.astype(int)


def build_plan(forecast: Forecast, catalog: pd.DataFrame, snapshot: pd.DataFrame,
               *, service_level: float = 0.6, budget: float | None = None,
               horizon_days: int = HORIZON_DAYS) -> OrderPlan:
    """Produce the order plan for the next `horizon_days`."""
    notes: list[str] = list(forecast.notes)

    table = forecast.frame.copy()
    table["planned_units"] = forecast.at_service_level(service_level).to_numpy()
    table["upper_units"] = table["p90"]

    if catalog is None or catalog.empty:
        raise ValueError("The catalog is empty, so there is nothing to order.")
    table = table.merge(catalog, on="sku", how="left")

    missing = table["item_name"].isna()
    if missing.any():
        notes.append(f"{int(missing.sum())} forecast items are not in the catalog and were "
                     "skipped. Add them to the catalog to include them in orders.")
        table = table[~missing].reset_index(drop=True)
    if table.empty:
        raise ValueError("No forecast items matched the catalog.")

    if snapshot is not None and not snapshot.empty and "sku" in snapshot.columns:
        table = table.merge(snapshot, on="sku", how="left")
    else:
        table["on_hand"] = 0.0
        table["days_to_expiry"] = np.nan
        notes.append("No current stock was provided, so every item is treated as empty.")

    shelf_life = _as_float(table, "shelf_life_days", 7.0).clip(lower=1).to_numpy()
    pack_size = _as_float(table, "pack_size", 1.0).clip(lower=1).to_numpy()
    min_order = _as_float(table, "min_order_qty", 0.0).clip(lower=0).to_numpy()
    lead_time = _as_float(table, "lead_time_days", 2.0).clip(lower=0).to_numpy()
    unit_cost = _as_float(table, "unit_cost", 0.0).clip(lower=0).to_numpy()
    unit_price = _as_float(table, "unit_price", np.nan).to_numpy()
    on_hand = _as_float(table, "on_hand", 0.0).clip(lower=0).to_numpy()

    expiry = _as_float(table, "days_to_expiry", np.nan)
    expiry = expiry.fillna(pd.Series(shelf_life, index=table.index))
    days_to_expiry = expiry.clip(lower=0).to_numpy()

    planned = table["planned_units"].clip(lower=0).to_numpy()
    upper = table["upper_units"].clip(lower=0).to_numpy()
    daily_rate = planned / horizon_days

    # Stock that can be sold before it expires. The rest is already lost.
    usable_on_hand = np.minimum(on_hand, daily_rate * days_to_expiry)
    expiring = np.maximum(on_hand - usable_on_hand, 0.0)

    # Total to buy across the horizon, and what it costs. This is the budget line.
    horizon_need = np.maximum(planned - usable_on_hand, 0.0)
    horizon_qty, horizon_packs = _round_to_packs(horizon_need, pack_size, min_order)

    # How the horizon splits into deliveries. One drop must clear within its own
    # shelf life, so short-life items get more, smaller drops.
    cover_days = np.minimum(shelf_life, float(horizon_days))
    deliveries = np.maximum(np.ceil(horizon_days / np.maximum(cover_days, 1.0)), 1).astype(int)

    first_need = np.where(deliveries > 1,
                          np.maximum(daily_rate * cover_days - usable_on_hand, 0.0),
                          horizon_need)
    delivery_qty, _ = _round_to_packs(first_need, pack_size, min_order)
    delivery_qty = np.minimum(delivery_qty, np.maximum(horizon_qty, 0.0))
    delivery_packs = np.ceil(np.divide(delivery_qty, pack_size,
                                       out=np.zeros_like(delivery_qty),
                                       where=pack_size > 0)).astype(int)

    days_of_stock = np.divide(usable_on_hand, daily_rate,
                              out=np.full_like(usable_on_hand, np.inf),
                              where=daily_rate > 0)

    table = table.assign(
        planned_units=np.round(planned, 1),
        daily_rate=np.round(daily_rate, 2),
        usable_on_hand=np.round(usable_on_hand, 1),
        expiring_before_use=np.round(expiring, 1),
        horizon_qty=np.round(horizon_qty, 1),
        horizon_packs=horizon_packs,
        delivery_qty=np.round(delivery_qty, 1),
        packs_per_delivery=delivery_packs,
        deliveries=deliveries,
        reorder_every_days=np.round(cover_days).astype(int),
        pack_size=pack_size.astype(int),
        shelf_life_days=shelf_life.astype(int),
        lead_time_days=lead_time.astype(int),
        min_order_qty=min_order,
        unit_cost=np.round(unit_cost, 3),
        on_hand=np.round(on_hand, 1),
        days_to_expiry=days_to_expiry.astype(int),
        days_of_stock_left=np.round(np.nan_to_num(days_of_stock, posinf=999), 1),
        upper_units=np.round(upper, 1),
    )

    table["horizon_cost"] = (table["horizon_qty"] * table["unit_cost"]).round(2)
    table["delivery_cost"] = (table["delivery_qty"] * table["unit_cost"]).round(2)

    # Value at risk decides who keeps their order when the budget bites. Items
    # with no sale price recorded still matter, so cost stands in for value.
    margin = np.where(np.isfinite(unit_price) & (unit_price > unit_cost),
                      unit_price - unit_cost, unit_cost * 0.45)
    table["value_at_risk"] = np.round(horizon_need * margin, 2)

    table, budget_notes = _apply_budget(table, budget)
    notes.extend(budget_notes)

    table = _finalise_lines(table, horizon_days)
    return OrderPlan(table, horizon_days, service_level, budget, forecast.as_of_week, notes)


def _apply_budget(table: pd.DataFrame, budget: float | None) -> tuple[pd.DataFrame, list[str]]:
    """Trim the order to fit a budget, protecting the highest value lines.

    Greedy by value at risk rather than proportional trimming: cutting every line
    by a fifth guarantees running short on everything, whereas dropping the lines
    that matter least keeps the busy items fully stocked.
    """
    table = table.copy()
    table["budget_trimmed"] = False
    table["budget_deferred"] = False

    if budget is None or budget <= 0:
        return table, []

    total = float(table["horizon_cost"].sum())
    if total <= budget:
        return table, []

    notes = [f"The full order came to {total:,.0f} against a budget of {budget:,.0f}, "
             "so the lowest value lines were cut first."]
    remaining = float(budget)

    for idx in table.sort_values("value_at_risk", ascending=False).index:
        cost = float(table.at[idx, "horizon_cost"])
        if cost <= remaining + 1e-9:
            remaining -= cost
            continue

        pack_size = float(table.at[idx, "pack_size"])
        unit_cost = float(table.at[idx, "unit_cost"])
        pack_cost = pack_size * unit_cost
        affordable = 0 if pack_cost <= 0 else math.floor(remaining / pack_cost)

        if affordable >= 1:
            qty = affordable * pack_size
            table.at[idx, "horizon_qty"] = round(qty, 1)
            table.at[idx, "horizon_packs"] = int(affordable)
            table.at[idx, "horizon_cost"] = round(qty * unit_cost, 2)
            table.at[idx, "budget_trimmed"] = True
            remaining -= qty * unit_cost
        else:
            table.at[idx, "horizon_qty"] = 0.0
            table.at[idx, "horizon_packs"] = 0
            table.at[idx, "horizon_cost"] = 0.0
            table.at[idx, "budget_deferred"] = True

    # Deliveries can never exceed what was actually bought.
    over = table["delivery_qty"] > table["horizon_qty"]
    table.loc[over, "delivery_qty"] = table.loc[over, "horizon_qty"]
    table.loc[over, "packs_per_delivery"] = table.loc[over, "horizon_packs"]
    table["delivery_cost"] = (table["delivery_qty"] * table["unit_cost"]).round(2)

    deferred = int(table["budget_deferred"].sum())
    if deferred:
        notes.append(f"{deferred} items were held back entirely. Raise the budget or "
                     "lower the service level to bring them back.")
    return table, notes


def _finalise_lines(table: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Add projected waste, flags and a plain-English reason per line."""
    table = table.copy()

    # Two different kinds of waste, kept apart because they answer to different
    # controls. Stock waste is already committed and only the demand assumption
    # moves it. Order waste is caused by pack sizes forcing more into a single
    # drop than its shelf life can clear, which is the part ordering can fix.
    capacity = table["daily_rate"] * table["shelf_life_days"]
    over_per_drop = (table["delivery_qty"] - capacity).clip(lower=0)
    order_waste = np.minimum(over_per_drop * table["deliveries"], table["horizon_qty"])

    table["waste_from_stock_units"] = table["expiring_before_use"].round(1)
    table["waste_from_order_units"] = order_waste.round(1)
    table["projected_waste_units"] = (order_waste + table["expiring_before_use"]).round(1)
    table["projected_waste_cost"] = (table["projected_waste_units"]
                                     * table["unit_cost"]).round(2)
    table["waste_from_order_cost"] = (table["waste_from_order_units"]
                                      * table["unit_cost"]).round(2)

    covered = table["horizon_qty"] + table["usable_on_hand"]
    table["covered_units"] = covered.round(1)
    # How exposed the item is to a busier than expected fortnight.
    table["busy_week_gap"] = (table["upper_units"] - covered).clip(lower=0).round(1)
    exposure = np.divide(table["busy_week_gap"], table["planned_units"].replace(0, np.nan))
    table["exposure_share"] = exposure.fillna(0.0).round(3)

    flags: list[str] = []
    reasons: list[str] = []
    for row in table.itertuples():
        marks = []
        if row.budget_deferred:
            marks.append("Held back for budget")
        elif row.budget_trimmed:
            marks.append("Trimmed for budget")

        if row.horizon_qty > 0 and row.covered_units < row.planned_units - 0.5:
            marks.append("Short of forecast")
        elif row.horizon_qty > 0 and row.exposure_share > 0.25:
            # Only worth saying when a busy fortnight would leave a real hole.
            marks.append("Stockout risk if busy")

        if row.days_of_stock_left < row.lead_time_days and row.horizon_qty > 0:
            marks.append("Order today")
        if row.expiring_before_use > 0.5:
            marks.append("Stock expiring")
        if row.deliveries > 1:
            marks.append(f"{row.deliveries} drops")
        if row.waste_from_order_units > 0.5:
            marks.append("Pack size forces waste")
        flags.append(" · ".join(marks))

        parts = [f"expect {row.planned_units:.0f} to sell in {horizon_days} days"]
        if row.usable_on_hand > 0.5:
            parts.append(f"{row.usable_on_hand:.0f} usable on hand")
        if row.expiring_before_use > 0.5:
            parts.append(f"{row.expiring_before_use:.0f} expiring first")
        if row.deliveries > 1:
            parts.append(f"{row.shelf_life_days} day shelf life, so {row.deliveries} drops of "
                         f"{row.delivery_qty:.0f} every {row.reorder_every_days} days")
        elif row.horizon_packs and row.pack_size > 1:
            parts.append(f"rounded to {row.horizon_packs} pack"
                         f"{'s' if row.horizon_packs != 1 else ''} of {row.pack_size}")
        reasons.append(", ".join(parts))

    table["flags"] = flags
    table["reason"] = reasons

    columns = [
        "sku", "item_name", "category", "unit",
        "horizon_qty", "horizon_packs", "horizon_cost",
        "delivery_qty", "packs_per_delivery", "deliveries", "reorder_every_days",
        "delivery_cost", "pack_size", "unit_cost",
        "planned_units", "p10", "p50", "p90", "upper_units", "covered_units",
        "recent_2w", "last_year_2w",
        "on_hand", "usable_on_hand", "days_to_expiry", "expiring_before_use",
        "days_of_stock_left", "shelf_life_days", "lead_time_days",
        "projected_waste_units", "projected_waste_cost",
        "waste_from_stock_units", "waste_from_order_units", "waste_from_order_cost",
        "busy_week_gap", "exposure_share",
        "value_at_risk", "trend_ratio", "holiday_weeks_ahead",
        "flags", "reason",
    ]
    present = [c for c in columns if c in table.columns]
    return (table[present]
            .sort_values(["horizon_cost", "value_at_risk"], ascending=False)
            .reset_index(drop=True))
