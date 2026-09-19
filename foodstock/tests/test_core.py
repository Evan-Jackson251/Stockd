"""Tests for the forecasting and ordering core.

Run from the project root:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from core import features as F
from core import model as M
from core import recommend as R
from core import schema, store, synth


# --------------------------------------------------------------------- schema

def test_aliases_and_currency_are_parsed():
    raw = pd.DataFrame({
        "Product Code": ["A1"], "Name": ["Apples"], "Dept": ["Produce"],
        "Cost per unit": ["$2.50"], "Shelf Life": ["7"],
    })
    report = schema.validate(raw, "catalog")
    assert report.ok
    assert report.frame.loc[0, "unit_cost"] == 2.50
    assert report.frame.loc[0, "shelf_life_days"] == 7
    # Optional columns fall back to their defaults.
    assert report.frame.loc[0, "pack_size"] == 1


def test_bad_required_cell_drops_only_that_row():
    raw = pd.DataFrame({
        "date": ["2026-01-01", "not-a-date", "2026-01-03"],
        "sku": ["A1", "A1", "A1"],
        "units_sold": ["5", "6", "7"],
    })
    report = schema.validate(raw, "history")
    assert report.rows_kept == 2
    assert len(report.errors) == 1
    assert report.errors[0].row == 3  # spreadsheet row, header is row 1


def test_missing_required_column_is_reported_once():
    raw = pd.DataFrame({"sku": ["A1"], "units_sold": ["5"]})
    report = schema.validate(raw, "history")
    assert not report.ok
    assert any(i.column == "date" for i in report.errors)


def test_duplicate_keys_keep_the_later_row():
    raw = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01"],
        "sku": ["A1", "A1"],
        "units_sold": ["5", "9"],
    })
    report = schema.validate(raw, "history")
    assert report.rows_kept == 1
    assert report.frame.loc[0, "units_sold"] == 9


def test_unknown_sku_warns_but_keeps_the_row():
    raw = pd.DataFrame({"date": ["2026-01-01"], "sku": ["GHOST"], "units_sold": ["5"]})
    report = schema.validate(raw, "history", known_skus={"A1"})
    assert report.ok
    assert any("catalog" in i.message for i in report.warnings)


def test_blank_optional_number_stays_missing():
    """A blank cell must not pick up the schema's boundary value.

    Comparing a nullable missing value against a bound yields <NA> rather than
    False, and Series.mask treats that as a hit, so an absent expiry date used
    to come out as the ten year maximum.
    """
    raw = pd.DataFrame({"sku": ["A1"], "on_hand": ["40"], "days_to_expiry": [""]})
    report = schema.validate(raw, "snapshot")
    assert pd.isna(report.frame.loc[0, "days_to_expiry"])


def test_out_of_range_numbers_are_still_corrected():
    raw = pd.DataFrame({"sku": ["A1", "A2"], "on_hand": ["5", "5"],
                        "days_to_expiry": ["99999", "-4"]})
    report = schema.validate(raw, "snapshot")
    frame = report.frame.set_index("sku")
    assert frame.loc["A1", "days_to_expiry"] == 3650      # clamped to the maximum
    assert pd.isna(frame.loc["A2", "days_to_expiry"])     # negative is dropped
    assert len(report.warnings) == 2


def test_blank_expiry_falls_back_to_shelf_life(trained):
    """The plan must treat unknown expiry as fresh stock, not as expired."""
    catalog, _, _, forecast = trained
    snapshot = pd.DataFrame({"sku": catalog["sku"], "on_hand": 5.0,
                             "days_to_expiry": np.nan})
    plan = R.build_plan(forecast, catalog, snapshot, service_level=0.6)
    merged = plan.lines.merge(catalog[["sku", "shelf_life_days"]], on="sku",
                              suffixes=("", "_cat"))
    assert (merged["days_to_expiry"] == merged["shelf_life_days_cat"]).all()


def test_table_detection():
    assert schema.detect_table(pd.DataFrame(columns=["date", "sku", "units_sold"])) == "history"
    assert schema.detect_table(pd.DataFrame(columns=["sku", "unit_cost", "shelf_life_days"])) == "catalog"
    assert schema.detect_table(pd.DataFrame(columns=["sku", "on_hand"])) == "snapshot"
    assert schema.detect_table(pd.DataFrame(columns=["colour", "size"])) is None


# ---------------------------------------------------------------------- store

def test_ingest_archives_and_merges(tmp_path):
    data = store.DataStore(tmp_path / "data")
    source = tmp_path / "week1.csv"
    pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "sku": ["A1", "A1"],
        "units_sold": [5, 6],
    }).to_csv(source, index=False)

    result = data.ingest(source)
    assert result.table == "history"
    assert result.added == 2
    assert result.archived_as is not None and result.archived_as.exists()
    assert len(data.load("history")) == 2


def test_reimporting_updates_rather_than_duplicates(tmp_path):
    data = store.DataStore(tmp_path / "data")
    first = tmp_path / "a.csv"
    pd.DataFrame({"date": ["2026-01-01"], "sku": ["A1"], "units_sold": [5]}).to_csv(first, index=False)
    data.ingest(first)

    corrected = tmp_path / "b.csv"
    pd.DataFrame({"date": ["2026-01-01"], "sku": ["A1"], "units_sold": [50]}).to_csv(corrected, index=False)
    result = data.ingest(corrected)

    stored = data.load("history")
    assert result.replaced == 1 and result.added == 0
    assert len(stored) == 1
    assert stored.loc[0, "units_sold"] == 50


def test_coverage_reports_span(tmp_path):
    data = store.DataStore(tmp_path / "data")
    catalog, history, snapshot = synth.generate(days=400, seed=1)
    data.replace("history", history)
    coverage = data.coverage()
    assert coverage.weeks >= 55 or not coverage.has_last_year
    assert coverage.skus == history["sku"].nunique()


# ------------------------------------------------------------------- features

def test_holiday_helpers():
    assert F._easter(2026) == date(2026, 4, 5)
    assert F._nth_weekday(2026, 11, 3, 4) == date(2026, 11, 26)   # Thanksgiving
    assert F._nth_weekday(2026, 5, 0, -1) == date(2026, 5, 25)    # Memorial Day


def test_target_is_the_next_two_weeks_only():
    days = [date(2026, 1, 5) + timedelta(days=i) for i in range(70)]
    history = pd.DataFrame({
        "date": days, "sku": "A1",
        "units_sold": 1.0, "units_bought": 0.0, "units_wasted": 0.0,
        "closing_stock": np.nan, "unit_cost": 1.0, "is_promo": False,
    })
    catalog = pd.DataFrame([{"sku": "A1", "item_name": "A", "category": "X",
                             "unit_cost": 1.0, "shelf_life_days": 5, "pack_size": 1}])
    table = F.build_features(history, catalog)
    complete = table[table[F.TARGET].notna()]
    # Seven sales a week, two weeks ahead.
    assert np.allclose(complete[F.TARGET].to_numpy(), 14.0)


def test_missing_weeks_are_filled_so_lags_stay_aligned():
    weeks = [date(2026, 1, 5), date(2026, 1, 12), date(2026, 2, 2)]
    history = pd.DataFrame({
        "date": weeks, "sku": "A1", "units_sold": [10.0, 10.0, 10.0],
        "units_bought": 0.0, "units_wasted": 0.0, "closing_stock": np.nan,
        "unit_cost": 1.0, "is_promo": False,
    })
    weekly = F._fill_week_gaps(F.to_weekly(history))
    assert len(weekly) == 5                      # two empty weeks inserted
    assert weekly["units_sold"].tolist() == [10.0, 10.0, 0.0, 0.0, 10.0]


def test_features_never_look_into_the_future():
    _, history, _ = synth.generate(days=200, seed=3)
    catalog = synth.build_catalog()
    table = F.build_features(history, catalog)
    one = table[table["sku"] == "TOM-RMA"].sort_values("week").reset_index(drop=True)
    weekly = F.to_weekly(history[history["sku"] == "TOM-RMA"]).sort_values("week").reset_index(drop=True)
    # lag_1w on row i must equal actual sales in week i-1.
    assert np.isclose(one.loc[5, "lag_1w"], weekly.loc[4, "units_sold"])


# ---------------------------------------------------------------------- model

def test_short_history_falls_back_and_says_so():
    days = [date(2026, 1, 5) + timedelta(days=i) for i in range(40)]
    history = pd.DataFrame({
        "date": days, "sku": "A1", "units_sold": 4.0, "units_bought": 0.0,
        "units_wasted": 0.0, "closing_stock": np.nan, "unit_cost": 1.0, "is_promo": False,
    })
    catalog = pd.DataFrame([{"sku": "A1", "item_name": "A", "category": "X",
                             "unit_cost": 1.0, "shelf_life_days": 5, "pack_size": 1}])
    forecast = M.fit_and_forecast(history, catalog)
    assert forecast.method == "moving_average"
    assert forecast.notes and "average" in forecast.notes[0]
    assert forecast.frame["p50"].iloc[0] > 0


@pytest.fixture(scope="module")
def trained():
    catalog, history, snapshot = synth.generate(days=760, seed=7)
    forecast = M.fit_and_forecast(history, catalog)
    return catalog, history, snapshot, forecast


def test_learned_model_is_used_and_beats_naive(trained):
    _, _, _, forecast = trained
    assert forecast.method == "gradient_boosting"
    assert forecast.scorecard.rows_tested > 0
    # Should beat repeating the last two weeks on held-back data.
    assert forecast.scorecard.model_wape < forecast.scorecard.baselines["repeat_last_2w"]


def test_quantiles_are_ordered(trained):
    _, _, _, forecast = trained
    frame = forecast.frame
    assert (frame["p10"] <= frame["p50"] + 1e-9).all()
    assert (frame["p50"] <= frame["p90"] + 1e-9).all()


def test_service_level_moves_planned_units_monotonically(trained):
    _, _, _, forecast = trained
    low = forecast.at_service_level(0.2)
    mid = forecast.at_service_level(0.5)
    high = forecast.at_service_level(0.9)
    assert (low <= mid + 1e-6).all()
    assert (mid <= high + 1e-6).all()


def test_last_year_signal_is_actually_used(trained):
    _, _, _, forecast = trained
    top = forecast.importances.head(12).index
    assert any(name.startswith("ly_") for name in top)


# ------------------------------------------------------------------ recommend

def test_short_shelf_life_is_split_into_drops(trained):
    catalog, _, snapshot, forecast = trained
    plan = R.build_plan(forecast, catalog, snapshot, service_level=0.6)
    croissant = plan.lines.set_index("sku").loc["CRO-PLN"]
    flour = plan.lines.set_index("sku").loc["FLR-APP"]
    assert croissant["deliveries"] > 1          # two day shelf life
    assert croissant["delivery_qty"] < croissant["horizon_qty"]
    assert flour["deliveries"] == 1             # keeps for months


def test_no_single_drop_exceeds_the_horizon_order(trained):
    catalog, _, snapshot, forecast = trained
    plan = R.build_plan(forecast, catalog, snapshot, service_level=0.6)
    assert (plan.lines["delivery_qty"] <= plan.lines["horizon_qty"] + 1e-9).all()


def test_orders_respect_pack_size_and_minimums(trained):
    catalog, _, snapshot, forecast = trained
    plan = R.build_plan(forecast, catalog, snapshot, service_level=0.6)
    for row in plan.ordered.itertuples():
        assert row.horizon_qty % row.pack_size == 0
        assert row.horizon_qty >= row.pack_size


def test_expiring_stock_is_not_counted_as_available():
    catalog = pd.DataFrame([{"sku": "A1", "item_name": "A", "category": "X", "unit": "each",
                             "unit_cost": 1.0, "unit_price": 3.0, "shelf_life_days": 10,
                             "pack_size": 1, "min_order_qty": 0, "lead_time_days": 1}])
    frame = pd.DataFrame([{"sku": "A1", "week": pd.Timestamp("2026-01-05"),
                           "p10": 90, "p30": 95, "p50": 100, "p70": 105, "p90": 110,
                           "expected": 100, "recent_2w": 100, "last_year_2w": 100,
                           "trend_ratio": 1.0, "holiday_weeks_ahead": 0}])
    forecast = M.Forecast(frame, pd.Timestamp("2026-01-05"), "test",
                          M.Scorecard(0, 0, float("nan")))
    # 100 units on hand but only one day before they spoil, selling ~7/day.
    snapshot = pd.DataFrame([{"sku": "A1", "on_hand": 100, "days_to_expiry": 1}])
    plan = R.build_plan(forecast, catalog, snapshot, service_level=0.5)
    line = plan.lines.iloc[0]
    assert line["usable_on_hand"] < 10
    assert line["expiring_before_use"] > 85
    assert line["horizon_qty"] > 85          # must still buy for the fortnight


def test_budget_caps_spend_and_protects_valuable_lines(trained):
    catalog, _, snapshot, forecast = trained
    full = R.build_plan(forecast, catalog, snapshot, service_level=0.6)
    budget = full.total_cost * 0.4
    tight = R.build_plan(forecast, catalog, snapshot, service_level=0.6, budget=budget)

    assert tight.total_cost <= budget + 1e-6
    assert tight.notes

    # The most valuable line is filled first, so it should survive as long as a
    # single pack of it is affordable at all.
    top = full.lines.sort_values("value_at_risk", ascending=False).iloc[0]
    if top["pack_size"] * top["unit_cost"] <= budget:
        assert tight.lines.set_index("sku").loc[top["sku"], "horizon_qty"] > 0

    # Nothing is left on the table: every dropped line costs more than the
    # leftover budget, otherwise the greedy fill would have taken it.
    leftover = budget - tight.total_cost
    dropped = tight.lines[tight.lines["horizon_qty"] == 0]
    for row in dropped.itertuples():
        assert row.pack_size * row.unit_cost > leftover - 1e-6


def test_higher_service_level_costs_more(trained):
    catalog, _, snapshot, forecast = trained
    lean = R.build_plan(forecast, catalog, snapshot, service_level=0.2)
    generous = R.build_plan(forecast, catalog, snapshot, service_level=0.9)
    assert generous.total_cost > lean.total_cost


def test_empty_catalog_is_rejected(trained):
    _, _, snapshot, forecast = trained
    with pytest.raises(ValueError):
        R.build_plan(forecast, pd.DataFrame(), snapshot)


def test_missing_snapshot_treats_stock_as_empty(trained):
    catalog, _, _, forecast = trained
    plan = R.build_plan(forecast, catalog, pd.DataFrame(), service_level=0.6)
    assert (plan.lines["on_hand"] == 0).all()
    assert any("no current stock" in n.lower() for n in plan.notes)


def test_order_sheet_has_supplier_columns(trained):
    catalog, _, snapshot, forecast = trained
    plan = R.build_plan(forecast, catalog, snapshot, service_level=0.6)
    sheet = plan.order_sheet()
    for column in ("sku", "item_name", "delivery_qty", "horizon_cost", "reason"):
        assert column in sheet.columns
    assert len(sheet) == plan.summary()["items_ordered"]
