"""Canonical data schemas and validation.

Three tables drive the whole app:

  catalog   one row per item you stock       (static facts: shelf life, cost, pack size)
  history   one row per (date, sku)          (what you bought, sold, threw away)
  snapshot  one row per sku                  (what is on the shelf right now)

Keeping the static item facts out of the history table means the person entering
data types shelf life and pack size once instead of on every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Column:
    """One column in a table schema."""

    name: str
    kind: Literal["str", "float", "int", "date", "bool"]
    required: bool = True
    default: object = None
    min_value: float | None = None
    max_value: float | None = None
    help: str = ""


@dataclass
class Issue:
    """A single validation problem, addressed to the person fixing the file."""

    row: int | None
    column: str
    message: str
    severity: Severity = "error"

    def describe(self) -> str:
        where = f"row {self.row}" if self.row is not None else "file"
        return f"{where} · {self.column}: {self.message}"


@dataclass
class ValidationReport:
    """Outcome of validating one file."""

    table: str
    frame: pd.DataFrame
    issues: list[Issue] = field(default_factory=list)
    rows_in: int = 0
    rows_kept: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors and self.rows_kept > 0

    def summary(self) -> str:
        if self.rows_in == 0:
            return "The file has no data rows."
        parts = [f"{self.rows_kept} of {self.rows_in} rows usable"]
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        return " · ".join(parts)


CATALOG_SCHEMA: tuple[Column, ...] = (
    Column("sku", "str", help="Short code you use for the item, e.g. TOM-RMA"),
    Column("item_name", "str", help="Name a staff member would recognise"),
    Column("category", "str", help="Produce, Dairy, Bakery, Meat, Dry goods, Beverage"),
    Column("unit", "str", required=False, default="each", help="each, kg, case, tray"),
    Column("unit_cost", "float", min_value=0, help="What you pay per unit"),
    Column("unit_price", "float", required=False, default=np.nan, min_value=0,
           help="What you sell it for, used to rank items when budget is tight"),
    Column("shelf_life_days", "int", min_value=1, max_value=3650,
           help="Days from delivery until it must be thrown out"),
    Column("pack_size", "int", required=False, default=1, min_value=1,
           help="Units per case. Orders round up to whole packs"),
    Column("min_order_qty", "int", required=False, default=0, min_value=0,
           help="Smallest order the supplier accepts"),
    Column("lead_time_days", "int", required=False, default=2, min_value=0, max_value=60,
           help="Days between placing the order and delivery"),
)

HISTORY_SCHEMA: tuple[Column, ...] = (
    Column("date", "date", help="Day the numbers describe (YYYY-MM-DD)"),
    Column("sku", "str", help="Must match a sku in the catalog"),
    Column("units_sold", "float", min_value=0, help="Units sold that day"),
    Column("units_bought", "float", required=False, default=0.0, min_value=0,
           help="Units delivered that day"),
    Column("units_wasted", "float", required=False, default=0.0, min_value=0,
           help="Units thrown out that day"),
    Column("closing_stock", "float", required=False, default=np.nan, min_value=0,
           help="Units left at close of business"),
    Column("unit_cost", "float", required=False, default=np.nan, min_value=0,
           help="What you paid per unit that day, if it moves around"),
    Column("is_promo", "bool", required=False, default=False,
           help="1 if the item was discounted or featured that day"),
)

SNAPSHOT_SCHEMA: tuple[Column, ...] = (
    Column("sku", "str", help="Must match a sku in the catalog"),
    Column("on_hand", "float", min_value=0, help="Units currently in stock"),
    Column("days_to_expiry", "int", required=False, default=None, min_value=0, max_value=3650,
           help="Days until the stock on hand spoils. Blank uses the catalog shelf life"),
)

SCHEMAS: dict[str, tuple[Column, ...]] = {
    "catalog": CATALOG_SCHEMA,
    "history": HISTORY_SCHEMA,
    "snapshot": SNAPSHOT_SCHEMA,
}

KEYS: dict[str, list[str]] = {
    "catalog": ["sku"],
    "history": ["date", "sku"],
    "snapshot": ["sku"],
}

# Header spellings people actually use, mapped onto canonical names.
ALIASES: dict[str, str] = {
    "item": "sku", "item_code": "sku", "code": "sku", "product": "sku",
    "product_code": "sku", "id": "sku",
    "name": "item_name", "description": "item_name", "item_description": "item_name",
    "dept": "category", "department": "category", "group": "category",
    "cost": "unit_cost", "cost_per_unit": "unit_cost", "purchase_price": "unit_cost",
    "price": "unit_price", "sale_price": "unit_price", "retail_price": "unit_price",
    "shelf_life": "shelf_life_days", "shelflife": "shelf_life_days",
    "expiry_days": "shelf_life_days", "life_days": "shelf_life_days",
    "case_size": "pack_size", "pack": "pack_size", "units_per_case": "pack_size",
    "moq": "min_order_qty", "minimum_order": "min_order_qty",
    "lead_time": "lead_time_days",
    "day": "date", "sale_date": "date", "txn_date": "date", "business_date": "date",
    "sold": "units_sold", "qty_sold": "units_sold", "sales": "units_sold",
    "sales_units": "units_sold", "quantity_sold": "units_sold",
    "bought": "units_bought", "received": "units_bought", "qty_received": "units_bought",
    "purchased": "units_bought", "deliveries": "units_bought",
    "waste": "units_wasted", "wasted": "units_wasted", "spoilage": "units_wasted",
    "thrown_out": "units_wasted", "shrink": "units_wasted",
    "stock": "closing_stock", "stock_on_hand": "closing_stock",
    "ending_stock": "closing_stock", "closing": "closing_stock",
    "promo": "is_promo", "promotion": "is_promo", "on_sale": "is_promo",
    "discounted": "is_promo",
    "qty": "on_hand", "quantity": "on_hand", "current_stock": "on_hand",
    "in_stock": "on_hand", "on_hand_units": "on_hand",
    "expires_in": "days_to_expiry", "days_left": "days_to_expiry",
    "days_until_expiry": "days_to_expiry",
}

_TRUEISH = {"1", "y", "yes", "true", "t", "promo", "on sale", "sale"}
_FALSEISH = {"", "0", "n", "no", "false", "f", "nan", "none", "-"}


def normalise_header(raw: str) -> str:
    """Lowercase, strip, collapse separators, then resolve aliases."""
    cleaned = str(raw).strip().lower()
    for ch in (" ", "-", ".", "/"):
        cleaned = cleaned.replace(ch, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return ALIASES.get(cleaned, cleaned)


def blank_frame(table: str) -> pd.DataFrame:
    """An empty frame with the right columns and dtypes."""
    cols = {}
    for col in SCHEMAS[table]:
        if col.kind == "date":
            cols[col.name] = pd.Series(dtype="datetime64[ns]")
        elif col.kind in ("float", "int"):
            cols[col.name] = pd.Series(dtype="float64")
        elif col.kind == "bool":
            cols[col.name] = pd.Series(dtype="bool")
        else:
            cols[col.name] = pd.Series(dtype="object")
    return pd.DataFrame(cols)


def detect_table(frame: pd.DataFrame) -> str | None:
    """Guess which table a frame holds from its headers."""
    headers = {normalise_header(c) for c in frame.columns}
    if {"date", "sku"} <= headers and ("units_sold" in headers or "units_bought" in headers):
        return "history"
    if "sku" in headers and {"shelf_life_days", "unit_cost"} & headers:
        return "catalog"
    if "sku" in headers and "on_hand" in headers:
        return "snapshot"
    if {"date", "sku"} <= headers:
        return "history"
    return None


def _coerce_bool(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = series.astype("string").str.strip().str.lower().fillna("")
    out = pd.Series(False, index=series.index, dtype="bool")
    out[text.isin(_TRUEISH)] = True
    bad = ~(text.isin(_TRUEISH) | text.isin(_FALSEISH))
    return out, bad


def validate(frame: pd.DataFrame, table: str, *, known_skus: set[str] | None = None) -> ValidationReport:
    """Clean and check a frame against a schema.

    Rows with a problem in a required column are dropped and reported. Rows with
    a problem in an optional column keep the row and fall back to the default,
    so one stray cell never costs somebody a whole week of records.
    """
    if table not in SCHEMAS:
        raise ValueError(f"Unknown table {table!r}")

    schema = SCHEMAS[table]
    issues: list[Issue] = []
    work = frame.copy()
    work.columns = [normalise_header(c) for c in work.columns]
    work = work.loc[:, ~work.columns.duplicated(keep="first")]
    rows_in = len(work)

    # Row numbers as they appear in a spreadsheet: header is row 1.
    source_row = pd.Series(np.arange(rows_in) + 2, index=work.index)

    missing_required = [c.name for c in schema if c.required and c.name not in work.columns]
    for name in missing_required:
        issues.append(Issue(None, name, "Required column is missing from the file"))
    if missing_required:
        return ValidationReport(table, blank_frame(table), issues, rows_in, 0)

    drop = pd.Series(False, index=work.index)

    for col in schema:
        if col.name not in work.columns:
            work[col.name] = col.default
            continue

        raw = work[col.name]

        if col.kind == "date":
            parsed = pd.to_datetime(raw, errors="coerce")
            bad = parsed.isna()
        elif col.kind in ("float", "int"):
            text = raw.astype("string").str.replace(r"[,$£€\s]", "", regex=True)
            parsed = pd.to_numeric(text, errors="coerce")
            bad = parsed.isna() & raw.notna() & (raw.astype("string").str.strip() != "")
            blank = raw.isna() | (raw.astype("string").str.strip() == "")
            # A comparison against <NA> yields <NA>, not False, and Series.mask
            # treats that as a hit. Without fillna a blank optional cell gets the
            # boundary value written into it, so a missing expiry date silently
            # became ten years. Both masks must be plain booleans.
            if col.min_value is not None:
                below = (parsed < col.min_value).fillna(False).astype(bool)
                for r in source_row[below]:
                    issues.append(Issue(int(r), col.name,
                                        f"Value is below the minimum of {col.min_value:g}",
                                        "error" if col.required else "warning"))
                parsed = parsed.mask(below, np.nan)
                bad = bad | below
            if col.max_value is not None:
                above = (parsed > col.max_value).fillna(False).astype(bool)
                for r in source_row[above]:
                    issues.append(Issue(int(r), col.name,
                                        f"Value is above the maximum of {col.max_value:g}",
                                        "warning"))
                parsed = parsed.mask(above, col.max_value)
            if col.required:
                bad = bad | blank
            else:
                parsed = parsed.fillna(col.default) if col.default is not None else parsed
            if col.kind == "int":
                parsed = parsed.round()
        elif col.kind == "bool":
            parsed, bad = _coerce_bool(raw)
        else:
            parsed = raw.astype("string").str.strip()
            bad = parsed.isna() | (parsed == "")
            parsed = parsed.astype("object")
            if not col.required:
                parsed = pd.Series(
                    [col.default if (v is None or v is pd.NA or v == "") else v for v in parsed],
                    index=work.index, dtype="object")
                bad = pd.Series(False, index=work.index)

        work[col.name] = parsed

        if col.required:
            flagged = bad.fillna(False)
            for r in source_row[flagged]:
                issues.append(Issue(int(r), col.name, "Missing or unreadable value"))
            drop = drop | flagged
        else:
            flagged = bad.fillna(False) if col.kind == "bool" else pd.Series(False, index=work.index)
            for r in source_row[flagged]:
                issues.append(Issue(int(r), col.name, "Not recognised, treated as no", "warning"))

    work = work.loc[:, [c.name for c in schema]]
    work = work[~drop]
    kept_rows = source_row[~drop]

    if known_skus is not None and "sku" in work.columns and len(work):
        unknown = ~work["sku"].isin(known_skus)
        for r, sku in zip(kept_rows[unknown], work.loc[unknown, "sku"]):
            issues.append(Issue(int(r), "sku", f"{sku} is not in the catalog yet", "warning"))

    if len(work):
        key = KEYS[table]
        dupes = work.duplicated(subset=key, keep="last")
        for r in kept_rows[dupes]:
            issues.append(Issue(int(r), "+".join(key),
                                "Repeated key, the later row wins", "warning"))
        work = work[~dupes]

    if table == "history" and len(work):
        work = work.sort_values(["sku", "date"]).reset_index(drop=True)
    elif len(work):
        work = work.sort_values(KEYS[table]).reset_index(drop=True)

    return ValidationReport(table, work, issues, rows_in, len(work))


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV without guessing types, so validation owns all coercion."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=True)


def template_frame(table: str, skus: list[str] | None = None) -> pd.DataFrame:
    """A one-row example file people can open in Excel and fill in."""
    examples = {
        "catalog": {
            "sku": "TOM-RMA", "item_name": "Roma tomatoes", "category": "Produce",
            "unit": "kg", "unit_cost": 2.40, "unit_price": 5.50, "shelf_life_days": 7,
            "pack_size": 5, "min_order_qty": 5, "lead_time_days": 2,
        },
        "history": {
            "date": "2026-09-01", "sku": "TOM-RMA", "units_sold": 12,
            "units_bought": 0, "units_wasted": 1, "closing_stock": 18,
            "unit_cost": 2.40, "is_promo": 0,
        },
        "snapshot": {"sku": "TOM-RMA", "on_hand": 18, "days_to_expiry": 4},
    }[table]
    frame = pd.DataFrame([examples])
    if skus:
        frame = pd.concat(
            [frame.assign(sku=s) for s in skus], ignore_index=True)
    return frame
