"""On-disk storage for the three tables.

Every import is archived verbatim under data/imports/ before anything is merged,
so a bad file can always be traced back to the sheet it came from. The merged
tables live as plain CSVs in the data folder, which means a user can open them
in Excel without the app.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from foodstock.core import schema
from foodstock.core.schema import ValidationReport


@dataclass
class MergeResult:
    """What changed when a file was folded into the stored table."""

    table: str
    report: ValidationReport
    added: int
    replaced: int
    archived_as: Path | None

    def summary(self) -> str:
        if self.added == 0 and self.replaced == 0:
            return "Nothing new to add. Every row was already stored."
        bits = []
        if self.added:
            bits.append(f"{self.added} new rows")
        if self.replaced:
            bits.append(f"{self.replaced} rows updated")
        return f"Added to {self.table}: " + " and ".join(bits) + "."


@dataclass
class Coverage:
    """How much history is stored, used to tell the user what accuracy to expect."""

    rows: int
    skus: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None

    @property
    def weeks(self) -> int:
        if self.first_date is None or self.last_date is None:
            return 0
        return int((self.last_date - self.first_date).days // 7) + 1

    @property
    def has_last_year(self) -> bool:
        return self.weeks >= 56

    def describe(self) -> str:
        if self.rows == 0:
            return "No history stored yet."
        span = f"{self.first_date:%d %b %Y} to {self.last_date:%d %b %Y}"
        note = "includes the same period last year" if self.has_last_year \
            else "not enough to compare against last year yet"
        return f"{self.rows:,} rows · {self.skus} items · {span} ({self.weeks} weeks, {note})"


class DataStore:
    """Reads, writes and merges the app's CSV tables."""

    FILES = {"catalog": "catalog.csv", "history": "history.csv", "snapshot": "snapshot.csv"}

    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        self.imports = self.root / "imports"
        self.root.mkdir(parents=True, exist_ok=True)
        self.imports.mkdir(parents=True, exist_ok=True)

    def path(self, table: str) -> Path:
        return self.root / self.FILES[table]

    def exists(self, table: str) -> bool:
        return self.path(table).exists()

    def load(self, table: str) -> pd.DataFrame:
        """Load a stored table, returning an empty frame if it is not there yet."""
        target = self.path(table)
        if not target.exists():
            return schema.blank_frame(table)
        report = schema.validate(schema.read_csv(str(target)), table)
        return report.frame

    def save(self, table: str, frame: pd.DataFrame) -> Path:
        target = self.path(table)
        out = frame.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        if "is_promo" in out.columns:
            out["is_promo"] = out["is_promo"].astype(int)
        out.to_csv(target, index=False)
        return target

    def archive(self, source: str | Path) -> Path:
        """Keep an untouched copy of an imported file."""
        source = Path(source)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = self.imports / f"{stamp}_{source.name}"
        shutil.copy2(source, destination)
        return destination

    def ingest(self, source: str | Path, table: str | None = None) -> MergeResult:
        """Validate a CSV, archive it, and merge its rows into the stored table."""
        source = Path(source)
        raw = schema.read_csv(str(source))

        table = table or schema.detect_table(raw)
        if table is None:
            report = ValidationReport("unknown", pd.DataFrame(), [
                schema.Issue(None, "header",
                             "Could not tell what this file holds. Expected columns for "
                             "catalog, history or snapshot")], len(raw), 0)
            return MergeResult("unknown", report, 0, 0, None)

        known = set(self.load("catalog")["sku"]) if table != "catalog" else None
        report = schema.validate(raw, table, known_skus=known)
        if not report.ok:
            return MergeResult(table, report, 0, 0, None)

        archived = self.archive(source)
        added, replaced = self.merge(table, report.frame)
        return MergeResult(table, report, added, replaced, archived)

    def merge(self, table: str, incoming: pd.DataFrame) -> tuple[int, int]:
        """Upsert rows on the table's key. Incoming rows win on conflict."""
        key = schema.KEYS[table]
        stored = self.load(table)

        if stored.empty:
            self.save(table, incoming)
            return len(incoming), 0

        if table == "history":
            stored["date"] = pd.to_datetime(stored["date"])
            incoming = incoming.copy()
            incoming["date"] = pd.to_datetime(incoming["date"])

        stored_keys = set(map(tuple, stored[key].astype(str).to_numpy()))
        incoming_keys = list(map(tuple, incoming[key].astype(str).to_numpy()))
        replaced = sum(1 for k in incoming_keys if k in stored_keys)
        added = len(incoming) - replaced

        combined = pd.concat([stored, incoming], ignore_index=True)
        combined = combined.drop_duplicates(subset=key, keep="last")
        combined = combined.sort_values(key).reset_index(drop=True)
        self.save(table, combined)
        return added, replaced

    def replace(self, table: str, frame: pd.DataFrame) -> Path:
        """Overwrite a table outright, used by the setup wizard."""
        return self.save(table, frame)

    def coverage(self) -> Coverage:
        history = self.load("history")
        if history.empty:
            return Coverage(0, 0, None, None)
        dates = pd.to_datetime(history["date"])
        return Coverage(len(history), history["sku"].nunique(), dates.min(), dates.max())

    def archived_imports(self) -> list[Path]:
        return sorted(self.imports.glob("*.csv"), reverse=True)
