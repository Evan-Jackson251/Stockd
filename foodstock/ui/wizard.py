"""The setup wizard: walks someone through their data and writes the CSVs.

One decision shapes this whole flow. Nobody is going to hand-type two years of
daily sales, so the wizard asks for four weeks of weekly totals per item and
says plainly that importing real history is what makes the forecast sharp. Four
weeks is enough for the fallback forecaster to produce something honest on day
one, which beats an empty screen or a fake number.

Weekly totals are written out as seven equal daily rows. The model aggregates
back to weeks before it does anything, so nothing is lost, and the stored file
stays in the same one-row-per-day shape as a real till export.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWizard, QWizardPage,
)

from core import schema, synth
from core.store import DataStore

from foodstock.ui.theme import PALETTE

CATALOG_FIELDS = [
    ("sku", "Code", 90),
    ("item_name", "Item", 190),
    ("category", "Category", 120),
    ("unit", "Unit", 80),
    ("unit_cost", "Cost", 80),
    ("unit_price", "Sells for", 90),
    ("shelf_life_days", "Shelf life (days)", 120),
    ("pack_size", "Pack size", 90),
    ("min_order_qty", "Min order", 90),
    ("lead_time_days", "Lead time (days)", 120),
]

CATEGORY_SUGGESTIONS = ["Produce", "Dairy", "Bakery", "Meat", "Dry goods", "Beverage", "Frozen"]

STARTER_ROWS = [
    ("", "", "Produce", "kg", "", "", "5", "1", "0", "2"),
    ("", "", "Dairy", "each", "", "", "14", "1", "0", "3"),
    ("", "", "Bakery", "each", "", "", "2", "12", "0", "1"),
]


def _cell(text: str = "", editable: bool = True, muted: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if muted:
        item.setForeground(Qt.GlobalColor.gray)
    return item


def _stretch_table(table: QTableWidget, stretch_column: int | None = None) -> None:
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    if stretch_column is not None:
        header.setSectionResizeMode(stretch_column, QHeaderView.ResizeMode.Stretch)


class IntroPage(QWizardPage):
    """Explains the three files and offers the demo shortcut."""

    def __init__(self, wizard: "SetupWizard"):
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Set up your stock data")
        self.setSubTitle("Three short steps. You can change any of it later.")

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        explainer = QLabel(
            "The app keeps three files:\n\n"
            "   Items — what you stock, what it costs, how long it keeps\n"
            "   Sales history — what sold, what was delivered, what got thrown out\n"
            "   Current stock — what is on the shelf right now\n\n"
            "This wizard creates all three. Sales history is the one that decides how "
            "accurate the forecast is, so it starts with four weeks typed in by hand and "
            "gets better every time you import a real export from your till.")
        explainer.setWordWrap(True)
        layout.addWidget(explainer)

        self.demo_check = QCheckBox(
            "Fill everything with two years of sample cafe data instead")
        self.demo_check.setToolTip(
            "Loads a simulated cafe with 16 items so you can try the forecast immediately. "
            "This replaces any data already stored.")
        self.demo_check.toggled.connect(self._on_demo_toggled)
        layout.addWidget(self.demo_check)

        self.folder_label = QLabel()
        self.folder_label.setObjectName("hint")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        pick = QPushButton("Choose where to save the files")
        pick.clicked.connect(self._pick_folder)
        layout.addWidget(pick, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)

        self._refresh_folder()

    def _refresh_folder(self) -> None:
        self.folder_label.setText(f"Saving to {self.wizard_ref.store.root.resolve()}")

    def _pick_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a data folder", str(self.wizard_ref.store.root.resolve()))
        if chosen:
            self.wizard_ref.store = DataStore(chosen)
            self._refresh_folder()

    def _on_demo_toggled(self, checked: bool) -> None:
        self.wizard_ref.use_demo = checked
        self.wizard_ref.button(QWizard.WizardButton.NextButton).setText(
            "Load sample data" if checked else "Next")

    def nextId(self) -> int:
        # Demo data skips straight to the finish.
        return SetupWizard.PAGE_DONE if self.wizard_ref.use_demo else SetupWizard.PAGE_ITEMS


class ItemsPage(QWizardPage):
    """The catalog: the static facts about each item."""

    def __init__(self, wizard: "SetupWizard"):
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("What do you stock?")
        self.setSubTitle("Shelf life is the important one. It decides how much the app "
                         "will let you order in a single delivery.")

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, len(CATALOG_FIELDS))
        self.table.setHorizontalHeaderLabels([label for _, label, _ in CATALOG_FIELDS])
        for i, (_, _, width) in enumerate(CATALOG_FIELDS):
            self.table.setColumnWidth(i, width)
        _stretch_table(self.table, stretch_column=1)
        self.table.itemChanged.connect(lambda *_: self.completeChanged.emit())
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        add = QPushButton("Add item")
        add.clicked.connect(lambda: self._add_row())
        remove = QPushButton("Remove selected")
        remove.setObjectName("danger")
        remove.clicked.connect(self._remove_rows)
        load = QPushButton("Import from a CSV instead")
        load.clicked.connect(self._import_csv)
        template = QPushButton("Save a blank template")
        template.clicked.connect(self._save_template)

        for widget in (add, remove, load, template):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status = QLabel()
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        for row in STARTER_ROWS:
            self._add_row(row)

    def _add_row(self, values: tuple[str, ...] | None = None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = values or ("",) * len(CATALOG_FIELDS)
        for column, value in enumerate(values):
            self.table.setItem(row, column, _cell(str(value)))

    def _remove_rows(self) -> None:
        for index in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(index)
        self.completeChanged.emit()

    def _save_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save items template", "items-template.csv", "CSV files (*.csv)")
        if path:
            schema.template_frame("catalog").to_csv(path, index=False)
            self.status.setText(f"Template saved to {path}. Fill it in and import it here.")

    def _import_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an items file", "", "CSV files (*.csv)")
        if not path:
            return
        report = schema.validate(schema.read_csv(path), "catalog")
        if not report.ok:
            problems = "\n".join(i.describe() for i in report.errors[:8])
            QMessageBox.warning(self, "That file needs a fix",
                                f"{report.summary()}\n\n{problems}")
            return

        self.table.setRowCount(0)
        for row in report.frame.itertuples():
            self._add_row(tuple(
                "" if pd.isna(getattr(row, name, np.nan)) else str(getattr(row, name))
                for name, _, _ in CATALOG_FIELDS))
        self.status.setText(f"Loaded {report.rows_kept} items. {report.summary()}")
        self.completeChanged.emit()

    def frame(self) -> pd.DataFrame:
        """Read the grid back out, skipping blank rows."""
        records = []
        for row in range(self.table.rowCount()):
            record = {}
            for column, (name, _, _) in enumerate(CATALOG_FIELDS):
                item = self.table.item(row, column)
                record[name] = item.text().strip() if item else ""
            if record["sku"] and record["item_name"]:
                records.append(record)
        return pd.DataFrame(records)

    def isComplete(self) -> bool:
        return len(self.frame()) > 0

    def validatePage(self) -> bool:
        report = schema.validate(self.frame(), "catalog")
        if not report.ok:
            problems = "\n".join(i.describe() for i in report.errors[:8])
            QMessageBox.warning(self, "Some items are missing details",
                                f"{report.summary()}\n\n{problems}\n\n"
                                "Code, item, category, cost and shelf life are all needed.")
            return False
        self.wizard_ref.catalog = report.frame
        return True


class SalesPage(QWizardPage):
    """Four weeks of weekly sales totals per item."""

    WEEKS = 4

    def __init__(self, wizard: "SetupWizard"):
        super().__init__()
        self.wizard_ref = wizard
        self._weeks: list[date] = []
        self.setTitle("How much sold recently?")
        self.setSubTitle("Rough weekly totals are fine. This gets the forecast started; "
                         "importing a real sales export later is what makes it accurate.")

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 2 + self.WEEKS)
        _stretch_table(self.table, stretch_column=1)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        fill = QPushButton("Copy the first week across")
        fill.setToolTip("Useful when trade is steady week to week.")
        fill.clicked.connect(self._copy_across)
        row.addWidget(fill)
        row.addStretch(1)
        layout.addLayout(row)

        self.status = QLabel("Leave an item blank if you do not know it. "
                             "Blank items are skipped rather than treated as zero sales.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def initializePage(self) -> None:
        catalog = self.wizard_ref.catalog
        today = date.today()
        this_monday = today - timedelta(days=today.weekday())
        self._weeks = [this_monday - timedelta(days=7 * (self.WEEKS - i))
                       for i in range(self.WEEKS)]

        headers = ["Code", "Item"] + [f"Week of {w:%d %b}" for w in self._weeks]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(0)

        for record in catalog.itertuples():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, _cell(str(record.sku), editable=False, muted=True))
            self.table.setItem(row, 1, _cell(str(record.item_name), editable=False))
            for column in range(2, 2 + self.WEEKS):
                self.table.setItem(row, column, _cell(""))
        self.table.setColumnWidth(0, 90)

    def _copy_across(self) -> None:
        for row in range(self.table.rowCount()):
            first = self.table.item(row, 2)
            if not first or not first.text().strip():
                continue
            for column in range(3, 2 + self.WEEKS):
                self.table.setItem(row, column, _cell(first.text().strip()))

    def frame(self) -> pd.DataFrame:
        """Expand weekly totals into daily rows.

        Splitting a weekly total evenly across seven days keeps the stored file
        in the same shape as a real till export. The model re-aggregates to weeks
        immediately, so the even split costs nothing.
        """
        records = []
        for row in range(self.table.rowCount()):
            sku_item = self.table.item(row, 0)
            if not sku_item:
                continue
            sku = sku_item.text().strip()

            for offset, monday in enumerate(self._weeks):
                cell = self.table.item(row, 2 + offset)
                text = cell.text().strip() if cell else ""
                if not text:
                    continue
                try:
                    weekly_total = float(text.replace(",", ""))
                except ValueError:
                    continue
                if weekly_total < 0:
                    continue

                per_day = weekly_total / 7.0
                for day in range(7):
                    records.append({
                        "date": monday + timedelta(days=day),
                        "sku": sku,
                        "units_sold": round(per_day, 3),
                        "units_bought": 0.0,
                        "units_wasted": 0.0,
                        "closing_stock": np.nan,
                        "unit_cost": np.nan,
                        "is_promo": False,
                    })
        return pd.DataFrame(records)

    def validatePage(self) -> bool:
        frame = self.frame()
        if frame.empty:
            answer = QMessageBox.question(
                self, "No sales entered",
                "Without any sales the app cannot forecast yet. You can still set up "
                "your items and import history later.\n\nContinue anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self.wizard_ref.history = frame
        return True


class StockPage(QWizardPage):
    """What is on the shelf right now, and how long it has left."""

    def __init__(self, wizard: "SetupWizard"):
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("What is in stock right now?")
        self.setSubTitle("Stock that expires before it can be sold is not counted as "
                         "available, so the days left column matters as much as the count.")

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Code", "Item", "Units on hand", "Days left"])
        _stretch_table(self.table, stretch_column=1)
        layout.addWidget(self.table)

        hint = QLabel("Leave days left blank to assume the stock is fresh and has its "
                      "full shelf life remaining.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def initializePage(self) -> None:
        catalog = self.wizard_ref.catalog
        self.table.setRowCount(0)
        for record in catalog.itertuples():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, _cell(str(record.sku), editable=False, muted=True))
            self.table.setItem(row, 1, _cell(str(record.item_name), editable=False))
            self.table.setItem(row, 2, _cell("0"))
            self.table.setItem(row, 3, _cell(""))
        self.table.setColumnWidth(0, 90)

    def frame(self) -> pd.DataFrame:
        records = []
        for row in range(self.table.rowCount()):
            sku_item = self.table.item(row, 0)
            if not sku_item:
                continue
            on_hand_item = self.table.item(row, 2)
            expiry_item = self.table.item(row, 3)
            try:
                on_hand = float((on_hand_item.text() or "0").replace(",", ""))
            except (ValueError, AttributeError):
                on_hand = 0.0
            expiry_text = (expiry_item.text().strip() if expiry_item else "")
            try:
                days = int(float(expiry_text)) if expiry_text else None
            except ValueError:
                days = None
            records.append({"sku": sku_item.text().strip(),
                            "on_hand": max(0.0, on_hand),
                            "days_to_expiry": days})
        return pd.DataFrame(records)

    def validatePage(self) -> bool:
        self.wizard_ref.snapshot = self.frame()
        return True


class DonePage(QWizardPage):
    """Writes the files and reports what was saved."""

    def __init__(self, wizard: "SetupWizard"):
        super().__init__()
        self.wizard_ref = wizard
        self.setTitle("Saved")
        self.setFinalPage(True)

        layout = QVBoxLayout(self)
        self.report = QLabel()
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.report)
        layout.addStretch(1)

    def initializePage(self) -> None:
        store = self.wizard_ref.store
        lines: list[str] = []

        if self.wizard_ref.use_demo:
            counts = synth.write_demo_data(store)
            lines.append("Loaded two years of sample cafe data.")
            lines.append(f"{counts['catalog']} items, {counts['history']:,} daily sales rows, "
                         f"{counts['snapshot']} stock counts.")
        else:
            store.replace("catalog", self.wizard_ref.catalog)
            lines.append(f"Saved {len(self.wizard_ref.catalog)} items.")

            history = self.wizard_ref.history
            if history is not None and not history.empty:
                store.replace("history", history)
                weeks = len(history) // 7 // max(history["sku"].nunique(), 1)
                lines.append(f"Saved {len(history):,} daily sales rows "
                             f"covering about {weeks} weeks.")
            else:
                lines.append("No sales history yet. Import a sales export to forecast.")

            snapshot = self.wizard_ref.snapshot
            if snapshot is not None and not snapshot.empty:
                store.replace("snapshot", snapshot)
                lines.append(f"Saved current stock for {len(snapshot)} items.")

        coverage = store.coverage()
        lines.append("")
        lines.append(f"Files are in {store.root.resolve()}")
        lines.append(coverage.describe())
        if not coverage.has_last_year:
            lines.append("")
            lines.append("With under a year of history the forecast leans on recent weeks "
                         "only. Import older sales to switch on the year-ago comparison.")

        self.report.setText("\n".join(lines))
        self.wizard_ref.completed = True


class SetupWizard(QWizard):
    """Collects everything needed to produce a first forecast."""

    PAGE_INTRO, PAGE_ITEMS, PAGE_SALES, PAGE_STOCK, PAGE_DONE = range(5)

    def __init__(self, store: DataStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.use_demo = False
        self.completed = False
        self.catalog: pd.DataFrame = pd.DataFrame()
        self.history: pd.DataFrame = pd.DataFrame()
        self.snapshot: pd.DataFrame = pd.DataFrame()

        self.setWindowTitle("Set up stock data")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setMinimumSize(920, 620)
        self.setStyleSheet(f"QWizard {{ background: {PALETTE['bg']}; }}")

        self.setPage(self.PAGE_INTRO, IntroPage(self))
        self.setPage(self.PAGE_ITEMS, ItemsPage(self))
        self.setPage(self.PAGE_SALES, SalesPage(self))
        self.setPage(self.PAGE_STOCK, StockPage(self))
        self.setPage(self.PAGE_DONE, DonePage(self))
        self.setStartId(self.PAGE_INTRO)
