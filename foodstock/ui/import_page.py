"""Import page: feed CSV files in, keep them, use them as history.

Validation results are shown before anything is merged, and the merge is an
upsert on (date, item) so re-importing a corrected export fixes the stored rows
instead of double counting them. Every file is copied into data/imports first,
which means a number that looks wrong can always be traced to its source sheet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QTableView, QVBoxLayout, QWidget,
)

from core import schema
from core.store import DataStore

from foodstock.ui.widgets import Banner, DataFrameModel, MetricCard, divider, title_block

TABLE_CHOICES = [
    ("Work it out from the columns", None),
    ("Sales history", "history"),
    ("Items", "catalog"),
    ("Current stock", "snapshot"),
]


class ImportPage(QWidget):
    """Choose files, see what is wrong with them, then merge."""

    data_changed = Signal()

    def __init__(self, store: DataStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self._pending: list[Path] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 26, 30, 26)
        outer.setSpacing(18)

        outer.addWidget(title_block(
            "Add sales history",
            "Every file you import is kept and folded into the stored history. More "
            "history means a sharper forecast, especially once it reaches back a year."))

        outer.addLayout(self._build_coverage())
        outer.addWidget(divider())
        outer.addWidget(self._build_picker())
        outer.addWidget(self._build_preview(), stretch=1)

        self.refresh()

    # ------------------------------------------------------------- building

    def _build_coverage(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        self.card_rows = MetricCard("Stored sales rows", "0")
        self.card_span = MetricCard("History covers", "—")
        self.card_items = MetricCard("Items tracked", "0")
        self.card_files = MetricCard("Files imported", "0")
        for card in (self.card_rows, self.card_span, self.card_items, self.card_files):
            row.addWidget(card)
        return row

    def _build_picker(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        heading = QLabel("Choose files")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        controls = QHBoxLayout()
        controls.setSpacing(10)

        pick = QPushButton("Choose CSV files")
        pick.setObjectName("primary")
        pick.clicked.connect(self._choose_files)
        controls.addWidget(pick)

        self.type_choice = QComboBox()
        for label, _ in TABLE_CHOICES:
            self.type_choice.addItem(label)
        self.type_choice.setToolTip(
            "Only needed when a file's column names are unusual enough that the app "
            "cannot tell what it holds.")
        controls.addWidget(QLabel("Treat as:"))
        controls.addWidget(self.type_choice)

        controls.addStretch(1)

        template = QPushButton("Save a blank template")
        template.clicked.connect(self._save_template)
        controls.addWidget(template)
        layout.addLayout(controls)

        hint = QLabel(
            "A sales file needs a date, an item code and units sold. Deliveries, waste, "
            "closing stock, cost and promotion flags are all optional and all make the "
            "forecast better. Common column names like \"qty sold\" or \"spoilage\" are "
            "recognised automatically.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(96)
        self.file_list.setVisible(False)
        layout.addWidget(self.file_list)

        self.banner = Banner()
        layout.addWidget(self.banner)

        actions = QHBoxLayout()
        self.merge_button = QPushButton("Add to stored history")
        self.merge_button.setObjectName("primary")
        self.merge_button.setEnabled(False)
        self.merge_button.clicked.connect(self._merge)
        actions.addWidget(self.merge_button)

        self.clear_button = QPushButton("Clear selection")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self._clear)
        actions.addWidget(self.clear_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        return card

    def _build_preview(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        self.preview_title = QLabel("Nothing selected yet")
        self.preview_title.setObjectName("sectionTitle")
        layout.addWidget(self.preview_title)

        self.issues = QListWidget()
        self.issues.setMaximumHeight(120)
        self.issues.setVisible(False)
        layout.addWidget(self.issues)

        self.preview_model = DataFrameModel()
        self.preview = QTableView()
        self.preview.setModel(self.preview_model)
        self.preview.setAlternatingRowColors(True)
        self.preview.verticalHeader().setVisible(False)
        self.preview.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        layout.addWidget(self.preview, stretch=1)

        return card

    # -------------------------------------------------------------- actions

    def _selected_table(self) -> str | None:
        return TABLE_CHOICES[self.type_choice.currentIndex()][1]

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose CSV files to import", "", "CSV files (*.csv);;All files (*)")
        if not paths:
            return
        self._pending = [Path(p) for p in paths]

        self.file_list.clear()
        for path in self._pending:
            self.file_list.addItem(QListWidgetItem(path.name))
        self.file_list.setVisible(True)
        self.clear_button.setEnabled(True)

        self._preview_first()

    def _preview_first(self) -> None:
        """Validate the first file so problems surface before any merge."""
        path = self._pending[0]
        raw = schema.read_csv(str(path))
        table = self._selected_table() or schema.detect_table(raw)

        if table is None:
            self.preview_title.setText(f"{path.name} — cannot tell what this file holds")
            self.banner.set_text(
                "The columns do not look like sales, items or stock. Pick a type from the "
                "dropdown, or check the header row is the first row of the file.", warn=True)
            self.merge_button.setEnabled(False)
            self.preview_model.set_frame(raw.head(12))
            self.issues.setVisible(False)
            return

        known = set(self.store.load("catalog")["sku"]) if table != "catalog" else None
        report = schema.validate(raw, table, known_skus=known)

        label = dict((v, k) for k, v in TABLE_CHOICES if v).get(table, table)
        extra = f" and {len(self._pending) - 1} more" if len(self._pending) > 1 else ""
        self.preview_title.setText(f"{path.name}{extra} — read as {label.lower()}")

        self.issues.clear()
        for issue in report.issues[:40]:
            item = QListWidgetItem(issue.describe())
            item.setForeground(Qt.GlobalColor.darkRed if issue.severity == "error"
                               else Qt.GlobalColor.darkYellow)
            self.issues.addItem(item)
        if len(report.issues) > 40:
            self.issues.addItem(QListWidgetItem(
                f"and {len(report.issues) - 40} more of the same kind"))
        self.issues.setVisible(bool(report.issues))

        if report.ok:
            self.banner.set_text(f"{report.summary()}. Ready to add.", warn=bool(report.warnings))
            self.merge_button.setEnabled(True)
        else:
            self.banner.set_text(
                f"{report.summary()}. Fix the errors listed below and choose the file again.",
                warn=True)
            self.merge_button.setEnabled(False)

        self.preview_model.set_frame(report.frame.head(50) if report.rows_kept else raw.head(12))
        self.preview.resizeColumnsToContents()

    def _merge(self) -> None:
        table_override = self._selected_table()
        merged, failed = [], []

        for path in self._pending:
            try:
                result = self.store.ingest(path, table=table_override)
            except Exception as exc:                      # noqa: BLE001
                failed.append(f"{path.name}: {exc}")
                continue
            if result.report.ok:
                merged.append(f"{path.name}: {result.summary()}")
            else:
                failed.append(f"{path.name}: {result.report.summary()}")

        message = ""
        if merged:
            message += "\n".join(merged)
        if failed:
            message += ("\n\nSkipped:\n" if merged else "Skipped:\n") + "\n".join(failed)

        QMessageBox.information(self, "Import finished", message or "Nothing to import.")
        self._clear()
        self.refresh()
        self.data_changed.emit()

    def _clear(self) -> None:
        self._pending = []
        self.file_list.clear()
        self.file_list.setVisible(False)
        self.issues.setVisible(False)
        self.banner.set_text("")
        self.preview_model.set_frame(pd.DataFrame())
        self.preview_title.setText("Nothing selected yet")
        self.merge_button.setEnabled(False)
        self.clear_button.setEnabled(False)

    def _save_template(self) -> None:
        table = self._selected_table() or "history"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save template", f"{table}-template.csv", "CSV files (*.csv)")
        if path:
            schema.template_frame(table).to_csv(path, index=False)
            QMessageBox.information(
                self, "Template saved",
                f"Saved to {path}. The example row shows the expected format; replace it "
                "with your own data.")

    # -------------------------------------------------------------- refresh

    def refresh(self) -> None:
        coverage = self.store.coverage()
        self.card_rows.set_value(f"{coverage.rows:,}")
        self.card_items.set_value(str(coverage.skus))
        self.card_files.set_value(str(len(self.store.archived_imports())))

        if coverage.rows:
            self.card_span.set_value(
                f"{coverage.weeks} weeks",
                f"{coverage.first_date:%b %Y} to {coverage.last_date:%b %Y}")
        else:
            self.card_span.set_value("—", "No history yet")
