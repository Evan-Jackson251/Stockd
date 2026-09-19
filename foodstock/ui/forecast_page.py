"""The forecast page: the screen the whole app exists to show.

Layout reasoning. The service level slider sits directly under the headline cost
number, because dragging it changes that number and cause should sit next to
effect. The model runs once on Run, then re-planning at a new service level or
budget is instant, since only the ordering arithmetic depends on those two.

Everything is re-planned from a cached forecast rather than refitting, so the
slider feels live.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QSlider, QTableView,
    QVBoxLayout, QWidget,
)

from core import recommend
from core.model import DemandForecaster
from core.store import DataStore

from foodstock.ui.widgets import Banner, ComparisonChart, DataFrameModel, MetricCard, divider, title_block

ORDER_COLUMNS = [
    "item_name", "category", "horizon_qty", "unit", "horizon_cost",
    "delivery_qty", "deliveries", "reorder_every_days",
    "planned_units", "on_hand", "days_to_expiry",
    "projected_waste_units", "flags",
]

HEADERS = {
    "item_name": "Item",
    "category": "Category",
    "horizon_qty": "Buy (2 weeks)",
    "unit": "Unit",
    "horizon_cost": "Est. cost",
    "delivery_qty": "Per delivery",
    "deliveries": "Deliveries",
    "reorder_every_days": "Reorder every",
    "planned_units": "Forecast sales",
    "on_hand": "On hand",
    "days_to_expiry": "Days left",
    "projected_waste_units": "Likely waste",
    "flags": "Watch out for",
}

MONEY = {"horizon_cost", "unit_cost", "delivery_cost", "projected_waste_cost"}


class ForecastPage(QWidget):
    """Runs the model, then lets the user tune the order."""

    needs_setup = Signal()

    def __init__(self, store: DataStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self._forecaster: DemandForecaster | None = None
        self._forecast = None
        self._plan: recommend.OrderPlan | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(30, 26, 30, 26)
        outer.setSpacing(16)

        outer.addWidget(title_block(
            "Order plan for the next two weeks",
            "Built from your sales history, what is on the shelf, and how long each "
            "item keeps."))

        self.state_banner = Banner()
        outer.addWidget(self.state_banner)

        outer.addLayout(self._build_metrics())
        outer.addWidget(self._build_controls())
        outer.addWidget(divider())
        outer.addWidget(self._build_results(), stretch=1)

        self.refresh()

    # ------------------------------------------------------------- building

    def _build_metrics(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        self.card_cost = MetricCard("Estimated cost", "—", lead=True)
        self.card_items = MetricCard("Items to order", "—")
        self.card_waste = MetricCard("Likely waste", "—")
        self.card_risk = MetricCard("Needs attention", "—")
        for card in (self.card_cost, self.card_items, self.card_waste, self.card_risk):
            row.addWidget(card)
        return row

    def _build_controls(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(10)

        self.run_button = QPushButton("Run forecast")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run)
        top.addWidget(self.run_button)

        top.addWidget(QLabel("Budget cap"))
        self.budget_input = QDoubleSpinBox()
        self.budget_input.setRange(0, 10_000_000)
        self.budget_input.setDecimals(0)
        self.budget_input.setSingleStep(100)
        self.budget_input.setPrefix("$ ")
        self.budget_input.setSpecialValueText("No cap")
        self.budget_input.setValue(0)
        self.budget_input.setToolTip(
            "Leave at no cap to see the full order. With a cap, the lines worth the least "
            "are cut first.")
        self.budget_input.valueChanged.connect(self._replan)
        top.addWidget(self.budget_input)

        self.only_orders = QCheckBox("Hide items with nothing to order")
        self.only_orders.setChecked(True)
        self.only_orders.toggled.connect(self._render_table)
        top.addWidget(self.only_orders)

        top.addStretch(1)

        self.export_button = QPushButton("Export order sheet")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        top.addWidget(self.export_button)

        self.copy_button = QPushButton("Copy summary")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self._copy_summary)
        top.addWidget(self.copy_button)

        layout.addLayout(top)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(12)

        slider_label = QLabel("How full do you want the shelves?")
        slider_row.addWidget(slider_label)

        self.service_slider = QSlider(Qt.Orientation.Horizontal)
        self.service_slider.setRange(10, 90)
        self.service_slider.setValue(60)
        self.service_slider.setSingleStep(5)
        self.service_slider.setPageStep(10)
        self.service_slider.setMinimumWidth(240)
        self.service_slider.valueChanged.connect(self._replan)
        slider_row.addWidget(self.service_slider, stretch=1)

        self.service_label = QLabel()
        self.service_label.setMinimumWidth(320)
        self.service_label.setWordWrap(True)
        self.service_label.setObjectName("hint")
        slider_row.addWidget(self.service_label)

        layout.addLayout(slider_row)

        self.accuracy_label = QLabel()
        self.accuracy_label.setObjectName("hint")
        self.accuracy_label.setWordWrap(True)
        layout.addWidget(self.accuracy_label)

        self._update_service_label()
        return card

    def _build_results(self) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        left = QFrame()
        left.setObjectName("card")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 14, 16, 14)
        left_layout.setSpacing(8)

        heading = QLabel("What to order")
        heading.setObjectName("sectionTitle")
        left_layout.addWidget(heading)

        self.table_model = DataFrameModel(
            headers=HEADERS, money=MONEY, colour_by="flags",
            decimals={"planned_units": 0, "horizon_qty": 0, "delivery_qty": 0})
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        left_layout.addWidget(self.table, stretch=1)

        hint = QLabel("Hover a row to see how the number was worked out.")
        hint.setObjectName("hint")
        left_layout.addWidget(hint)

        layout.addWidget(left, stretch=3)

        right = QFrame()
        right.setObjectName("card")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(8)

        chart_heading = QLabel("Forecast in context")
        chart_heading.setObjectName("sectionTitle")
        right_layout.addWidget(chart_heading)

        self.chart = ComparisonChart()
        right_layout.addWidget(self.chart, stretch=1)

        self.notes_label = QLabel()
        self.notes_label.setObjectName("hint")
        self.notes_label.setWordWrap(True)
        right_layout.addWidget(self.notes_label)

        layout.addWidget(right, stretch=2)
        return holder

    # -------------------------------------------------------------- running

    @property
    def service_level(self) -> float:
        return self.service_slider.value() / 100.0

    @property
    def budget(self) -> float | None:
        value = self.budget_input.value()
        return value if value > 0 else None

    def _update_service_label(self) -> None:
        name, blurb = recommend.service_level_label(self.service_level)
        self.service_label.setText(f"{name} — {blurb}")

    def clear_forecast(self) -> None:
        """Drop the cached forecast so a new data folder cannot show stale numbers."""
        self._forecaster = None
        self._forecast = None
        self._plan = None
        self.table_model.set_frame(pd.DataFrame())
        self.chart.set_data(pd.DataFrame())
        self.accuracy_label.setText("")
        self.notes_label.setText("")
        self.export_button.setEnabled(False)
        self.copy_button.setEnabled(False)
        for card in (self.card_cost, self.card_items, self.card_waste, self.card_risk):
            card.set_value("—", "")

    def refresh(self) -> None:
        """Called when stored data changes, to enable or explain."""
        coverage = self.store.coverage()
        catalog = self.store.load("catalog")

        if catalog.empty or coverage.rows == 0:
            self.state_banner.set_text(
                "There is no stock data yet. Run setup or import a sales export to get "
                "a forecast.", warn=True)
            self.run_button.setEnabled(False)
            return

        self.run_button.setEnabled(True)
        message = coverage.describe()
        if not coverage.has_last_year:
            message += ". The year-ago comparison switches on at 56 weeks."
        self.state_banner.set_text(message, warn=False)

    def run(self) -> None:
        """Fit the model and build the first plan."""
        catalog = self.store.load("catalog")
        history = self.store.load("history")

        if catalog.empty or history.empty:
            QMessageBox.information(
                self, "Nothing to forecast",
                "Add items and some sales history first, then run the forecast again.")
            return

        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.run_button.setText("Working…")
        self.run_button.setEnabled(False)
        QApplication.processEvents()
        try:
            self._forecaster = DemandForecaster().fit(history, catalog)
            self._forecast = self._forecaster.predict(history, catalog)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(
                self, "The forecast could not run",
                f"{exc}\n\nThis usually means the sales history is too thin or the item "
                "codes do not match between files.")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
            self.run_button.setText("Run forecast")
            self.run_button.setEnabled(True)

        scorecard = self._forecast.scorecard
        method = ("a learned model" if self._forecast.method == "gradient_boosting"
                  else "a four week average")
        self.accuracy_label.setText(
            f"Forecast made with {method}, from history up to "
            f"{self._forecast.as_of_week:%d %b %Y}. {scorecard.describe()}")

        self._replan()

    def _replan(self) -> None:
        """Re-run only the ordering arithmetic. Cheap enough to be live."""
        self._update_service_label()
        if self._forecast is None:
            return

        catalog = self.store.load("catalog")
        snapshot = self.store.load("snapshot")
        try:
            self._plan = recommend.build_plan(
                self._forecast, catalog, snapshot,
                service_level=self.service_level, budget=self.budget)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot build an order", str(exc))
            return

        self._render_metrics()
        self._render_table()
        self.chart.set_data(self._plan.lines)

        notes = list(self._plan.notes)
        self.notes_label.setText("\n\n".join(notes) if notes else "")
        self.export_button.setEnabled(True)
        self.copy_button.setEnabled(True)

    # ------------------------------------------------------------ rendering

    def _render_metrics(self) -> None:
        plan = self._plan
        summary = plan.summary()

        budget_note = ""
        if summary["budget"]:
            left = summary["budget_left"]
            budget_note = (f"{left:,.0f} under the cap" if left >= 0
                           else f"{abs(left):,.0f} over the cap")
        else:
            budget_note = f"first delivery {summary['first_delivery_cost']:,.0f}"
        self.card_cost.set_value(f"${summary['total_cost']:,.0f}", budget_note)

        self.card_items.set_value(
            str(summary["items_ordered"]),
            f"{summary['items_skipped']} need nothing" if summary["items_skipped"]
            else "everything needs topping up")

        self.card_waste.set_value(
            f"${summary['projected_waste_cost']:,.0f}",
            f"{summary['waste_share_pct']:.0f}% of the order, "
            f"{summary['projected_waste_units']:,.0f} units")

        attention = summary["order_today"] + summary["stockout_risks"]
        bits = []
        if summary["order_today"]:
            bits.append(f"{summary['order_today']} to order today")
        if summary["stockout_risks"]:
            bits.append(f"{summary['stockout_risks']} could run short")
        if summary["deferred"]:
            bits.append(f"{summary['deferred']} held back")
        self.card_risk.set_value(str(attention), ", ".join(bits) or "nothing urgent")

    def _render_table(self) -> None:
        if self._plan is None:
            return
        lines = self._plan.ordered if self.only_orders.isChecked() else self._plan.lines
        frame = lines[[c for c in ORDER_COLUMNS if c in lines.columns]].copy()
        if "reason" in lines.columns:
            frame["reason"] = lines["reason"].to_numpy()
        self.table_model.set_frame(frame)
        self.table.resizeColumnsToContents()
        if "reason" in frame.columns:
            self.table.setColumnHidden(list(frame.columns).index("reason"), True)

    # -------------------------------------------------------------- outputs

    def _summary_text(self) -> str:
        plan = self._plan
        summary = plan.summary()
        name, _ = recommend.service_level_label(plan.service_level)

        lines = [
            f"Order plan for the {plan.horizon_days} days from "
            f"{plan.as_of_week:%d %b %Y}",
            f"Shelf target: {name}",
            f"Items to order: {summary['items_ordered']}",
            f"Estimated cost: ${summary['total_cost']:,.2f}",
            f"First delivery: ${summary['first_delivery_cost']:,.2f}",
            f"Likely waste: {summary['projected_waste_units']:,.0f} units "
            f"(${summary['projected_waste_cost']:,.2f})",
            "",
        ]
        if summary["budget"]:
            lines.insert(4, f"Budget cap: ${summary['budget']:,.2f}")

        for row in plan.order_sheet().itertuples():
            if row.deliveries > 1:
                lines.append(
                    f"{row.item_name}: {row.horizon_qty:,.0f} {row.unit} total, as "
                    f"{row.deliveries} deliveries of {row.delivery_qty:,.0f} every "
                    f"{row.reorder_every_days} days — ${row.horizon_cost:,.2f}")
            else:
                lines.append(f"{row.item_name}: {row.horizon_qty:,.0f} {row.unit} "
                             f"— ${row.horizon_cost:,.2f}")

        flagged = plan.flagged
        if not flagged.empty:
            lines.append("")
            lines.append("Watch out for:")
            for row in flagged.itertuples():
                lines.append(f"  {row.item_name}: {row.flags}")

        return "\n".join(lines)

    def _copy_summary(self) -> None:
        QGuiApplication.clipboard().setText(self._summary_text())
        self.state_banner.set_text("Order summary copied to the clipboard.", warn=False)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save order sheet",
            f"order-{pd.Timestamp.today():%Y-%m-%d}.csv", "CSV files (*.csv)")
        if not path:
            return
        self._plan.order_sheet().to_csv(path, index=False)

        summary_path = path.rsplit(".", 1)[0] + "-summary.txt"
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(self._summary_text())

        QMessageBox.information(
            self, "Order sheet saved",
            f"Saved the order to {path}\nand a readable summary to {summary_path}")
