"""Shared widgets: a pandas table model, metric cards, and a small chart.

The chart is hand drawn with QPainter rather than pulled from a charting library
so the app has no dependency beyond PySide6 and the data stack.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from foodstock.ui.theme import PALETTE, flag_colour

RIGHT = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
LEFT = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


class DataFrameModel(QAbstractTableModel):
    """Read-only view of a DataFrame with per-column formatting.

    Numbers are right aligned and money is formatted once here rather than in
    every call site, so the order table and the export always agree.
    """

    def __init__(self, frame: pd.DataFrame | None = None,
                 headers: dict[str, str] | None = None,
                 money: set[str] | None = None,
                 decimals: dict[str, int] | None = None,
                 colour_by: str | None = None,
                 currency: str = "$"):
        super().__init__()
        self._frame = frame if frame is not None else pd.DataFrame()
        self._headers = headers or {}
        self._money = money or set()
        self._decimals = decimals or {}
        self._colour_by = colour_by
        self._currency = currency

    # -- Qt plumbing ------------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame.columns)

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            name = str(self._frame.columns[section])
            return self._headers.get(name, name.replace("_", " ").capitalize())
        return str(section + 1)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        column = str(self._frame.columns[index.column()])
        value = self._frame.iat[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            return self._format(column, value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return RIGHT if self._is_numeric(column) else LEFT

        if role == Qt.ItemDataRole.ForegroundRole and self._colour_by:
            if column == self._colour_by:
                colour = flag_colour(str(value))
                if colour:
                    return QColor(colour)
            return QColor(PALETTE["ink"])

        if role == Qt.ItemDataRole.ToolTipRole:
            if "reason" in self._frame.columns:
                return str(self._frame.iloc[index.row()]["reason"])
        return None

    # -- formatting -------------------------------------------------------

    def _is_numeric(self, column: str) -> bool:
        return pd.api.types.is_numeric_dtype(self._frame[column])

    def _format(self, column: str, value) -> str:
        if value is None or value is pd.NA:
            return "—"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%d %b %Y")
        # numpy scalars are not instances of int, so test the value not the type.
        if isinstance(value, (bool, str)):
            return str(value)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if pd.isna(number):
            return "—"
        if column in self._money:
            return f"{self._currency}{number:,.2f}"
        places = self._decimals.get(column, 0 if number.is_integer() else 1)
        return f"{number:,.{places}f}"

    # -- updates ----------------------------------------------------------

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame.reset_index(drop=True)
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame


class MetricCard(QFrame):
    """One headline number with a label and an optional note underneath."""

    def __init__(self, label: str, value: str = "—", note: str = "",
                 lead: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("metricCardLead" if lead else "metricCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(2)

        self._label = QLabel(label)
        self._label.setObjectName("metricLabel")
        self._value = QLabel(value)
        self._value.setObjectName("metricValueAccent" if lead else "metricValue")
        self._note = QLabel(note)
        self._note.setObjectName("metricNote")
        self._note.setWordWrap(True)
        self._note.setVisible(bool(note))

        layout.addWidget(self._label)
        layout.addWidget(self._value)
        layout.addWidget(self._note)

    def set_value(self, value: str, note: str = "") -> None:
        self._value.setText(value)
        self._note.setText(note)
        self._note.setVisible(bool(note))


class Banner(QFrame):
    """An inline message. Used for coverage notes and model caveats."""

    def __init__(self, text: str = "", warn: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("bannerWarn" if warn else "banner")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(13, 10, 13, 10)
        self._label = QLabel(text)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("background: transparent;")
        layout.addWidget(self._label)
        self.setVisible(bool(text))

    def set_text(self, text: str, warn: bool = False) -> None:
        self._label.setText(text)
        self.setObjectName("bannerWarn" if warn else "banner")
        self.style().unpolish(self)
        self.style().polish(self)
        self.setVisible(bool(text))


class ComparisonChart(QWidget):
    """Grouped bars comparing the forecast against the two obvious benchmarks.

    This exists to answer the first question anyone asks of a forecast: is it
    just repeating what happened last time? Three bars per item make that
    visible without reading a table.
    """

    SERIES = [
        ("Forecast", "accent"),
        ("Last 2 weeks", "slate"),
        ("Same time last year", "line_strong"),
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._rows: list[tuple[str, float, float, float]] = []
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _number(value) -> float:
        """NaN is truthy, so `value or 0` is not enough here.

        The year-ago column is genuinely missing for any item with under a year
        of history, and an unguarded NaN poisons the scale for every bar.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if pd.isna(number) else max(number, 0.0)

    def set_data(self, frame: pd.DataFrame, limit: int = 8) -> None:
        """Take the busiest items by forecast volume."""
        if frame.empty or "planned_units" not in frame.columns:
            self._rows = []
        else:
            top = frame.nlargest(min(limit, len(frame)), "planned_units")
            self._rows = [
                (str(r.item_name),
                 self._number(r.planned_units),
                 self._number(getattr(r, "recent_2w", 0)),
                 self._number(getattr(r, "last_year_2w", 0)))
                for r in top.itertuples()
            ]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width, height = self.width(), self.height()
        painter.fillRect(0, 0, width, height, QColor(PALETTE["surface"]))

        if not self._rows:
            painter.setPen(QColor(PALETTE["ink_muted"]))
            painter.drawText(QRectF(0, 0, width, height),
                             Qt.AlignmentFlag.AlignCenter,
                             "Run a forecast to compare it against last year")
            painter.end()
            return

        legend_h, label_h = 26, 42
        pad_left, pad_right, pad_top = 14, 14, 10
        plot_top = pad_top + legend_h
        plot_bottom = height - label_h
        plot_height = max(plot_bottom - plot_top, 10)
        plot_width = width - pad_left - pad_right

        self._draw_legend(painter, pad_left, pad_top, legend_h)

        peak = max(max(r[1], r[2], r[3]) for r in self._rows)
        if not peak or peak <= 0:
            peak = 1.0

        # Gridlines give the eye something to measure against.
        painter.setPen(QPen(QColor(PALETTE["line"]), 1, Qt.PenStyle.DashLine))
        small = QFont(painter.font())
        small.setPointSizeF(max(7.5, small.pointSizeF() - 2.5))
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = plot_bottom - fraction * plot_height
            painter.drawLine(int(pad_left), int(y), int(width - pad_right), int(y))
            painter.setPen(QColor(PALETTE["ink_muted"]))
            painter.setFont(small)
            painter.drawText(QRectF(pad_left, y - 13, 46, 12),
                             LEFT, f"{peak * fraction:,.0f}")
            painter.setPen(QPen(QColor(PALETTE["line"]), 1, Qt.PenStyle.DashLine))

        painter.setPen(QColor(PALETTE["line_strong"]))
        painter.drawLine(int(pad_left), int(plot_bottom), int(width - pad_right), int(plot_bottom))

        slot = plot_width / len(self._rows)
        bar_w = min(slot / 4.4, 26)
        gap = bar_w * 0.14

        for i, (name, forecast, recent, last_year) in enumerate(self._rows):
            centre = pad_left + slot * (i + 0.5)
            group_w = bar_w * 3 + gap * 2
            x = centre - group_w / 2

            for value, (_, colour_key) in zip((forecast, recent, last_year), self.SERIES):
                bar_h = (value / peak) * plot_height
                rect = QRectF(x, plot_bottom - bar_h, bar_w, bar_h)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(PALETTE[colour_key]))
                painter.drawRoundedRect(rect, 2.5, 2.5)
                x += bar_w + gap

            painter.setPen(QColor(PALETTE["ink_soft"]))
            painter.setFont(small)
            painter.drawText(
                QRectF(centre - slot / 2, plot_bottom + 6, slot, label_h - 8),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
                    | Qt.TextFlag.TextWordWrap),
                name)

        painter.end()

    def _draw_legend(self, painter: QPainter, left: int, top: int, height: int) -> None:
        font = QFont(painter.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 2))
        painter.setFont(font)
        x = left
        for name, colour_key in self.SERIES:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(PALETTE[colour_key]))
            painter.drawRoundedRect(QRectF(x, top + height / 2 - 5, 11, 11), 2, 2)
            painter.setPen(QColor(PALETTE["ink_muted"]))
            text_w = painter.fontMetrics().horizontalAdvance(name) + 8
            painter.drawText(QRectF(x + 16, top, text_w, height), LEFT, name)
            x += 16 + text_w + 14


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    return line


def title_block(title: str, lead: str = "") -> QWidget:
    """Page heading. The lead line says what the page is for in one sentence."""
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    heading = QLabel(title)
    heading.setObjectName("pageTitle")
    layout.addWidget(heading)

    if lead:
        sub = QLabel(lead)
        sub.setObjectName("pageLead")
        sub.setWordWrap(True)
        layout.addWidget(sub)
    return holder
