"""Main window: sidebar navigation over three pages.

The order of the sidebar follows the order someone actually does the work in:
set up the items, feed in history, get the order. First run opens the wizard,
because an empty forecast screen explains nothing.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from core.store import DataStore

from foodstock.ui.forecast_page import ForecastPage
from foodstock.ui.import_page import ImportPage
from foodstock.ui.theme import stylesheet
from foodstock.ui.widgets import title_block
from foodstock.ui.wizard import SetupWizard


class SetupPage(QWidget):
    """A landing page that explains the state of the data and opens the wizard."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window_ref = window

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 26, 30, 26)
        layout.setSpacing(18)

        layout.addWidget(title_block(
            "Your stock data",
            "The app keeps three files: your items, your sales history, and what is "
            "on the shelf right now."))

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(14)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card_layout.addWidget(self.status)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        run_wizard = QPushButton("Run setup")
        run_wizard.setObjectName("primary")
        run_wizard.clicked.connect(self.window_ref.open_wizard)
        buttons.addWidget(run_wizard)

        change_folder = QPushButton("Change data folder")
        change_folder.clicked.connect(self._change_folder)
        buttons.addWidget(change_folder)

        open_folder = QPushButton("Show the files")
        open_folder.clicked.connect(self._show_files)
        buttons.addWidget(open_folder)

        buttons.addStretch(1)
        card_layout.addLayout(buttons)

        layout.addWidget(card)
        layout.addStretch(1)
        self.refresh()

    def _change_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a data folder", str(self.window_ref.store.root.resolve()))
        if chosen:
            self.window_ref.set_store(DataStore(chosen))

    def _show_files(self) -> None:
        store = self.window_ref.store
        files = [store.path(t) for t in ("catalog", "history", "snapshot")]
        listing = "\n".join(
            f"{'  saved  ' if p.exists() else ' missing  '}{p.name}" for p in files)
        imports = store.archived_imports()
        extra = (f"\n\n{len(imports)} imported file"
                 f"{'s' if len(imports) != 1 else ''} archived in "
                 f"{store.imports.name}/") if imports else ""
        QMessageBox.information(
            self, "Data files",
            f"{store.root.resolve()}\n\n{listing}{extra}")

    def refresh(self) -> None:
        store = self.window_ref.store
        coverage = store.coverage()
        catalog = store.load("catalog")
        snapshot = store.load("snapshot")

        lines = [f"Folder: {store.root.resolve()}", ""]
        lines.append(f"Items: {len(catalog)}" if len(catalog)
                     else "Items: none yet — run setup to add them")
        lines.append(f"Sales history: {coverage.describe()}")
        lines.append(f"Current stock: {len(snapshot)} items counted" if len(snapshot)
                     else "Current stock: not recorded, so stock is treated as empty")

        if len(catalog) and coverage.rows:
            lines.append("")
            lines.append("Ready to forecast.")
        self.status.setText("\n".join(lines))


class MainWindow(QMainWindow):
    """Shell holding the sidebar and the three pages."""

    NAV = [("Stock data", 0), ("Add history", 1), ("Order plan", 2)]

    def __init__(self, store: DataStore):
        super().__init__()
        self.store = store

        self.setWindowTitle("Stock Planner")
        self.resize(1320, 840)
        self.setMinimumSize(1060, 700)
        self.setStyleSheet(stylesheet())

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())

        self.pages = QStackedWidget()
        self.setup_page = SetupPage(self)
        self.import_page = ImportPage(store)
        self.forecast_page = ForecastPage(store)

        self.import_page.data_changed.connect(self.refresh_all)

        self.pages.addWidget(self.setup_page)
        self.pages.addWidget(self.import_page)
        self.pages.addWidget(self.forecast_page)
        layout.addWidget(self.pages, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(f"Data folder: {store.root.resolve()}")

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(208)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        brand = QLabel("Stock Planner")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        sub = QLabel("Two week ordering for\nsmall kitchens")
        sub.setObjectName("brandSub")
        layout.addWidget(sub)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        for label, index in self.NAV:
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.clicked.connect(lambda _=False, i=index: self._go(i))
            self.nav_group.addButton(button, index)
            layout.addWidget(button)

        layout.addStretch(1)
        return sidebar

    def _go(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        if index == 0:
            self.setup_page.refresh()
        elif index == 1:
            self.import_page.refresh()
        else:
            self.forecast_page.refresh()

    def set_store(self, store: DataStore) -> None:
        """Point every page at a different data folder."""
        self.store = store
        self.import_page.store = store
        self.forecast_page.store = store
        self.forecast_page.clear_forecast()
        self.statusBar().showMessage(f"Data folder: {store.root.resolve()}")
        self.refresh_all()

    def refresh_all(self) -> None:
        self.setup_page.refresh()
        self.import_page.refresh()
        self.forecast_page.refresh()

    def open_wizard(self) -> None:
        wizard = SetupWizard(self.store, self)
        wizard.exec()
        if wizard.completed:
            self.set_store(wizard.store)
            self._go(2)
            self.nav_group.button(2).setChecked(True)

    def maybe_first_run(self) -> None:
        """Open the wizard when there is nothing to work with."""
        if self.store.load("catalog").empty and self.store.coverage().rows == 0:
            self.open_wizard()
