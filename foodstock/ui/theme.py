"""Visual language for the app.

The palette is built around a cold-store neutral, and colour is reserved for
meaning rather than decoration. Three signal colours only, each with one job:

  green   the order is sound
  amber   something will spoil
  berry   something will run out

A line in the order table is never coloured just to look organised, because the
whole value of this screen is that a glance tells the owner where the money and
the spoilage are.
"""

from __future__ import annotations

PALETTE = {
    "bg": "#EEF1EC",
    "surface": "#FFFFFF",
    "surface_alt": "#F7F9F6",
    "surface_sunk": "#E4E9E3",
    "ink": "#17211B",
    "ink_soft": "#3D4C44",
    "ink_muted": "#6B7A71",
    "line": "#D2DAD2",
    "line_strong": "#B6C2B8",
    "accent": "#1F6B4A",
    "accent_hover": "#195A3E",
    "accent_soft": "#E2EFE7",
    "amber": "#A8601C",
    "amber_soft": "#F7EBDC",
    "berry": "#8C2F41",
    "berry_soft": "#F7E4E7",
    "slate": "#3F5C6B",
    "slate_soft": "#E6EDF0",
}

FONT_STACK = '"Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif'


def stylesheet() -> str:
    """Qt stylesheet. Qt supports a CSS subset, so no variables or shadows."""
    c = PALETTE
    return f"""
    QWidget {{
        background: {c['bg']};
        color: {c['ink']};
        font-family: {FONT_STACK};
        font-size: 14px;
    }}

    QLabel#pageTitle {{
        font-size: 25px;
        font-weight: 600;
        color: {c['ink']};
    }}
    QLabel#pageLead {{
        font-size: 14px;
        color: {c['ink_muted']};
    }}
    QLabel#sectionTitle {{
        font-size: 16px;
        font-weight: 600;
        padding-top: 4px;
    }}
    QLabel#hint {{
        color: {c['ink_muted']};
        font-size: 13px;
    }}
    QLabel#metricValue {{
        font-size: 30px;
        font-weight: 600;
        color: {c['ink']};
    }}
    QLabel#metricValueAccent {{
        font-size: 34px;
        font-weight: 700;
        color: {c['accent']};
    }}
    QLabel#metricLabel {{
        font-size: 13px;
        color: {c['ink_muted']};
    }}
    QLabel#metricNote {{
        font-size: 12px;
        color: {c['ink_muted']};
    }}

    QFrame#card {{
        background: {c['surface']};
        border: 1px solid {c['line']};
        border-radius: 10px;
    }}
    QFrame#metricCard {{
        background: {c['surface']};
        border: 1px solid {c['line']};
        border-radius: 10px;
    }}
    QFrame#metricCardLead {{
        background: {c['accent_soft']};
        border: 1px solid {c['accent']};
        border-radius: 10px;
    }}
    QFrame#banner {{
        background: {c['slate_soft']};
        border: 1px solid {c['line']};
        border-radius: 8px;
    }}
    QFrame#bannerWarn {{
        background: {c['amber_soft']};
        border: 1px solid {c['amber']};
        border-radius: 8px;
    }}
    QFrame#divider {{
        background: {c['line']};
        max-height: 1px;
        border: none;
    }}

    /* Sidebar navigation */
    QFrame#sidebar {{
        background: {c['surface']};
        border: none;
        border-right: 1px solid {c['line']};
    }}
    QLabel#brand {{
        font-size: 17px;
        font-weight: 700;
        color: {c['ink']};
        padding: 18px 18px 2px 18px;
    }}
    QLabel#brandSub {{
        font-size: 12px;
        color: {c['ink_muted']};
        padding: 0 18px 14px 18px;
    }}
    QPushButton#navButton {{
        background: transparent;
        border: none;
        border-left: 3px solid transparent;
        padding: 11px 16px;
        text-align: left;
        font-size: 14px;
        color: {c['ink_soft']};
    }}
    QPushButton#navButton:hover {{
        background: {c['surface_alt']};
        color: {c['ink']};
    }}
    QPushButton#navButton:checked {{
        background: {c['accent_soft']};
        border-left: 3px solid {c['accent']};
        color: {c['accent']};
        font-weight: 600;
    }}

    /* Buttons */
    QPushButton {{
        background: {c['surface']};
        border: 1px solid {c['line_strong']};
        border-radius: 7px;
        padding: 8px 15px;
        color: {c['ink']};
    }}
    QPushButton:hover {{ background: {c['surface_alt']}; }}
    QPushButton:disabled {{
        color: {c['ink_muted']};
        background: {c['surface_sunk']};
        border-color: {c['line']};
    }}
    QPushButton#primary {{
        background: {c['accent']};
        border: 1px solid {c['accent']};
        color: #FFFFFF;
        font-weight: 600;
        padding: 10px 20px;
    }}
    QPushButton#primary:hover {{ background: {c['accent_hover']}; }}
    QPushButton#primary:disabled {{
        background: {c['line_strong']};
        border-color: {c['line_strong']};
        color: {c['surface']};
    }}
    QPushButton#danger {{
        color: {c['berry']};
        border-color: {c['berry']};
    }}
    QPushButton#danger:hover {{ background: {c['berry_soft']}; }}

    /* Inputs */
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QDateEdit {{
        background: {c['surface']};
        border: 1px solid {c['line_strong']};
        border-radius: 7px;
        padding: 7px 9px;
        selection-background-color: {c['accent_soft']};
        selection-color: {c['ink']};
    }}
    QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus,
    QComboBox:focus, QDateEdit:focus {{
        border: 1px solid {c['accent']};
    }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QCheckBox {{ spacing: 8px; }}

    /* Slider: the service level control is the one thing people will drag */
    QSlider::groove:horizontal {{
        height: 6px;
        background: {c['surface_sunk']};
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {c['accent']};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {c['surface']};
        border: 2px solid {c['accent']};
        width: 17px;
        height: 17px;
        margin: -7px 0;
        border-radius: 10px;
    }}
    QSlider::handle:horizontal:hover {{ background: {c['accent_soft']}; }}

    /* Tables */
    QTableView, QTableWidget {{
        background: {c['surface']};
        alternate-background-color: {c['surface_alt']};
        border: 1px solid {c['line']};
        border-radius: 8px;
        gridline-color: {c['line']};
        selection-background-color: {c['accent_soft']};
        selection-color: {c['ink']};
    }}
    QTableView::item, QTableWidget::item {{ padding: 5px 7px; }}
    QHeaderView::section {{
        background: {c['surface_alt']};
        color: {c['ink_muted']};
        border: none;
        border-bottom: 1px solid {c['line_strong']};
        border-right: 1px solid {c['line']};
        padding: 8px 7px;
        font-weight: 600;
    }}
    QTableCornerButton::section {{
        background: {c['surface_alt']};
        border: none;
        border-bottom: 1px solid {c['line_strong']};
    }}

    QListWidget {{
        background: {c['surface']};
        border: 1px solid {c['line']};
        border-radius: 8px;
        padding: 4px;
    }}
    QListWidget::item {{ padding: 6px 8px; border-radius: 5px; }}
    QListWidget::item:selected {{ background: {c['accent_soft']}; color: {c['ink']}; }}

    QTextEdit, QPlainTextEdit {{
        background: {c['surface']};
        border: 1px solid {c['line']};
        border-radius: 8px;
        padding: 10px;
    }}

    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: transparent; width: 11px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {c['line_strong']};
        border-radius: 5px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {c['ink_muted']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar:horizontal {{
        background: transparent; height: 11px; margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c['line_strong']};
        border-radius: 5px;
        min-width: 28px;
    }}

    QProgressBar {{
        background: {c['surface_sunk']};
        border: none;
        border-radius: 4px;
        height: 7px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 4px; }}

    QWizard {{ background: {c['bg']}; }}
    QToolTip {{
        background: {c['ink']};
        color: #FFFFFF;
        border: none;
        padding: 6px 8px;
        border-radius: 5px;
    }}
    """


# Which flag words map to which signal colour, most severe first.
FLAG_COLOURS: list[tuple[str, str]] = [
    ("Held back for budget", "berry"),
    ("Short of forecast", "berry"),
    ("Stockout risk", "berry"),
    ("Order today", "amber"),
    ("Stock expiring", "amber"),
    ("Pack size forces waste", "amber"),
    ("Trimmed for budget", "amber"),
    ("drops", "slate"),
]


def flag_colour(flags: str) -> str | None:
    """The colour a row should carry, based on its worst flag."""
    for needle, key in FLAG_COLOURS:
        if needle in flags:
            return PALETTE[key]
    return None
