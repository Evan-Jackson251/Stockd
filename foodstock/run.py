#!/usr/bin/env python3
"""Start the app.

    python run.py                     open the desktop app
    python run.py --data ./mydata     use a different data folder
    python run.py --demo              load sample data first, then open
    python run.py --headless          run a forecast in the terminal and exit

The headless mode exists so the forecasting core can be checked on a machine
with no display, which is also handy for a quick sanity check before a demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import recommend, synth
from core.model import DemandForecaster
from core.store import DataStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two week stock ordering")
    parser.add_argument("--data", default="data", help="folder for the CSV files")
    parser.add_argument("--demo", action="store_true",
                        help="load two years of sample cafe data first")
    parser.add_argument("--headless", action="store_true",
                        help="print an order plan and exit, no window")
    parser.add_argument("--service-level", type=float, default=0.6,
                        help="0.1 lean to 0.9 never run out")
    parser.add_argument("--budget", type=float, default=None, help="cap on total spend")
    return parser.parse_args()


def headless(store: DataStore, service_level: float, budget: float | None) -> int:
    catalog = store.load("catalog")
    history = store.load("history")
    snapshot = store.load("snapshot")

    if catalog.empty or history.empty:
        print("No data found. Run again with --demo to load sample data.")
        return 1

    print(store.coverage().describe())
    print("Fitting the model…")
    forecaster = DemandForecaster().fit(history, catalog)
    forecast = forecaster.predict(history, catalog)

    print(f"Method: {forecast.method}")
    print(forecast.scorecard.describe())
    for note in forecast.notes:
        print(f"Note: {note}")

    plan = recommend.build_plan(forecast, catalog, snapshot,
                                service_level=service_level, budget=budget)
    summary = plan.summary()
    label, blurb = recommend.service_level_label(service_level)

    print(f"\nShelf target: {label} — {blurb}")
    print(f"Items to order: {summary['items_ordered']}")
    print(f"Estimated cost: ${summary['total_cost']:,.2f}")
    print(f"First delivery: ${summary['first_delivery_cost']:,.2f}")
    print(f"Likely waste: {summary['projected_waste_units']:,.0f} units "
          f"(${summary['projected_waste_cost']:,.2f})")

    print("\nOrder sheet")
    sheet = plan.order_sheet()
    for row in sheet.itertuples():
        drops = (f"  ({row.deliveries} drops of {row.delivery_qty:,.0f} every "
                 f"{row.reorder_every_days}d)" if row.deliveries > 1 else "")
        print(f"  {row.item_name:<22} {row.horizon_qty:>7,.0f} {row.unit:<6} "
              f"${row.horizon_cost:>9,.2f}{drops}")

    flagged = plan.flagged
    if not flagged.empty:
        print("\nWatch out for")
        for row in flagged.itertuples():
            print(f"  {row.item_name:<22} {row.flags}")
    return 0


def main() -> int:
    args = parse_args()
    store = DataStore(args.data)

    if args.demo:
        counts = synth.write_demo_data(store)
        print(f"Loaded sample data: {counts['history']:,} sales rows "
              f"across {counts['catalog']} items.")

    if args.headless:
        return headless(store, args.service_level, args.budget)

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Install it with:\n\n"
              "    pip install -r requirements.txt\n\n"
              "Or run the forecast without a window using --headless.")
        return 1

    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Stock Planner")

    window = MainWindow(store)
    window.show()
    window.maybe_first_run()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
