# Stock Planner

Tells a small kitchen what to buy for the next two weeks, what it will cost, and
what is about to spoil.

Built for small food businesses, where ordering is usually done from memory and
the two ways to get it wrong both cost money: order short and you lose sales,
order long and you bin produce.

## Running it

```bash
pip install -r requirements.txt
python run.py                 # opens the app
python run.py --demo          # loads two years of sample cafe data first
```

No data on first launch opens the setup wizard.

Other flags:

```bash
python run.py --data ./mydata            # a different data folder
python run.py --demo --headless          # print an order plan, no window
python run.py --headless --budget 2000   # see how the budget cap behaves
```

`--headless` runs the whole forecast in the terminal, which is useful for a
sanity check before a demo and for machines with no display.

## The three features

**1. CSV generator.** `ui/wizard.py`. Walks through items, recent sales and
current stock, then writes three CSVs. Sales entry asks for four weeks of weekly
totals rather than daily rows, because nobody will type two years of history by
hand. Four weeks is enough for an honest forecast on day one; the wizard says so
rather than pretending otherwise.

**2. The forecast.** `core/model.py`. Gradient boosting over engineered weekly
features, predicting the two-week sales total per item. It predicts a range, not
a single number, because the order decision needs a range. Under 60 weeks of
usable rows it falls back to a moving average and says which one it used.

**3. Import history.** `core/store.py` and `ui/import_page.py`. Point it at any
CSV export. Column names like `qty sold` or `spoilage` are recognised
automatically. Files are archived verbatim before merging, and merging is an
upsert on (date, item), so re-importing a corrected export fixes rows instead of
double counting.

## How the forecast works

Aggregate daily rows to weeks, then predict the next two weeks directly. One
two-week prediction is much steadier than fourteen daily ones added up, and it
matches how orders are actually placed.

Features, all computed from the current week backwards so nothing leaks:

| Signal | Features |
| --- | --- |
| Recent trend | sales lagged 1–4 weeks, rolling means over 4, 8 and 13 weeks |
| Recent vs three months | `trend_ratio` = 4-week mean ÷ 13-week mean |
| The year before | same week last year, plus the mean of the surrounding five weeks as "that month last year" |
| Seasonality | week of year as sine and cosine, month, holiday weeks ahead |
| Waste behaviour | waste rate and sell-through over the last four weeks |
| Item facts | shelf life, unit cost, category, item |

Five quantile models are fitted (10th to 90th percentile). The service level
slider picks a point on that range, interpolating between them.

Accuracy is measured by holding back the most recent ten weeks, never a random
split, and is reported against the three rules a manager would otherwise use:
repeat the last two weeks, copy the same weeks last year, or take a four-week
average.

On the sample data: **10.7% average error, about 10% better than the best of
those rules.** `ly_month_mean` lands in the top six features by permutation
importance, so the year-ago signal is genuinely being used rather than just
being available.

## From forecast to order

The forecast is not the order. `core/recommend.py` applies four constraints:

- **Stock on hand**, minus anything that will expire before it can be sold.
  Expiring stock is not counted as available, which is the difference between
  "you have 100" and "you can use 20 of the 100".
- **Shelf life.** The two-week total is what you budget for; the delivery
  quantity is what goes on the next order form, capped at what one drop can
  clear. Croissants come out as 240 units across 7 drops of 48 every 2 days, not
  one drop of 240.
- **Pack sizes and supplier minimums**, rounded up to whole cases. Where that
  forces more into a drop than its shelf life can clear, the line says so.
- **Budget**, filled greedily by value at risk. Cutting every line by a fifth
  guarantees running short on everything, so the lines worth least are dropped
  whole instead.

Waste is reported split two ways, because the two answer to different controls:
stock already committed, and waste caused by pack sizes.

## Layout

```
core/
  schema.py      three table schemas, header aliases, per-row validation
  store.py       CSV persistence, import archiving, upsert merging
  features.py    weekly aggregation, lag/seasonal/year-ago features, holidays
  model.py       quantile gradient boosting, backtest, moving-average fallback
  recommend.py   forecast to order sheet under expiry/pack/budget constraints
  synth.py       two years of simulated cafe data with FIFO spoilage
ui/
  theme.py         palette and Qt stylesheet
  widgets.py       DataFrame table model, metric cards, comparison chart
  wizard.py        the CSV generator
  import_page.py   import and validation
  forecast_page.py the order plan screen
  main_window.py   sidebar shell
tests/test_core.py  30 tests
run.py
```

Data lives in `data/` as plain CSVs (`catalog.csv`, `history.csv`,
`snapshot.csv`) so they open in Excel without the app. Imported files are kept
under `data/imports/`.

## Tests

```bash
python -m pytest tests -q
```

30 tests covering header aliasing, per-row error isolation, re-import upserts,
holiday date maths, no-future-leakage in features, the fallback path, quantile
ordering, model-beats-naive, shelf-life splitting, pack rounding, expiring stock
exclusion, and budget capping.

## Known gaps

- The model fits on the UI thread with a wait cursor. It takes a few seconds on
  two years of data. Moving it to a `QThread` is the obvious next step.
- The wizard's weekly sales entry spreads a weekly total evenly across seven
  days. The model re-aggregates to weeks immediately so nothing is lost, but the
  stored daily rows are flat within a week.
- Lead time is used to flag "order today" but does not yet shift the delivery
  schedule.
- Promotions are read from history but there is nowhere to tell the app about a
  promotion planned for next week, which is the single biggest available accuracy
  win left.
