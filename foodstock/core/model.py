"""Two-week demand forecasting.

The model predicts a range, not a single number, because the order decision needs
a range. Ordering to the median means running out half the time; ordering to the
90th percentile means padding stock. Which one is right depends on whether the
item spoils in two days or two years, and on how much the owner hates empty
shelves, so the app exposes that choice as a service level and the model supplies
the matching quantile.

Gradient boosting on engineered features is used rather than anything heavier:
this is small tabular data with strong seasonal structure, it trains in under a
second, and feature importances give the UI something honest to show.

If there is too little history to learn from, the class falls back to a
moving-average forecast and says so, rather than fitting noise and looking
confident about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from foodstock.core import features as F

QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
MIN_TRAINING_ROWS = 60
MIN_WEEKS_PER_SKU = 8


@dataclass
class Scorecard:
    """Backtest accuracy against the baselines a manager would otherwise use."""

    weeks_tested: int
    rows_tested: int
    model_wape: float
    baselines: dict[str, float] = field(default_factory=dict)

    @property
    def best_baseline(self) -> tuple[str, float]:
        if not self.baselines:
            return ("none", float("nan"))
        name = min(self.baselines, key=lambda k: self.baselines[k])
        return (name, self.baselines[name])

    @property
    def improvement(self) -> float:
        """Percent reduction in error against the best baseline."""
        _, best = self.best_baseline
        if not np.isfinite(best) or best == 0:
            return float("nan")
        return 100.0 * (best - self.model_wape) / best

    def describe(self) -> str:
        if self.rows_tested == 0:
            return "Not enough history to measure accuracy yet."
        name, best = self.best_baseline
        lead = f"Average error {self.model_wape:.1f}% over {self.weeks_tested} held-back weeks"
        if not np.isfinite(best):
            return lead + "."
        gap = self.improvement
        verb = "better than" if gap > 0 else "worse than"
        return (f"{lead}, {abs(gap):.0f}% {verb} the best simple rule "
                f"({BASELINE_LABELS.get(name, name)} at {best:.1f}%).")


BASELINE_LABELS = {
    "repeat_last_2w": "repeat the last two weeks",
    "same_weeks_last_year": "copy the same weeks last year",
    "moving_average_4w": "four week moving average",
}


@dataclass
class Forecast:
    """Per-item two-week demand prediction."""

    frame: pd.DataFrame          # sku, week, p10..p90, expected
    as_of_week: pd.Timestamp
    method: str                  # "gradient_boosting" or "moving_average"
    scorecard: Scorecard
    importances: pd.Series | None = None
    notes: list[str] = field(default_factory=list)

    def at_service_level(self, service_level: float) -> pd.Series:
        """Units to plan for at a given service level, interpolated across quantiles."""
        level = float(np.clip(service_level, QUANTILES[0], QUANTILES[-1]))
        cols = [f"p{int(q * 100)}" for q in QUANTILES]
        values = self.frame[cols].to_numpy()
        planned = np.array([np.interp(level, QUANTILES, row) for row in values])
        return pd.Series(planned, index=self.frame.index, name="planned_units")


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error.

    Preferred over MAPE here because low-volume items would otherwise dominate:
    being one croissant off on a week that sold two is a 50% error but costs
    almost nothing.
    """
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return 100.0 * np.abs(actual - predicted).sum() / denominator


def _baseline_predictions(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """What a manager with a spreadsheet would guess."""
    last_2w = (table["lag_1w"].fillna(0) + table["lag_2w"].fillna(table["lag_1w"]).fillna(0))
    ly = (table["ly_month_mean"] * F.HORIZON_WEEKS)
    ma = table["roll4_mean"].fillna(0) * F.HORIZON_WEEKS
    return {
        "repeat_last_2w": last_2w.to_numpy(dtype=float),
        "same_weeks_last_year": ly.fillna(ma).to_numpy(dtype=float),
        "moving_average_4w": ma.to_numpy(dtype=float),
    }


class DemandForecaster:
    """Fits quantile models over the weekly feature table."""

    def __init__(self, *, holdout_weeks: int = 10, random_state: int = 0):
        self.holdout_weeks = holdout_weeks
        self.random_state = random_state
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.method = "untrained"
        self.scorecard = Scorecard(0, 0, float("nan"))
        self.importances: pd.Series | None = None
        self.notes: list[str] = []
        self._categories: dict[str, pd.Index] = {}

    # ---------------------------------------------------------------- fitting

    def _make_estimator(self, quantile: float) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=300,
            learning_rate=0.06,
            max_depth=6,
            min_samples_leaf=12,
            l2_regularization=1.0,
            early_stopping=False,
            categorical_features="from_dtype",
            random_state=self.random_state,
        )

    def _design(self, table: pd.DataFrame) -> pd.DataFrame:
        design = table[F.FEATURE_COLUMNS].copy()
        for col in F.CATEGORICAL:
            if col in self._categories:
                design[col] = pd.Categorical(
                    design[col].astype(str), categories=self._categories[col])
            else:
                design[col] = design[col].astype(str).astype("category")
                self._categories[col] = design[col].cat.categories
        return design

    def fit(self, history: pd.DataFrame, catalog: pd.DataFrame) -> "DemandForecaster":
        table = F.build_features(history, catalog, with_target=True)
        train = F.training_rows(table)

        weeks_per_sku = train.groupby("sku", observed=True).size()
        enough = (len(train) >= MIN_TRAINING_ROWS
                  and (weeks_per_sku >= MIN_WEEKS_PER_SKU).any())

        if not enough:
            self.method = "moving_average"
            self.notes.append(
                f"Only {len(train)} usable weeks of history, so the forecast uses a "
                "four week average. Import more history to switch on the learned model.")
            self.scorecard = self._score_baselines_only(train)
            return self

        self.method = "gradient_boosting"
        self._categories = {}
        design = self._design(train)
        target = train[F.TARGET].to_numpy(dtype=float)

        self.scorecard = self._backtest(train)

        for q in QUANTILES:
            estimator = self._make_estimator(q)
            estimator.fit(design, target)
            self.models[q] = estimator

        self.importances = self._permutation_importance(design, target)
        if self.scorecard.rows_tested and self.scorecard.improvement < 0:
            self.notes.append(
                "The learned model did not beat the simple rules on held-back weeks. "
                "Treat its numbers as a second opinion until there is more history.")
        return self

    # -------------------------------------------------------------- appraisal

    def _score_baselines_only(self, train: pd.DataFrame) -> Scorecard:
        if train.empty:
            return Scorecard(0, 0, float("nan"))
        actual = train[F.TARGET].to_numpy(dtype=float)
        baselines = {k: wape(actual, v) for k, v in _baseline_predictions(train).items()}
        return Scorecard(0, len(train), baselines.get("moving_average_4w", float("nan")),
                         baselines)

    def _backtest(self, train: pd.DataFrame) -> Scorecard:
        """Hold back the most recent weeks, never a random sample.

        A random split would let the model see next month while predicting this
        one, which inflates accuracy on any seasonal series.
        """
        weeks = np.sort(train["week"].unique())
        if len(weeks) <= self.holdout_weeks + 12:
            return self._score_baselines_only(train)

        cutoff = weeks[-self.holdout_weeks - 1]
        fit_part = train[train["week"] <= cutoff]
        test_part = train[train["week"] > cutoff]
        if fit_part.empty or test_part.empty:
            return self._score_baselines_only(train)

        saved = dict(self._categories)
        self._categories = {}
        estimator = self._make_estimator(0.5)
        estimator.fit(self._design(fit_part), fit_part[F.TARGET].to_numpy(dtype=float))
        predicted = np.clip(estimator.predict(self._design(test_part)), 0, None)
        self._categories = saved

        actual = test_part[F.TARGET].to_numpy(dtype=float)
        baselines = {k: wape(actual, v) for k, v in _baseline_predictions(test_part).items()}
        return Scorecard(int(test_part["week"].nunique()), len(test_part),
                         wape(actual, predicted), baselines)

    def _permutation_importance(self, design: pd.DataFrame, target: np.ndarray,
                                repeats: int = 2) -> pd.Series:
        """Shuffle one feature at a time and measure the damage.

        Used instead of split-count importance because it reflects predictive
        value rather than how many ways a column can be cut.
        """
        model = self.models.get(0.5)
        if model is None:
            return pd.Series(dtype=float)
        rng = np.random.default_rng(self.random_state)
        base = wape(target, np.clip(model.predict(design), 0, None))
        scores: dict[str, float] = {}
        for column in design.columns:
            losses = []
            for _ in range(repeats):
                shuffled = design.copy()
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                losses.append(wape(target, np.clip(model.predict(shuffled), 0, None)) - base)
            scores[column] = float(np.mean(losses))
        return pd.Series(scores).sort_values(ascending=False)

    # ------------------------------------------------------------ prediction

    def predict(self, history: pd.DataFrame, catalog: pd.DataFrame,
                as_of: pd.Timestamp | None = None) -> Forecast:
        table = F.build_features(history, catalog, with_target=False)
        rows = F.latest_rows(table, as_of=as_of)
        if rows.empty:
            raise ValueError("No history available to forecast from.")

        as_of_week = pd.Timestamp(rows["week"].max())
        out = rows[["sku", "week"]].copy()

        if self.method == "gradient_boosting" and self.models:
            design = self._design(rows)
            for q in QUANTILES:
                out[f"p{int(q * 100)}"] = np.clip(self.models[q].predict(design), 0, None)
            # Quantile models are fitted independently and can cross over on
            # sparse items; sorting each row restores a usable interval.
            cols = [f"p{int(q * 100)}" for q in QUANTILES]
            out[cols] = np.sort(out[cols].to_numpy(), axis=1)
        else:
            centre = (rows["roll4_mean"].fillna(rows["lag_1w"]).fillna(0)
                      * F.HORIZON_WEEKS).to_numpy(dtype=float)
            spread = (rows["roll4_std"].fillna(rows["roll4_mean"] * 0.35).fillna(1.0)
                      * np.sqrt(F.HORIZON_WEEKS)).to_numpy(dtype=float)
            # Normal approximation is crude but adequate for a fallback path.
            z = {0.1: -1.2816, 0.3: -0.5244, 0.5: 0.0, 0.7: 0.5244, 0.9: 1.2816}
            for q in QUANTILES:
                out[f"p{int(q * 100)}"] = np.clip(centre + z[q] * spread, 0, None)

        out["expected"] = out["p50"]
        out["recent_2w"] = (rows["lag_1w"].fillna(0) + rows["lag_2w"].fillna(0)).to_numpy()
        out["last_year_2w"] = (rows["ly_month_mean"] * F.HORIZON_WEEKS).to_numpy()
        out["trend_ratio"] = rows["trend_ratio"].to_numpy()
        out["holiday_weeks_ahead"] = rows["holiday_weeks_ahead"].to_numpy()

        return Forecast(out.reset_index(drop=True), as_of_week, self.method,
                        self.scorecard, self.importances, list(self.notes))


def fit_and_forecast(history: pd.DataFrame, catalog: pd.DataFrame,
                     as_of: pd.Timestamp | None = None) -> Forecast:
    """Convenience path used by the UI."""
    return DemandForecaster().fit(history, catalog).predict(history, catalog, as_of=as_of)
