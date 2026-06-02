from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_label, get_feature_columns


@dataclass
class TrainingResult:
    model: Pipeline
    feature_columns: list[str]
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    model_comparison: pd.DataFrame
    feature_importance: pd.DataFrame


def build_classifier(model_type: str, random_state: int) -> Any:
    normalized = model_type.lower().strip()
    if normalized == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if normalized == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if normalized == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=30,
            random_state=random_state,
        )
    if normalized == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    if normalized == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    if normalized == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def train_model(featured_df: pd.DataFrame, config: dict[str, Any]) -> TrainingResult:
    model_config = config["model"]
    feature_columns = get_feature_columns(featured_df)
    if not feature_columns:
        raise ValueError("No numeric feature columns found")

    training = featured_df.dropna(subset=["target_up"]).copy()
    training = training.sort_values(["date", "ticker"]).reset_index(drop=True)
    if training["target_up"].nunique() < 2:
        raise ValueError("Training target only has one class")

    split_date = _split_date(training["date"], float(model_config["test_size"]))
    train_mask = training["date"] <= split_date
    test_mask = training["date"] > split_date
    if not test_mask.any():
        raise ValueError("Not enough data for a time-based test split")

    y_true = training.loc[test_mask, "target_up"].astype(int)
    x_train = training.loc[train_mask, feature_columns]
    y_train = training.loc[train_mask, "target_up"].astype(int)
    x_test = training.loc[test_mask, feature_columns]

    candidate_names = _model_candidates(model_config)
    selection_metric = str(model_config.get("selection_metric", "roc_auc")).lower()
    comparison_rows: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_model_name = ""
    best_metrics: dict[str, float] = {}
    best_probabilities: np.ndarray | None = None
    best_score = float("-inf")

    for candidate_name in candidate_names:
        try:
            candidate_model = _build_pipeline(candidate_name, int(model_config["random_state"]))
            candidate_model.fit(x_train, y_train)
            probabilities = candidate_model.predict_proba(x_test)[:, 1]
            metrics = _classification_metrics(y_true, probabilities)
            selection_score = _selection_score(metrics, selection_metric)
            comparison_rows.append(
                {
                    "model": candidate_name,
                    "status": "success",
                    "selected": False,
                    "accuracy": metrics["accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "selection_score": selection_score,
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    "feature_count": len(feature_columns),
                    "error": "",
                }
            )
            if selection_score > best_score:
                best_score = selection_score
                best_model = candidate_model
                best_model_name = candidate_name
                best_metrics = metrics
                best_probabilities = probabilities
        except Exception as error:
            comparison_rows.append(
                {
                    "model": candidate_name,
                    "status": "failed",
                    "selected": False,
                    "accuracy": np.nan,
                    "roc_auc": np.nan,
                    "selection_score": np.nan,
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    "feature_count": len(feature_columns),
                    "error": str(error),
                }
            )

    if best_model is None or best_probabilities is None:
        errors = "; ".join(f"{row['model']}: {row['error']}" for row in comparison_rows)
        raise ValueError(f"All candidate models failed: {errors}")

    model_comparison = pd.DataFrame(comparison_rows)
    model_comparison.loc[model_comparison["model"].eq(best_model_name), "selected"] = True
    model_comparison = _sort_model_comparison(model_comparison)

    labels = (best_probabilities >= 0.5).astype(int)
    predictions = training.loc[test_mask, ["date", "ticker", "close", "future_return", "target_up"]].copy()
    predictions["prob_up"] = best_probabilities
    predictions["prob_down"] = 1 - best_probabilities
    predictions["predicted_up"] = labels
    metrics = {"selected_model": best_model_name, **best_metrics}
    feature_importance = extract_feature_importance(best_model, feature_columns)
    return TrainingResult(
        model=best_model,
        feature_columns=feature_columns,
        metrics=metrics,
        predictions=predictions,
        model_comparison=model_comparison,
        feature_importance=feature_importance,
    )


def predict_signals(
    model: Pipeline,
    featured_df: pd.DataFrame,
    feature_columns: list[str],
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    usable = featured_df.dropna(subset=feature_columns, how="all").copy()
    probabilities = model.predict_proba(usable[feature_columns])[:, 1]
    usable["prob_up"] = probabilities
    usable["prob_down"] = 1 - probabilities
    usable["ml_signal"] = np.select(
        [usable["prob_up"] >= buy_threshold, usable["prob_up"] <= sell_threshold],
        ["BUY", "SELL"],
        default="HOLD",
    )
    columns = [
        "date",
        "ticker",
        "stock_name",
        "stock_group",
        "close",
        "prob_up",
        "prob_down",
        "ml_signal",
        "future_return",
        "future_high_max",
        "future_low_min",
        "daily_return",
        "atr_14",
    ]
    return usable[[column for column in columns if column in usable.columns]].sort_values(["date", "ticker"])


def latest_signals(signal_history: pd.DataFrame) -> pd.DataFrame:
    latest_dates = signal_history.groupby("ticker")["date"].transform("max")
    return signal_history[signal_history["date"].eq(latest_dates)].sort_values(["ml_signal", "prob_up"], ascending=[True, False])


def save_model(result: TrainingResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "feature_columns": result.feature_columns,
            "metrics": result.metrics,
            "model_comparison": result.model_comparison,
            "feature_importance": result.feature_importance,
        },
        output_path,
    )


def load_model(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def extract_feature_importance(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        raw_importance = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        raw_importance = np.abs(np.asarray(classifier.coef_, dtype=float)).mean(axis=0)
    else:
        return pd.DataFrame(columns=["rank", "feature", "feature_label", "importance", "importance_pct"])

    if raw_importance.size != len(feature_columns):
        return pd.DataFrame(columns=["rank", "feature", "feature_label", "importance", "importance_pct"])

    total = raw_importance.sum()
    importance = pd.DataFrame({"feature": feature_columns, "importance": raw_importance})
    importance["feature_label"] = importance["feature"].map(feature_label)
    importance["importance_pct"] = 0.0 if total == 0 else importance["importance"] / total
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    return importance[["rank", "feature", "feature_label", "importance", "importance_pct"]]


def _build_pipeline(model_type: str, random_state: int) -> Pipeline:
    classifier = build_classifier(model_type, random_state)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def _model_candidates(model_config: dict[str, Any]) -> list[str]:
    candidates = model_config.get("candidates")
    if not candidates:
        candidates = [model_config["type"]]
    names = []
    for candidate in candidates:
        normalized = str(candidate).strip().lower()
        if normalized and normalized not in names:
            names.append(normalized)
    return names


def _classification_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    labels = (probabilities >= 0.5).astype(int)
    metrics = {"accuracy": float(accuracy_score(y_true, labels))}
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    preferred = metrics.get(selection_metric, float("nan"))
    if not pd.isna(preferred):
        return float(preferred)
    fallback = metrics.get("accuracy", float("nan"))
    if not pd.isna(fallback):
        return float(fallback)
    return float("-inf")


def _sort_model_comparison(model_comparison: pd.DataFrame) -> pd.DataFrame:
    ordered = model_comparison.copy()
    ordered["status_order"] = ordered["status"].map({"success": 0, "failed": 1}).fillna(2)
    ordered = ordered.sort_values(
        ["selected", "status_order", "selection_score"],
        ascending=[False, True, False],
        na_position="last",
    )
    return ordered.drop(columns=["status_order"]).reset_index(drop=True)


def _split_date(dates: pd.Series, test_size: float) -> pd.Timestamp:
    unique_dates = pd.Series(pd.to_datetime(dates).sort_values().unique())
    split_index = max(0, min(len(unique_dates) - 2, int(len(unique_dates) * (1 - test_size))))
    return pd.Timestamp(unique_dates.iloc[split_index])
