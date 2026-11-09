from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def classification_baselines(random_state: int, params: dict | None = None) -> dict[str, BaseEstimator]:
    params = params or {}
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=random_state, **params.get("logistic_regression", {})),
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state, **params.get("random_forest", {})
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=random_state, **params.get("hist_gradient_boosting", {})
        ),
    }


def regression_baselines(random_state: int, params: dict | None = None) -> dict[str, BaseEstimator]:
    params = params or {}
    return {
        "Linear Regression": LinearRegression(**params.get("linear_regression", {})),
        "Random Forest": RandomForestRegressor(
            random_state=random_state, **params.get("random_forest", {})
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=random_state, **params.get("hist_gradient_boosting", {})
        ),
    }


def build_baselines(task: str, random_state: int, params: dict | None = None) -> dict[str, BaseEstimator]:
    if task == "classification":
        return classification_baselines(random_state, params)
    if task == "regression":
        return regression_baselines(random_state, params)
    raise ValueError(f"Unsupported task '{task}'")
