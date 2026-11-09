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


def suite_baselines(task: str, random_state: int) -> dict[str, BaseEstimator]:
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from xgboost import XGBClassifier, XGBRegressor

    if task == "classification":
        return {
            "Logistic Regression": make_pipeline(
                StandardScaler(), LogisticRegression(random_state=random_state, max_iter=5000)),
            "Random Forest": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=random_state),
            "Extra Trees": ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=random_state),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
            "XGBoost": XGBClassifier(random_state=random_state, n_jobs=-1, verbosity=0),
            "LightGBM": LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1),
            "CatBoost": CatBoostClassifier(
                random_seed=random_state, thread_count=-1, verbose=0, allow_writing_files=False),
        }
    if task == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=random_state),
            "Extra Trees": ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=random_state),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
            "XGBoost": XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0),
            "LightGBM": LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1),
            "CatBoost": CatBoostRegressor(
                random_seed=random_state, thread_count=-1, verbose=0, allow_writing_files=False),
        }
    raise ValueError(f"Unsupported task '{task}'")
