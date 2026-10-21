"""Reusable data and modeling helpers for MatSci Lens."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class Evaluation:
    model_name: str
    mae: float
    rmse: float
    r2: float
    y_test: np.ndarray
    y_pred: np.ndarray

def clean_xy(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Coerce inputs to numeric, discard invalid targets and infinite values."""
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)
    x = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    x, y = x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if len(x) < 10:
        raise ValueError("At least 10 rows with a numeric target are required.")
    if x.shape[1] == 0:
        raise ValueError("The dataset has no feature columns.")
    x = x.dropna(axis=1, how="all")
    if x.shape[1] == 0:
        raise ValueError("All feature columns are empty or non-numeric.")
    return x, y

def load_uci_superconductivity() -> tuple[pd.DataFrame, pd.Series]:
    """Download the UCI Superconductivity dataset using ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("Install project requirements first: pip install -r requirements.txt") from exc
    dataset = fetch_ucirepo(id=464)
    features, targets = dataset.data.features, dataset.data.targets
    if features is None or targets is None or len(targets.columns) == 0:
        raise RuntimeError("UCI returned data without both features and a target.")
    return clean_xy(features, targets.iloc[:, 0])

def build_model(model_name: str, random_state: int = 42) -> Any:
    """Create a supported regressor; baseline is handled without fitting."""
    if model_name == "Median Baseline":
        return None
    if model_name == "Ridge Regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if model_name == "Extra Trees":
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2, random_state=random_state, n_jobs=-1))
    raise ValueError(f"Unknown model: {model_name}")

def evaluate_model(x: pd.DataFrame, y: pd.Series, model_name: str, random_state: int = 42, test_size: float = 0.2) -> Evaluation:
    """Evaluate with a reproducible hold-out split."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    x, y = clean_xy(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    model = build_model(model_name, random_state=random_state)
    if model is None:
        prediction = np.full(len(y_test), float(y_train.median()))
    else:
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_test), dtype=float)
    truth = np.asarray(y_test, dtype=float)
    mae = float(mean_absolute_error(truth, prediction))
    rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
    r2 = float(r2_score(truth, prediction))
    if not np.isfinite([mae, rmse, r2]).all():
        raise ValueError("Evaluation produced a non-finite metric.")
    return Evaluation(model_name, mae, rmse, r2, truth, prediction)
