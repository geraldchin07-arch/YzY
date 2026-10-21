"""Data and modeling helpers for CrystalCast.

Predict crystallization onset temperature (Tx, kelvin) for iron-based
metallic-glass alloys from composition descriptors only.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
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

DATA_URL = "https://publications.rwth-aachen.de/record/969330/files/Fe_MG_Database.xlsx"

ELEMENTS = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split())

@dataclass
class Evaluation:
    model_name: str
    mae: float
    rmse: float
    r2: float
    y_test: np.ndarray
    y_pred: np.ndarray


def clean_xy(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Coerce inputs to numeric and remove rows with unusable targets."""
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)
    x = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    x, y = x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if len(x) < 10:
        raise ValueError("At least 10 rows with a numeric crystallization-temperature target are required.")
    x = x.dropna(axis=1, how="all")
    if x.shape[1] == 0:
        raise ValueError("No usable composition features were found.")
    return x, y


def parse_composition(formula: Any) -> dict[str, float]:
    """Parse a compact alloy formula (for example Fe72B20Si8) into at.% features.

    Values are normalized to sum to 100. An omitted amount is interpreted as 1.
    This is a simple alloy-formula parser, not a general-purpose chemistry parser.
    """
    if pd.isna(formula):
        return {}
    text = re.sub(r"[\s,;_\-]+", "", str(formula))
    if not text:
        return {}
    matches = list(re.finditer(r"([A-Z][a-z]?)(\d+(?:\.\d*)?|\.\d+)?", text))
    if not matches or "".join(m.group(0) for m in matches) != text:
        return {}
    amounts: dict[str, float] = {}
    for match in matches:
        symbol = match.group(1)
        if symbol not in ELEMENTS:
            return {}
        amount = float(match.group(2)) if match.group(2) else 1.0
        if amount < 0:
            return {}
        amounts[symbol] = amounts.get(symbol, 0.0) + amount
    total = sum(amounts.values())
    if total <= 0:
        return {}
    return {element: 100.0 * amount / total for element, amount in amounts.items()}


def _element_symbol(column: Any) -> str | None:
    """Return an element symbol when a column appears to contain its at.% value."""
    label = str(column).strip()
    # Temperature shorthand can collide with real element symbols: Tl is thallium,
    # Tm is thulium, but Tl/Tm can also mean liquidus/melting temperature in datasets.
    normalized = re.sub(r"[^a-z0-9]", "", label.lower())
    if re.fullmatch(r"t(?:x|g|l|m)(?:k|c|degc|celsius)?", normalized):
        return None
    match = re.match(r"^([A-Z][a-z]?)(?:$|[\s_(\[/,%])", label)
    if match:
        symbol = match.group(1)
        return symbol if symbol in ELEMENTS else None
    return label if label in ELEMENTS else None


def composition_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract composition-only features, excluding measured property columns."""
    element_columns: dict[str, pd.Series] = {}
    for column in raw.columns:
        symbol = _element_symbol(column)
        if symbol is None:
            continue
        values = pd.to_numeric(raw[column], errors="coerce")
        if values.notna().sum() > 1:
            element_columns[symbol] = values
    if element_columns:
        features = pd.DataFrame(element_columns, index=raw.index)
        return features.loc[:, ~features.isna().all(axis=0)]

    preferred_names = {"composition", "alloy", "formula", "chemistry", "alloycomposition", "chemicalcomposition"}
    candidates = [c for c in raw.columns if re.sub(r"[^a-z]", "", str(c).lower()) in preferred_names]
    if not candidates:
        candidates = [c for c in raw.columns if raw[c].dtype == "object"]
    for column in candidates:
        parsed = [parse_composition(value) for value in raw[column]]
        if sum(bool(item) for item in parsed) >= 10:
            return pd.DataFrame(parsed, index=raw.index).fillna(0.0)
    raise ValueError(
        "Could not identify alloy composition columns. Expected element at.% columns "
        "or a composition/formula column. Available columns: " + ", ".join(map(str, raw.columns))
    )


def find_tx_column(raw: pd.DataFrame) -> Any:
    """Find the crystallization-onset temperature (Tx) target column."""
    for column in raw.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if (
            ("crystall" in normalized or "kristallisation" in normalized)
            and ("temp" in normalized or "tx" in normalized or "temperatur" in normalized or normalized in {"crystallization", "crystallisation"})
        ):
            return column
    for column in raw.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if normalized.startswith("tx"):
            return column
    raise ValueError(
        "Could not identify the crystallization temperature column (expected Tx or a column containing 'crystallization'). "
        "Available columns: " + ", ".join(map(str, raw.columns))
    )


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare composition-only features and the crystallization-onset target."""
    target_column = find_tx_column(raw)
    features = composition_features(raw)
    target = pd.to_numeric(raw[target_column], errors="coerce")
    # Normalize explicitly labelled Celsius targets to kelvin for consistent output.
    label = str(target_column)
    if re.search(r"(?:°\s*C\b|\bdeg(?:ree)?s?\s*C(?:elsius)?\b|\bCelsius\b|\[\s*C\s*\]|\(\s*C\s*\)|/\s*C\b)", label, flags=re.IGNORECASE):
        target = target + 273.15
    x, y = clean_xy(features, target)
    x = x.dropna(axis=1, how="all")
    x.columns = [str(column) for column in x.columns]
    return x, y


def load_crystallization_dataset(url: str = DATA_URL) -> tuple[pd.DataFrame, pd.Series]:
    """Load a local copy if provided, otherwise download the open RWTH Excel dataset."""
    local_path = Path("data/Fe_MG_Database.xlsx")
    try:
        raw = pd.read_excel(local_path if local_path.exists() else url)
    except Exception as exc:
        raise RuntimeError(
            "Could not download/read the RWTH metallic-glass dataset. Check internet access "
            "and make sure openpyxl is installed. You can also download the XLSX from " + DATA_URL +
            " and place it at data/Fe_MG_Database.xlsx."
        ) from exc
    return prepare_dataset(raw)


def build_model(model_name: str, random_state: int = 42) -> Any:
    """Build a supported regressor; baseline is calculated separately."""
    if model_name == "Transformer Neural Network":
        # Import lazily so classical models remain independent of PyTorch startup.
        from .transformer_model import CompositionTransformerRegressor
        return CompositionTransformerRegressor(random_state=random_state)
    if model_name == "Median Baseline":
        return None
    if model_name == "Ridge Regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if model_name == "Extra Trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=160, min_samples_leaf=2, random_state=random_state, n_jobs=-1),
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Evaluation:
    """Evaluate a regressor on a reproducible held-out test split."""
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


def predict_tx(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    formula: str,
    random_state: int = 42,
) -> tuple[float, list[str]]:
    """Fit on all labeled rows and predict Tx for a user-provided alloy formula.

    This is for exploratory screening only; it is not experimental validation.
    """
    composition = parse_composition(formula)
    if not composition:
        raise ValueError("Enter a compact alloy formula such as Fe72B20Si8, with element symbols and amounts.")
    known_elements = set(map(str, x.columns))
    unknown = sorted(set(composition) - known_elements)
    if unknown:
        raise ValueError("The dataset has no composition features for: " + ", ".join(unknown) + ". Use only elements represented by this dataset.")
    input_row = pd.DataFrame([{element: composition.get(element, 0.0) for element in x.columns}], columns=x.columns)
    model = build_model(model_name, random_state=random_state)
    if model is None:
        return float(pd.to_numeric(y, errors="coerce").median()), sorted(composition)
    model.fit(x, y)
    prediction = float(model.predict(input_row)[0])
    if not np.isfinite(prediction):
        raise ValueError("The model produced a non-finite prediction.")
    return prediction, sorted(composition)
