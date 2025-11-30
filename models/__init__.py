"""Model registry for experiment configs."""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from .lstm import LSTMRegressor


def _create_polynomial_regression(degree=2, **kwargs):
    """Factory function to create a polynomial regression pipeline."""
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression(**kwargs))
    ])


def _create_xgboost_regressor(**kwargs):
    try:  # Local import to keep dependency optional until needed
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "xgboost is required for the 'xgboost_regressor'. Install it via pip install xgboost."
        ) from exc

    try:
        return XGBRegressor(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize XGBRegressor. Ensure system libraries such as libomp are installed."
        ) from exc


MODEL_REGISTRY = {
    "linear_regression": LinearRegression,
    "polynomial_regression": _create_polynomial_regression,
    "xgboost_regressor": _create_xgboost_regressor,
    "lstm_regressor": LSTMRegressor,
}


def get_model(name, params):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    model_class = MODEL_REGISTRY[name]
    # Handle callable factory functions vs direct classes
    if callable(model_class) and not isinstance(model_class, type):
        return model_class(**params)
    return model_class(**params)
