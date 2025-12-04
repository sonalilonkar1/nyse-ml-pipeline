"""Model registry for experiment configs."""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from .lstm import LSTMRegressor


def _create_polynomial_regression(degree=2, alpha=None, **kwargs):
    """Factory function to create a polynomial regression pipeline.
    
    Args:
        degree: Polynomial degree (default: 2)
        alpha: Regularization strength for Ridge. If None, uses LinearRegression.
               If provided, uses Ridge with the specified alpha.
        **kwargs: Additional arguments passed to the regressor
    """
    # If alpha is provided, use Ridge regression; otherwise use LinearRegression
    if alpha is not None:
        regressor = Ridge(alpha=alpha, **kwargs)
    else:
        regressor = LinearRegression(**kwargs)
    
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("regressor", regressor),
        ]
    )


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


def _create_random_forest_regressor(**kwargs):
    """Factory function to create a Random Forest regressor."""
    return RandomForestRegressor(**kwargs)


def _create_gradient_boosting_regressor(**kwargs):
    """Factory function to create a Gradient Boosting regressor."""
    return GradientBoostingRegressor(**kwargs)


def _create_mlp_regressor(**kwargs):
    """Factory function to create an MLP regressor."""
    return MLPRegressor(**kwargs)


MODEL_REGISTRY = {
    "linear_regression": LinearRegression,
    "ridge_regression": Ridge,
    "polynomial_regression": _create_polynomial_regression,
    "random_forest_regressor": _create_random_forest_regressor,
    "gradient_boosting_regressor": _create_gradient_boosting_regressor,
    "mlp_regressor": _create_mlp_regressor,
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
