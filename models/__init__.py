from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def _create_polynomial_regression(degree=2, **kwargs):
    """Factory function to create a polynomial regression pipeline."""
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression(**kwargs))
    ])


MODEL_REGISTRY = {
    "linear_regression": LinearRegression,
    "polynomial_regression": _create_polynomial_regression,
}


def get_model(name, params):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    model_class = MODEL_REGISTRY[name]
    # Handle callable factory functions vs direct classes
    if callable(model_class) and not isinstance(model_class, type):
        return model_class(**params)
    return model_class(**params)
