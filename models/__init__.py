from sklearn.linear_model import LinearRegression

MODEL_REGISTRY = {
    "linear_regression": LinearRegression,
}


def get_model(name, params):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](**params)
