from sklearn.preprocessing import FunctionTransformer


def binary_homeless(X):
    """Convert homeless percentage to binary (1 if > 0, else 0)."""
    return (X > 0).astype(int)


def get_binary_homeless_transformer():
    """Return a FunctionTransformer for binary_homeless."""
    return FunctionTransformer(
        func=binary_homeless,
        feature_names_out="one-to-one",
    )
