"""
Minimal spline feature expansion utilities.

- Uses sklearn.preprocessing.SplineTransformer (well tested).
- Primary function: add_spline_features(X, columns, n_knots=5, degree=3, include_bias=False,
  transformers=None)

Returns:
    X_new (pd.DataFrame): DataFrame with the selected columns replaced by their spline basis columns
    transformers (dict): dict mapping column -> fitted SplineTransformer (so the same
                         configuration can be applied to test data)

The function fits one SplineTransformer per column for clarity and stable naming.
"""

from typing import List, Tuple, Dict, Optional
import pandas as pd
from sklearn.preprocessing import SplineTransformer


def add_spline_features(X: pd.DataFrame,
                        columns: List[str],
                        n_knots: int = 5,
                        degree: int = 3,
                        include_bias: bool = False,
                        transformers: Optional[Dict[str, SplineTransformer]] = None
                        ) -> Tuple[pd.DataFrame, Dict[str, SplineTransformer]]:
    """Replace selected columns with spline basis features.

    Args:
        X: input DataFrame
        columns: list of column names to expand (must exist in X)
        n_knots: number of interior knots (passed to SplineTransformer)
        degree: spline degree (default cubic)
        include_bias: whether to include bias term in the basis
        transformers: if provided, a dict col->fitted SplineTransformer to use for transform.
                      If None, new transformers are fitted on X[columns].

    Returns:
        (X_new, transformers): X_new has original columns dropped and new columns appended
                               with names like "{col}_spline_0" ...
    """
    X_new = X.copy()
    fitted = {} if transformers is None else dict(transformers)

    for col in columns:
        if col not in X_new.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

        # Fit transformer per column if not provided
        if col not in fitted:
            tr = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=include_bias)
            tr.fit(X_new[[col]].values)
            fitted[col] = tr
        else:
            tr = fitted[col]

        # Transform and create named columns
        out = tr.transform(X_new[[col]].values)
        n_out = out.shape[1]
        names = [f"{col}_spline_{i}" for i in range(n_out)]
        df_out = pd.DataFrame(out, columns=names, index=X_new.index)

        # Drop original and append new features
        X_new = X_new.drop(columns=[col])
        X_new = pd.concat([X_new, df_out], axis=1)

    return X_new, fitted


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np
    df = pd.DataFrame({
        'Temp': np.linspace(0, 10, 11),
        'toy': np.linspace(0, 1, 11),
        'A': [0, 1] * 5 + [0]
    })
    Xs, tf = add_spline_features(df, ['Temp', 'toy'], n_knots=4)
    print('Original cols:', df.columns.tolist())
    print('Expanded cols:', Xs.columns.tolist())
    print('Transformers keys:', list(tf.keys()))
