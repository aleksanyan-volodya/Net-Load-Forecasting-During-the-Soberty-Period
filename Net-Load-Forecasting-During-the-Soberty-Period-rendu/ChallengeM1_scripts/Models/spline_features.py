"""Utilities to expand numeric columns into spline basis features.

The main goal is to keep a simple and reusable API for train/test workflows.

Inspiration
-----------
- ``sklearn.preprocessing.SplineTransformer`` usage pattern.
- Project need for GAM-like non-linear features with linear models.
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

    Parameters
    ----------
    X : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        Columns to expand with splines.
    n_knots : int, default=5
        Number of knots passed to ``SplineTransformer``.
    degree : int, default=3
        Degree of the spline basis.
    include_bias : bool, default=False
        If True, include the bias basis term.
    transformers : dict[str, SplineTransformer] or None, default=None
        Pre-fitted transformers per column. If None, transformers are fitted
        from ``X``.

    Returns
    -------
    X_new : pandas.DataFrame
        DataFrame where each selected column is replaced by spline columns.
    fitted : dict[str, SplineTransformer]
        Transformers used for each expanded column.

    Raises
    ------
    KeyError
        If one requested column is not present in ``X``.
    """
    X_new = X.copy()
    fitted = {} if transformers is None else dict(transformers)

    for col in columns:
        if col not in X_new.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

        if col not in fitted:
            # Fit one transformer per column for clear naming and easy reuse.
            tr = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=include_bias)
            tr.fit(X_new[[col]].values)
            fitted[col] = tr
        else:
            tr = fitted[col]

        out = tr.transform(X_new[[col]].values)
        n_out = out.shape[1]
        names = [f"{col}_spline_{i}" for i in range(n_out)]
        df_out = pd.DataFrame(out, columns=names, index=X_new.index)

        X_new = X_new.drop(columns=[col])
        X_new = pd.concat([X_new, df_out], axis=1)

    return X_new, fitted


if __name__ == "__main__":
    """Small local smoke test."""
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
