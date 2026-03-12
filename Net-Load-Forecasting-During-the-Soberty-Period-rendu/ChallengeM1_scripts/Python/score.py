import numpy as np


def rmse(y, yhat, digits=0):
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y : array-like
        True values.
    yhat : array-like
        Predicted values.
    digits : int, default=0
        Number of decimal places to round result.

    Returns
    -------
    float
        RMSE between y and yhat, rounded to specified digits.

    """
    return np.round(
        np.sqrt(np.nanmean((y - yhat) ** 2)),
        decimals=digits
    )


def mape(y, yhat):
    """
    Mean Absolute Percentage Error.

    Parameters
    ----------
    y : array-like
        True values (non-zero).
    yhat : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE in percentage (0-100), rounded to 2 decimals.

    """
    return np.round(
        100 * np.nanmean(np.abs(y - yhat) / np.abs(y)),
        decimals=2
    )


def rmse_old(residuals, digits=0):
    """
    Root Mean Squared Error from residuals.

    Legacy function that computes RMSE directly from residuals (y - yhat).

    Parameters
    ----------
    residuals : array-like
        Residuals (y - yhat).
    digits : int, default=0
        Number of decimal places to round result.

    Returns
    -------
    float
        RMSE computed from residuals, rounded to specified digits.

    """
    return np.round(
        np.sqrt(np.nanmean(residuals ** 2)),
        decimals=digits
    )


def absolute_loss(y, yhat):
    """
    Mean Absolute Error (MAE).

    Parameters
    ----------
    y : array-like
        True values.
    yhat : array-like
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.

    """
    return np.nanmean(np.abs(y - yhat))


def bias(y, yhat):
    """
    Mean forecast bias.

    Positive bias indicates systematic under-prediction; negative indicates over-prediction.

    Parameters
    ----------
    y : array-like
        True values.
    yhat : array-like
        Predicted values.

    Returns
    -------
    float
        Mean bias (y - yhat).

    """
    return np.nanmean(y - yhat)


def pinball_loss(y, yhat_quant, quant, output_vect=False):
    """
    Pinball loss for quantile regression evaluation.

    Pinball loss (also called quantile loss or check loss) evaluates quantile predictions:
    L(y, q_tau) = (y - q_tau) * (tau - I(y < q_tau))

    Parameters
    ----------
    y : array-like, shape (n,)
        True values.
    yhat_quant : array-like, shape (n,) or (n, nq)
        Predicted quantiles. If 1D, assumes single quantile. If 2D, each column
        corresponds to a quantile in `quant`.
    quant : array-like, shape (nq,)
        Quantile levels (e.g., [0.1, 0.5, 0.9]). Values should be in (0, 1).
    output_vect : bool, default=False
        If True, return loss for each quantile separately.
        If False, return mean loss across all quantiles.

    Returns
    -------
    float or np.ndarray
        If output_vect=False: scalar mean pinball loss across all quantiles.
        If output_vect=True: array of losses, one per quantile.

    """
    yhat_quant = np.asarray(yhat_quant)
    quant = np.asarray(quant)

    if yhat_quant.ndim == 1:
        yhat_quant = yhat_quant[:, None]

    nq = yhat_quant.shape[1]
    loss_q = np.zeros(nq)

    for q in range(nq):
        loss_q[q] = np.nanmean(
            (y - yhat_quant[:, q]) *
            (quant[q] - (y < yhat_quant[:, q]))
        )

    if output_vect:
        return loss_q
    else:
        return np.mean(loss_q)


def pinball_loss2(res, quant, output_vect=False):
    """
    Pinball loss computed directly from residuals.

    Simplified pinball loss function when residuals are already computed.

    Parameters
    ----------
    res : array-like
        Residuals (y - yhat).
    quant : float or array-like
        Target quantile level(s) in (0, 1).
    output_vect : bool, default=False
        Currently unused; kept for API consistency.

    Returns
    -------
    float
        Mean pinball loss for given quantile.

    """
    return np.nanmean(
        res * (quant - (res < 0))
    )
