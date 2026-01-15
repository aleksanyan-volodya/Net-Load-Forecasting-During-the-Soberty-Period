import numpy as np

def rmse(y, yhat, digits=0):
    return np.round(
        np.sqrt(np.nanmean((y - yhat) ** 2)),
        decimals=digits
    )


def mape(y, yhat):
    return np.round(
        100 * np.nanmean(np.abs(y - yhat) / np.abs(y)),
        decimals=2
    )


def rmse_old(residuals, digits=0):
    return np.round(
        np.sqrt(np.nanmean(residuals ** 2)),
        decimals=digits
    )


def absolute_loss(y, yhat):
    return np.nanmean(np.abs(y - yhat))


def bias(y, yhat):
    return np.nanmean(y - yhat)


def pinball_loss(y, yhat_quant, quant, output_vect=False):
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
    return np.nanmean(
        res * (quant - (res < 0))
    )
