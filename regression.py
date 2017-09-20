# builtins
import code
from typing import Tuple

# 3rd party
import numpy as np
import torch

# local
import constants
import dataio


def least_squares(x: torch.FloatTensor, y: torch.IntTensor) -> torch.FloatTensor:
    """
    w = (X^T X)^-1 X^T y

    Arguments:
        x: 2D (N x D) tensor
        y: 1D (D) tensor

    Returns
        weights: 1D (D) tensor
    """
    # for regression, y becomes real values instead of labels
    y = y.type(torch.FloatTensor)

    # save X^T, as used twice
    x_t = x.t()

    # (X^T X)^-1
    # because matrix is singular,
    # - tensor.inverse() does not work
    # - np.linalg.inv()  does not work
    # - np.linalg.pinv() works; uses SVD
    inv = torch.from_numpy(np.linalg.pinv(x_t.matmul(x).numpy()))

    # ... X^T y
    # note that matmul 2D x 2D tensors does matrix multiplication
    #           matmul 2D x 1D tensors does matrix-vector product
    w = inv.matmul(x_t).matmul(y)
    return w


def evaluate(
        w: torch.FloatTensor, x: torch.FloatTensor,
        y: torch.IntTensor) -> Tuple[float, float]:
    """
    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
    """
    # predict
    y_hat = x.matmul(w)

    # for correct count, see how often the rounded predicted value matches gold
    corr = (y_hat.round().type(torch.IntTensor) == y).sum()

    # for l2 loss, compute sum of squared residuals
    l2_loss = (y_hat - y.type(torch.FloatTensor)).pow(2).sum()

    # report num correct and loss
    return corr, l2_loss


def report(method_name, w, x, y):
    corr, l2_loss = evaluate(w, x, y)
    total = len(y)

    print('{} accuracy: {}/{} ({}%)'.format(
        method_name, corr, total, round((corr/total)*100, 2)))
    print('{} l2 loss: {}'.format(method_name, l2_loss))


# execution starts here

# load
train_y, train_x = dataio.bin_to_tensors(constants.TRAIN_BIAS)
val_y, val_x = dataio.bin_to_tensors(constants.VAL_BIAS)

# get model parameters
w = least_squares(train_x, train_y)

# see train perf
report('least squares', w, train_x, train_y)

# try against val
report('least squares', w, val_x, val_y)
