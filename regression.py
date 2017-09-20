"""
TODO: clean up these names.

Know when to use the following:
- linear
- least squares
- l2
- regression

... as some might imply the others and be redundant.
"""

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


def regression_gradient(
        w: torch.FloatTensor, x: torch.FloatTensor,
        y: torch.FloatTensor) -> torch.FloatTensor:
    """
    Computes (average, per datum) gradient for linear regression using squared
    error loss.

    The loss function L is the sum of squared errors

        L(w,x,y) = ||Xw - y||_2^2

                 = \sum_{i=1}^n (w^T x_i - y_i)^2

    We'll redefine L to be the average loss per datum:

        L(w,x,y) = 1/n \sum_{i=1}^n (w^T x_i - y_i)^2

    The gradient with respect to the weight vector w is:

        dL/dw = 1/n \sum_{i=1}^n 2(w^T X_i - y_i) X_i

    Note that the gradient is a vector:

        dL/dw = [dL/dw_1,  dL/dw_2,  ...,  dL/dw_d]

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels

    Returns:
        dL/dw: 1D (D) derivative of squared error loss L, with the model of
            linear regression (y_hat = w^T X), with respect to weights w.
    """
    n = len(x)
    d = len(w)
    dwdl = torch.zeros(d)
    for i in range(n):
        x_i = x[i,:]
        dwdl += 2 * (w.dot(x_i) - y[i]) * x_i
    dwdl /= n
    return dwdl


def gradient_descent_regression(
        x: torch.FloatTensor, y: torch.IntTensor) -> torch.FloatTensor:
    """
    Arguments:
        x: 2D (N x D) input data
        y: 1D (D) target labels

    Returns:
        w: 1D (D) weights of linear estimator
    """
    # settings
    lr = 0.02
    epochs = 200

    # setup
    y = y.type(torch.FloatTensor)
    n, d = x.size()
    w = torch.randn(d)  # initial w is drawn from gaussian(0, 1)

    for epoch in range(epochs):
        # compute loss
        y_hat = x.matmul(w)
        l2_loss = (y_hat - y).pow(2).sum()

        # compute gradient
        grad = regression_gradient(w, x, y)

        # maybe adjust lr
        # if epoch > 0 and epoch % 40 == 0:
        #     lr *= 0.8

        # update weights
        w -= lr * grad

        # maybe report
        if epoch % 10 == 0:
            print(' .. epoch {}, lr: {:.4f}, loss: {:.4f} (gradient mag: {:.4f})'.format(
                epoch, lr, l2_loss, torch.norm(grad, p=2)))

    # give back final weights
    return w


def regression_eval(
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
    corr, l2_loss = regression_eval(w, x, y)
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

# see train & val perf
report('least squares', w, train_x, train_y)
report('least squares', w, val_x, val_y)

# try gradient descent
w = gradient_descent_regression(train_x, train_y)
report('gradient descent linear regression', w, train_x, train_y)
report('gradient descent linear regression', w, val_x, val_y)
