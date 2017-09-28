"""
TODO: clean up these names.

Know when to use the following:
- linear
- least squares
- l2
- regression

... as some might imply the others and be redundant.
"""

# imports
# ---

# builtins
import code
from typing import Tuple, Union, Callable

# 3rd party
import numpy as np
import torch

# local
import constants
import dataio


# types
# ---

FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
IntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]
LossFn = Callable[[FloatTensor, FloatTensor, FloatTensor, float], float]

# functions
# ---

#
# ordinary least squares (OLS)
#

def ols_analytic(x: torch.FloatTensor, y_int: torch.IntTensor) -> torch.cuda.FloatTensor:
    """
    Returns ordinary least squares (OLS) analytic solution:

        w = (X^T X)^+ X^T y

    Arguments:
        x: 2D (N x D) tensor
        y: 1D (D) tensor

    Returns
        weights: 1D (D) tensor
    """
    # for regression, y becomes real values instead of labels
    y: torch.FloatTensor = y_int.type(torch.FloatTensor)

    # save X^T, as used twice
    x_t = x.t()

    # (X^T X)^-1
    # because matrix is singular,
    # - tensor.inverse() does not work
    # - np.linalg.inv()  does not work
    # - np.linalg.pinv() works; uses SVD
    inv: torch.FloatTensor = torch.from_numpy(np.linalg.pinv(x_t.matmul(x).numpy()))

    # ... X^T y
    # note that matmul 2D x 2D tensors does matrix multiplication
    #           matmul 2D x 1D tensors does matrix-vector product
    w = inv.matmul(x_t).matmul(y)
    return w.cuda()


def ols_avg_loss(w: FloatTensor, x: FloatTensor, y: FloatTensor, _: float) -> float:
    """
    Returns ordinary least squares (OLS) loss, averaged per datum:

        1/n ||y - Xw||_2^2

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
        _: unused (for API compatibility with regularized loss functions)

    Returns:
        ordinary least squares loss
    """
    n, d = x.size()
    return (x.matmul(w) - y).pow(2).sum()/n


def ols_gradient(w: FloatTensor, x: FloatTensor, y: FloatTensor) -> FloatTensor:
    """
    Returns ordinary least squares (OLS) gradient for per-datum averaged loss.

    See the README section for the derivation:
    https://github.com/mbforbes/rndjam1/#least-squares-loss

    Note that the gradient is a vector:

        dL/dw = [dL/dw_1,  dL/dw_2,  ...,  dL/dw_d]

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels

    Returns:
        dL/dw: 1D (D) derivative of 1/n averaged OLS loss L with respect to
            weights w.
    """
    n, d = x.size()
    return (2/n)*(w.matmul(x.t()).matmul(x) - y.matmul(x))


#
# ridge regression
#

def ridge_analytic(x: torch.cuda.FloatTensor, y_int: IntTensor, lmb: float) -> torch.cuda.FloatTensor:
    # setup
    n, d = x.size()
    x_t = x.t()
    i = torch.nn.init.eye(torch.cuda.FloatTensor(d,d))
    y = y_int.type(torch.cuda.FloatTensor)

    # formula
    return (x_t.matmul(x) + lmb*i).inverse().matmul(x_t).matmul(y)


def ridge_loss(w: FloatTensor, x: FloatTensor, y: FloatTensor, lmb: float) -> float:
    """
    Returns ridge loss, where the the data component is averaged per datum (as
    in ols_avg_loss(...)), and the regularization component is not:

        1/n ( ||y - Xw||_2^2 ) + lmb * w_2^2

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels

    Returns:
        ridge loss
    """
    return ols_avg_loss(w, x, y, 0.0) + lmb*w.pow(2).sum()


#
# TODO: generalize the following functions to support all naive regression
# methods
#

def gradient_descent_regression(
        x: torch.cuda.FloatTensor, y_int: torch.cuda.IntTensor) -> torch.cuda.FloatTensor:
    """
    Arguments:
        x: 2D (N x D) input data
        y: 1D (D) target labels

    Returns:
        w: 1D (D) weights of linear estimator
    """
    # settings
    lr = 0.022
    epochs = 1500
    dummy = 0.0

    # setup
    y: torch.cuda.FloatTensor = y_int.type(torch.cuda.FloatTensor)
    n, d = x.size()
    # initial w is drawn from gaussian(0, 1)
    w: torch.cuda.FloatTensor = torch.randn(d).type(torch.cuda.FloatTensor)

    for epoch in range(epochs):
        # compute loss
        loss = ols_avg_loss(w, x, y, dummy)

        # compute gradient
        grad = ols_gradient(w, x, y)

        # maybe adjust lr
        # if epoch > 0 and epoch % 40 == 0:
        #     lr *= 0.8

        # update weights
        w -= lr * grad

        # maybe report
        if epoch % 100 == 0:
            print(' .. epoch {}, lr: {:.4f}, loss: {:.4f} (gradient mag: {:.4f})'.format(
                epoch, lr, loss, grad.norm(p=2)))

    # give back final weights
    return w


def regression_eval(
        w: torch.cuda.FloatTensor, x: torch.cuda.FloatTensor,
        y: torch.cuda.IntTensor, lmb: float, loss_fn: LossFn,) -> Tuple[float, float]:
    """
    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
    """
    # predict
    y_hat = x.matmul(w)

    # for correct count, see how often the rounded predicted value matches gold
    corr = (y_hat.round().type(torch.cuda.IntTensor) == y).sum()

    # for loss, compute sum of squared residuals
    loss = loss_fn(w, x, y.type(torch.cuda.FloatTensor), lmb)

    # report num correct and loss
    return corr, loss


def report(
        method_name: str, w: torch.cuda.FloatTensor, x: torch.cuda.FloatTensor,
        y: torch.cuda.IntTensor, lmb: float, loss_fn: LossFn) -> None:
    corr, loss = regression_eval(w, x, y, lmb, loss_fn)
    total = len(y)

    print('{} accuracy: {}/{} ({}%)'.format(
        method_name, corr, total, round((corr/total)*100, 2)))
    print('{} average loss: {}'.format(method_name, loss))


# execution starts here

# load
train_y_cpu, train_x_cpu = dataio.bin_to_tensors(constants.TRAIN_BIAS)
val_y_cpu, val_x_cpu = dataio.bin_to_tensors(constants.VAL_BIAS)

train_y: torch.cuda.IntTensor = train_y_cpu.cuda()
train_x: torch.cuda.FloatTensor = train_x_cpu.cuda()
val_y: torch.cuda.IntTensor = val_y_cpu.cuda()
val_x: torch.cuda.FloatTensor = val_x_cpu.cuda()

dummy = 0.0

# OLS analytic solution. uses CPU tensors to go to/from numpy for pseudoinverse.
w = ols_analytic(train_x_cpu, train_y_cpu)
report('OLS analytic (train)', w, train_x, train_y, dummy, ols_avg_loss)
report('OLS analytic (val)', w, val_x, val_y, dummy, ols_avg_loss)

# OLS gradient descent
# w = gradient_descent_regression(train_x, train_y)
# report('OLS GD (train)', w, train_x, train_y, dummy, ols_avg_loss)
# report('OLS GD (val)', w, val_x, val_y, dummy, ols_avg_loss)

# ridge analytic solution
for lmb in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
    w = ridge_analytic(train_x, train_y, lmb)
    report('Ridge analytic (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, ridge_loss)
    report('Ridge analytic (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, ridge_loss)
