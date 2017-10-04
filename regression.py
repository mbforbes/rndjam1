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
from typing import Tuple, Dict, Union, Callable
from mypy_extensions import TypedDict

# 3rd party
import numpy as np
import torch

# local
import constants
import dataio
import viewer


# types
# ---

FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
IntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]
LossFn = Callable[[FloatTensor, FloatTensor, FloatTensor, float], float]
GradientFn = Callable[[FloatTensor, FloatTensor, FloatTensor, float], FloatTensor]
class GDSettings(TypedDict):
    lr: float
    epochs: int
    report_interval: int
class CDSettings(TypedDict):
    epochs: int
    report_interval: int

# functions
# ---

#
# util (consider pulling out)
#

def soft_thresh(x: float, lmb: float) -> float:
    """
    This is a scalar version of torch.nn.functional.softshrink.

              x + lmb  if  x  <   -lmb
    Returns   0        if  x \in [-lmb, lmb]
              x - lmb  if  x  >    lmb
    """
    if x < lmb:
        return x + lmb
    elif x > lmb:
        return x - lmb
    else:
        return 0.0


#
# ordinary least squares (OLS)
#

def ols_loss(w: FloatTensor, x: FloatTensor, y: FloatTensor, _: float = -1) -> float:
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


def ols_analytic(x: torch.FloatTensor, y_int: torch.IntTensor) -> torch.cuda.FloatTensor:
    """
    Returns ordinary least squares (OLS) analytic solution:

        w = (X^T X)^+ X^T y

    See the README section for the derivation:

        https://github.com/mbforbes/rndjam1#ordinary-least-squares-ls

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


def ols_gradient(w: FloatTensor, x: FloatTensor, y: FloatTensor, _: float) -> FloatTensor:
    """
    Returns ordinary least squares (OLS) gradient for per-datum averaged loss.

    See the README section for the derivation:

        https://github.com/mbforbes/rndjam1#ordinary-least-squares-ls

    Note that the gradient is a vector:

        dL/dw = [dL/dw_1,  dL/dw_2,  ...,  dL/dw_d]

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
        _: unused (for API compatibility with regularized loss functions)

    Returns:
        dL/dw: 1D (D) derivative of 1/n averaged OLS loss L with respect to
            weights w.
    """
    n, d = x.size()
    return (2/n)*(x.t().matmul(x.matmul(w) - y))


def ols_coordinate_descent(x: torch.cuda.FloatTensor, y_int: torch.cuda.IntTensor,
        settings: CDSettings) -> torch.cuda.FloatTensor:
    """
    Runs OLS coordinate descent.

    See the README section for the derivation:

        https://github.com/mbforbes/rndjam1#ordinary-least-squares-ls

    Arguments:
        x: 2d (N x D) input data
        y: 1d (D) target labels
        settings: 'epochs' and 'report_interval'

    Returns:
        w: 1d (D) weights
    """
    epochs = settings['epochs']
    report_interval = settings['report_interval']
    y = y_int.type(torch.cuda.FloatTensor)
    n, d = x.size()
    # initial w is drawn from gaussian(0, 1)
    w = torch.randn(d).type(torch.cuda.FloatTensor)

    # precompute sq l2 column norms (don't change as x stays fixed)
    col_l2s = x.pow(2).sum(0)

    # compute initial residual
    r = y - x.matmul(w)

    # iterate
    for epoch in range(epochs):
        if epoch % report_interval == 0:
            print('OLS CD iter {}, loss = {}'.format(epoch, ols_loss(w, x, y)))

        for j in range(d):
            # save old val for redisual update
            w_j_old = w[j]

            # update j (avoiding divde by zero)
            norm = col_l2s[j]
            w[j] = w[j] + x[:,j].matmul(r) / norm if norm != 0.0 else 0.0

            # update residual
            r += (w_j_old - w[j]) * x[:,j]
    return w


#
# ridge regression
#

def ridge_loss(w: FloatTensor, x: FloatTensor, y: FloatTensor, lmb: float) -> float:
    """
    Returns ridge loss, where the the data component is averaged per datum (as
    in ols_loss(...)), and the regularization component is not:

        1/n ( ||y - Xw||_2^2 ) + lmb * ||w||_2^2

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
        lmb: regularization strength (lambda)

    Returns:
        ridge loss
    """
    return ols_loss(w, x, y, 0.0) + lmb*w.pow(2).sum()


def ridge_analytic(x: torch.cuda.FloatTensor, y_int: IntTensor, lmb: float) -> torch.cuda.FloatTensor:
    """
    See the README section for the derivation:

        https://github.com/mbforbes/rndjam1#ridge-regression-rr

    Arguments:
        x: 2D (N x D) input data
        y_int: 1D (D) target labels
        lmb: regularization strength (lambda)

    Returns:
        ridge weights
    """
    # setup
    n, d = x.size()
    x_t = x.t()
    i = torch.nn.init.eye(torch.cuda.FloatTensor(d,d))
    y = y_int.type(torch.cuda.FloatTensor)

    # formula
    return (x_t.matmul(x) + lmb*n*i).inverse().matmul(x_t).matmul(y)


def ridge_gradient(w: FloatTensor, x: FloatTensor, y: FloatTensor, lmb: float) -> torch.cuda.FloatTensor:
    """
    See the README section for the derivation:

        https://github.com/mbforbes/rndjam1#ridge-regression-rr

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
        lmb: regularization strength (lambda)

    Returns:
        ridge gradient
    """
    n, d = x.size()
    return (2/n)*(x.t().matmul(x.matmul(w) - y)) + 2*lmb*w


#
# lasso
#

def lasso_loss(w: FloatTensor, x: FloatTensor, y: FloatTensor, lmb: float) -> float:
    """
    Returns lasso loss, where the the data component is averaged per datum (as
    in ols_loss(...)), and the regularization component is not:

        1/n ( ||y - Xw||_2^2 ) + lmb * ||w||_1

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
        lmb: regularization strength (lambda)

    Returns:
        lasso loss
    """
    return ols_loss(w, x, y, 0.0) + lmb*w.abs().sum()


def lasso_gradient(w: FloatTensor, x: FloatTensor, y: FloatTensor, lmb: float) -> torch.cuda.FloatTensor:
    """
    See the README section for the derivation:

        https://github.com/mbforbes/rndjam1#lasso

    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
        lmb: regularization strength (lambda)

    Returns:
        lasso gradient
    """
    n, d = x.size()
    return (2/n)*(x.t().matmul(x.matmul(w) - y)) + lmb*w.sign()


def lasso_coordinate_descent(x: torch.cuda.FloatTensor, y_int: torch.cuda.IntTensor,
        lmb: float, settings: CDSettings) -> torch.cuda.FloatTensor:
    """
    Runs lasso coordinate descent.

    See the README section for the derivation (still TODO):

        https://github.com/mbforbes/rndjam1#lasso

    TODO: refactor with ols coordinate descent once this is working.

    Arguments:
        x: 2d (N x D) input data
        y: 1d (D) target labels
        settings: 'epochs' and 'report_interval'

    Returns:
        w: 1d (D) weights
    """
    epochs = settings['epochs']
    report_interval = settings['report_interval']
    y = y_int.type(torch.cuda.FloatTensor)
    n, d = x.size()
    w = torch.randn(d).type(torch.cuda.FloatTensor)  # N(0,1)

    # precompute sq l2 column norms (don't change as x stays fixed)
    col_l2s = x.pow(2).sum(0)

    # compute initial residual
    r = y - x.matmul(w)

    # iterate
    for epoch in range(epochs):
        # maybe report
        if epoch % report_interval == 0:
            print('lasso CD epoch {}, loss: {:.4f} (0 ws: {})'.format(
                epoch, lasso_loss(w, x, y, lmb), (w == 0).sum()))

        for j in range(d):
            # save old val for redisual update
            w_j_old = w[j]

            # compute new weight value
            if col_l2s[j] != 0.0:
                rho = x[:,j].matmul(r + w[j]*x[:,j])
                z = (n * lmb) / 2
                w[j] = soft_thresh(rho, z) / col_l2s[j]
            else:
                w[j] = 0.0

            # update residual
            r += (w_j_old - w[j]) * x[:,j]
    return w

#
# general
#

def gradient_descent_regression(
        x: torch.cuda.FloatTensor, y_int: torch.cuda.IntTensor, lmb: float,
        loss_fn: LossFn, grad_fn: GradientFn, settings: GDSettings) -> torch.cuda.FloatTensor:
    """
    Arguments:
        x: 2D (N x D) input data
        y: 1D (D) target labels

    Returns:
        w: 1D (D) weights of linear estimator
    """
    # extract GD settings
    lr = settings['lr']
    epochs = settings['epochs']
    report_interval = settings['report_interval']

    # setup
    y = y_int.type(torch.cuda.FloatTensor)
    n, d = x.size()
    # initial w is drawn from gaussian(0, 1)
    w = torch.randn(d).type(torch.cuda.FloatTensor)

    for epoch in range(epochs):
        # NOTE: can adjust lr if desired

        # compute gradient
        grad = grad_fn(w, x, y, lmb)

        # update weights
        w -= lr * grad

        # maybe report
        if epoch % report_interval == 0:
            print(' .. epoch {}, lr: {:.4f}, loss: {:.4f} (gradient mag: {:.4f}) (0 ws: {})'.format(
                epoch, lr, loss_fn(w, x, y, lmb), grad.norm(p=2), (w == 0).sum()))

    # give back final weights
    return w


def naive_regression_eval(
        method_name: str, w: torch.cuda.FloatTensor, x: torch.cuda.FloatTensor,
        y: torch.cuda.IntTensor, lmb: float, loss_fn: LossFn) -> None:
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

    total = len(y)
    print('{} accuracy: {}/{} ({}%)'.format(
        method_name, corr, total, round((corr/total)*100, 2)))
    print('{} average loss: {}'.format(method_name, loss))


# execution starts here

# load
print('Loading data...')
train_y_cpu, train_x_cpu = dataio.bin_to_tensors(constants.TRAIN_BIAS)
val_y_cpu, val_x_cpu = dataio.bin_to_tensors(constants.VAL_BIAS)

print('Moving data to GPU...')
train_y: torch.cuda.IntTensor = train_y_cpu.cuda()
train_x: torch.cuda.FloatTensor = train_x_cpu.cuda()
val_y: torch.cuda.IntTensor = val_y_cpu.cuda()
val_x: torch.cuda.FloatTensor = val_x_cpu.cuda()

print('Starting experiments...')
dummy = 0.0

# # OLS analytic solution. uses CPU tensors to go to/from numpy for pseudoinverse.
# w = ols_analytic(train_x_cpu, train_y_cpu)
# naive_regression_eval('OLS analytic (train)', w, train_x, train_y, dummy, ols_loss)
# naive_regression_eval('OLS analytic (val)', w, val_x, val_y, dummy, ols_loss)

# # OLS gradient descent
# ols_gd_settings: GDSettings = {'lr': 0.02, 'epochs': 1500, 'report_interval': 100}
# w = gradient_descent_regression(train_x, train_y, -1, ols_loss, ols_gradient, ols_gd_settings)
# naive_regression_eval('OLS GD (train)', w, train_x, train_y, dummy, ols_loss)
# naive_regression_eval('OLS GD (val)', w, val_x, val_y, dummy, ols_loss)

# # OLS coordinate descent
# w = ols_coordinate_descent(train_x, train_y, {'epochs': 150, 'report_interval': 10})
# naive_regression_eval('Coordinate descent (train)', w, train_x, train_y, dummy, ols_loss)
# naive_regression_eval('Coordinate descent (val)', w, val_x, val_y, dummy, ols_loss)

# # ridge analytic solution
# for lmb in [0.2]:
#     w = ridge_analytic(train_x, train_y, lmb)
#     # code.interact(local=dict(globals(), **locals()))
#     naive_regression_eval('Ridge analytic (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, ridge_loss)
#     naive_regression_eval('Ridge analytic (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, ridge_loss)

# # ridge GD
# ridge_gd_settings: GDSettings = {'lr': 0.02, 'epochs': 500, 'report_interval': 100}
# for lmb in [0.2]:
#     w = gradient_descent_regression(train_x, train_y, lmb, ridge_loss, ridge_gradient, ridge_gd_settings)
#     naive_regression_eval('Ridge GD (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, ridge_loss)
#     naive_regression_eval('Ridge GD (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, ridge_loss)

# # lasso GD
# lasso_gd_settings: GDSettings = {'lr': 0.02, 'epochs': 1000, 'report_interval': 100}
# for lmb in [0.2]:
#     w = gradient_descent_regression(train_x, train_y, lmb, lasso_loss, lasso_gradient, lasso_gd_settings)
#     naive_regression_eval('Lasso GD (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, lasso_loss)
#     naive_regression_eval('Lasso GD (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, lasso_loss)

# lasso CD
lasso_cd_settings: CDSettings = {'epochs': 100, 'report_interval': 10}
for lmb in [0.2]:
    w = lasso_coordinate_descent(train_x, train_y, lmb, lasso_cd_settings)
    naive_regression_eval('Lasso CD (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, lasso_loss)
    naive_regression_eval('Lasso CD (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, lasso_loss)
