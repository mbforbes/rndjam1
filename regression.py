"""
Linear regression.
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

# for type checking
FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
IntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]
EvalFn = Callable[[FloatTensor, IntTensor], int]
LossFn = Callable[[FloatTensor, FloatTensor, FloatTensor, float], float]
GradientFn = Callable[[FloatTensor, FloatTensor, FloatTensor, float], FloatTensor]
class GDSettings(TypedDict):
    lr: float
    epochs: int
    report_interval: int
class CDSettings(TypedDict):
    epochs: int
    report_interval: int

# tensor types (TT); for use in code
FloatTT = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
IntTT = torch.cuda.IntTensor if torch.cuda.is_available() else torch.IntTensor

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
        w: either 1d (D) or 2d (D x C) weights of linear estimator (depending
            on y's dimensions)
        x: 2d (N x D) input data
        y: either 1d (N) target labels,
               or 2d (N x C) one-hot representation of target labels
        _: unused (for API compatibility with regularized loss functions)

    Returns:
        ordinary least squares loss
    """
    n, d = x.size()
    return (x.matmul(w) - y).pow(2).sum()/n


def ols_analytic(x: torch.FloatTensor, y_int: torch.IntTensor) -> FloatTensor:
    """
    Returns ordinary least squares (OLS) analytic solution:

        w = (X^T X)^+ X^T y

    See the README section for the derivation:

        https://github.com/mbforbes/rndjam1#ordinary-least-squares-ls

    Arguments:
        x: 2d (N x D) input data
        y: either 1d (N) target labels,
               or 2d (N x C) one-hot representation of target labels

    Returns
        w: either 1d (D) or 2d (D x C) weights of linear estimator (depending
            on y's dimensions)
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
    return w.type(FloatTT)


def ols_gradient(w: FloatTensor, x: FloatTensor, y: FloatTensor, _: float) -> FloatTensor:
    """
    Returns ordinary least squares (OLS) gradient for per-datum averaged loss.

    See the README section for the derivation:

        https://github.com/mbforbes/rndjam1#ordinary-least-squares-ls

    Note that the gradient is either a vector (if Y has dimensions (N)):

        dL/dw = [dL/dw_1,  dL/dw_2,  ...,  dL/dw_d]

    ... or a matrix (if Y has dimensions (N x C)):

        dL/dW = [[dL/dw_11,  dL/dw_12,  ...,  dL/dw_1c],
                 [dL/dw_21,  dL/dw_22,  ...,  dL/dw_2c],
                 ...
                 [dL/dw_d1,  dL/dw_d2,  ...,  dL/dw_dc]]

    Arguments:
        w: either 1d (D) or 2d (D x C) weights of linear estimator (depending
            on y's dimensions)
        x: 2d (N x D) input data
        y: either 1d (N) target labels,
               or 2d (N x C) one-hot representation of target labels
        _: unused (for API compatibility with regularized loss functions)

    Returns:
        dL/dw: either 1d (D) or 2d (D x C) derivative of 1/n averaged OLS loss
            L with respect to weights w (depending on y's dimensions)
    """
    n, d = x.size()
    return (2/n)*(x.t().matmul(x.matmul(w) - y))


def ols_coordinate_descent(x: FloatTensor, y_int: IntTensor,
        settings: CDSettings) -> FloatTensor:
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
    y = y_int.type(FloatTT)
    n, d = x.size()
    # initial w is drawn from gaussian(0, 1)
    w = torch.randn(d).type(FloatTT)

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


def ridge_analytic(
        x: FloatTensor, y_int: IntTensor,
        lmb: float) -> FloatTensor:
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
    i = torch.nn.init.eye(torch.FloatTensor(d,d).type(FloatTT))
    y = y_int.type(FloatTT)

    # formula
    return (x_t.matmul(x) + lmb*n*i).inverse().matmul(x_t).matmul(y)


def ridge_gradient(
        w: FloatTensor, x: FloatTensor, y: FloatTensor,
        lmb: float) -> FloatTensor:
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

def lasso_loss(
        w: FloatTensor, x: FloatTensor, y: FloatTensor, lmb: float) -> float:
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


def lasso_gradient(
        w: FloatTensor, x: FloatTensor, y: FloatTensor,
        lmb: float) -> FloatTensor:
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


def lasso_coordinate_descent(
        x: FloatTensor, y_int: IntTensor, lmb: float,
        settings: CDSettings) -> FloatTensor:
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
    y = y_int.type(FloatTT)
    n, d = x.size()
    w = torch.randn(d).type(FloatTT)  # N(0,1)

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
        x: FloatTensor, y_int: IntTensor, lmb: float, loss_fn: LossFn,
        grad_fn: GradientFn, settings: GDSettings) -> FloatTensor:
    """
    Arguments:
        x: 2d (N x D) input data
        y: either 1d (N) target labels,
               or 2d (N x C) one-hot representation of target labels

    Returns:
        w: either 1d (D) or 2d (D x C) weights of linear estimator (depending
            on y's dimensions)

    """
    # extract GD settings
    lr = settings['lr']
    epochs = settings['epochs']
    report_interval = settings['report_interval']

    # setup
    y = y_int.type(FloatTT)
    n, d = x.size()
    w_dims = (d,) if len(y.size()) == 1 else (d, y.size()[1])
    # initial w is drawn from gaussian(0, 1)
    w = torch.randn(torch.Size(w_dims)).type(FloatTT)

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


def scalar_eval(y_hat: FloatTensor, y: IntTensor) -> int:
    """
    Returns number correct: how often the rounded predicted value matches gold.

    Arguments:
        y_hat: 1d (N): guesses for class label
        y: 1d (N): numeric class labels

    Returns:
        nubmer correct
    """
    return (y_hat.round().type(IntTT) == y).sum()


def multiclass_eval(y_hat: FloatTensor, y: IntTensor) -> int:
    """
    Returns number correct: how often the rounded predicted value matches gold.

    Arguments:
        y_hat: 2d (N x C): guesses for each class
        y: 2d (N x C): onehot representation of class labels

    Returns:
        nubmer correct
    """
    # max(dim) returns both values and indices. compare best indices from
    # predictions and gold (which are just onehot)
    _, pred_idxes = y_hat.max(1)
    _, gold_idxes = y.max(1)
    return (pred_idxes == gold_idxes).sum()


def regression_report(
        method_name: str, w: FloatTensor, x: FloatTensor, y: IntTensor,
        lmb: float, eval_fn: EvalFn, loss_fn: LossFn) -> None:
    """
    Arguments:
        method_name: printable representation of estimation technique (probably
            scalar vs multiclass, model, and optimization procedure)
        w: either 1d (D) or 2d (D x C) weights of linear estimator
        x: 2d (N x D) input data
        y: either 1d (N) or 2d (N x C) target labels
        lmb: regularization lambda (if relevant), or dummy val
        eval_fn: function to determine number gotten correct
        loss_fn: function to determine loss
    """
    # compute
    y_hat = x.matmul(w)
    corr = eval_fn(y_hat, y)
    total = len(y)
    loss = loss_fn(w, x, y.type(FloatTT), lmb)

    # report
    print('{} accuracy: {}/{} ({}%)'.format(
        method_name, corr, total, round((corr/total)*100, 2)))
    print('{} average loss: {}'.format(method_name, loss))


def scalar():
    """
    Regress to scalar value, e.g. the image of "5" goes to the number 5.
    """
    # load
    print('Loading data...')
    train_y_cpu, train_x_cpu = dataio.bin_to_tensors(constants.TRAIN_BIAS)
    val_y_cpu, val_x_cpu = dataio.bin_to_tensors(constants.VAL_BIAS)

    print('Moving data to GPU...')
    train_y = train_y_cpu.type(IntTT)
    train_x = train_x_cpu.type(FloatTT)
    val_y = val_y_cpu.type(IntTT)
    val_x = val_x_cpu.type(FloatTT)

    print('Starting experiments...')
    dummy = 0.0

    # OLS analytic solution. uses CPU tensors to go to/from numpy for pseudoinverse.
    w = ols_analytic(train_x_cpu, train_y_cpu)
    regression_report('OLS analytic (train)', w, train_x, train_y, dummy, scalar_eval, ols_loss)
    regression_report('OLS analytic (val)', w, val_x, val_y, dummy, scalar_eval, ols_loss)

    # OLS gradient descent
    ols_gd_settings: GDSettings = {'lr': 0.02, 'epochs': 1500, 'report_interval': 100}
    w = gradient_descent_regression(train_x, train_y, -1, ols_loss, ols_gradient, ols_gd_settings)
    regression_report('OLS GD (train)', w, train_x, train_y, dummy, scalar_eval, ols_loss)
    regression_report('OLS GD (val)', w, val_x, val_y, dummy, scalar_eval, ols_loss)

    # OLS coordinate descent
    w = ols_coordinate_descent(train_x, train_y, {'epochs': 150, 'report_interval': 10})
    regression_report('Coordinate descent (train)', w, train_x, train_y, dummy, scalar_eval, ols_loss)
    regression_report('Coordinate descent (val)', w, val_x, val_y, dummy, scalar_eval, ols_loss)

    # ridge analytic solution
    for lmb in [0.2]:
        w = ridge_analytic(train_x, train_y, lmb)
        # code.interact(local=dict(globals(), **locals()))
        regression_report('Ridge analytic (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, scalar_eval, ridge_loss)
        regression_report('Ridge analytic (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, scalar_eval, ridge_loss)

    # ridge GD
    ridge_gd_settings: GDSettings = {'lr': 0.02, 'epochs': 500, 'report_interval': 100}
    for lmb in [0.2]:
        w = gradient_descent_regression(train_x, train_y, lmb, ridge_loss, ridge_gradient, ridge_gd_settings)
        regression_report('Ridge GD (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, scalar_eval, ridge_loss)
        regression_report('Ridge GD (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, scalar_eval, ridge_loss)

    # lasso GD
    lasso_gd_settings: GDSettings = {'lr': 0.02, 'epochs': 1000, 'report_interval': 100}
    for lmb in [0.2]:
        w = gradient_descent_regression(train_x, train_y, lmb, lasso_loss, lasso_gradient, lasso_gd_settings)
        regression_report('Lasso GD (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, scalar_eval, lasso_loss)
        regression_report('Lasso GD (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, scalar_eval, lasso_loss)

    # lasso CD
    lasso_cd_settings: CDSettings = {'epochs': 100, 'report_interval': 10}
    for lmb in [0.2]:
        w = lasso_coordinate_descent(train_x, train_y, lmb, lasso_cd_settings)
        regression_report('Lasso CD (train) lambda={}'.format(lmb), w, train_x, train_y, lmb, scalar_eval, lasso_loss)
        regression_report('Lasso CD (val) lambda={}'.format(lmb), w, val_x, val_y, lmb, scalar_eval, lasso_loss)


def multi():
    """
    Regress to 10 binary-valued classes at once, e.g., the image of "5" goes to
    [0, 0, 0, 0, 1, 0, 0, 0, 0].
    """
    # load
    print('Loading data...')
    train_y_cpu, train_x_cpu = dataio.bin_to_tensors(constants.TRAIN_ONEHOT, 10)
    val_y_cpu, val_x_cpu = dataio.bin_to_tensors(constants.VAL_ONEHOT, 10)

    print('Moving data to GPU...')
    train_y = train_y_cpu.type(IntTT)
    train_x = train_x_cpu.type(FloatTT)
    val_y = val_y_cpu.type(IntTT)
    val_x = val_x_cpu.type(FloatTT)

    print('Starting experiments...')
    dummy = 0.0

    # OLS analytic solution. uses CPU tensors to go to/from numpy for
    # pseudoinverse.
    w = ols_analytic(train_x_cpu, train_y_cpu)
    regression_report('OLS analytic (train)', w, train_x, train_y, dummy, multiclass_eval, ols_loss)
    regression_report('OLS analytic (val)', w, val_x, val_y, dummy, multiclass_eval, ols_loss)

    # OLS gradient descent
    ols_gd_settings: GDSettings = {'lr': 0.02, 'epochs': 3500, 'report_interval': 500}
    w = gradient_descent_regression(train_x, train_y, -1, ols_loss, ols_gradient, ols_gd_settings)
    regression_report('OLS GD (train)', w, train_x, train_y, dummy, multiclass_eval, ols_loss)
    regression_report('OLS GD (val)', w, val_x, val_y, dummy, multiclass_eval, ols_loss)


# execution starts here
# scalar()
multi()
