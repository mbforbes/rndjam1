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


def least_squares(x: torch.FloatTensor, y_int: torch.IntTensor) -> torch.cuda.FloatTensor:
    """
    w = (X^T X)^-1 X^T y

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
    inv: torch.FloatTensor = torch.from_numpy(np.linalg.pinv(x_t.matmul(x).numpy()))  # type: ignore

    # ... X^T y
    # note that matmul 2D x 2D tensors does matrix multiplication
    #           matmul 2D x 1D tensors does matrix-vector product
    w = inv.matmul(x_t).matmul(y)
    return w.cuda()


def regression_gradient_1loop(
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
    dwdl = torch.cuda.FloatTensor(d)
    dwdl.zero_()
    for i in range(n):
        x_i = x[i,:]
        dwdl += 2 * (w.dot(x_i) - y[i]) * x_i
    dwdl /= n
    return dwdl


def regression_gradient_0loops(
        w: torch.FloatTensor, x: torch.FloatTensor,
        y: torch.FloatTensor) -> torch.FloatTensor:
    """
    See regression_gradient_1loop(...).
    """
    n, d = x.size()
    return ((x.matmul(w) - y).resize_(n, 1) * x).sum(dim=0) * 2 / n


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

    # setup
    y: torch.cuda.FloatTensor = y_int.type(torch.cuda.FloatTensor)
    n, d = x.size()
    # initial w is drawn from gaussian(0, 1)
    w: torch.cuda.FloatTensor = torch.randn(d).type(torch.cuda.FloatTensor)  # type: ignore

    for epoch in range(epochs):
        # compute loss
        y_hat = x.matmul(w)
        l2_loss = (y_hat - y).pow(2).sum()

        # compute gradient
        # grad = regression_gradient_1loop(w, x, y)
        grad = regression_gradient_0loops(w, x, y)

        # maybe adjust lr
        # if epoch > 0 and epoch % 40 == 0:
        #     lr *= 0.8

        # update weights
        w -= lr * grad

        # maybe report
        if epoch % 100 == 0:
            print(' .. epoch {}, lr: {:.4f}, loss: {:.4f} (gradient mag: {:.4f})'.format(
                epoch, lr, l2_loss, grad.norm(p=2)))

    # give back final weights
    return w


def regression_eval(
        w: torch.cuda.FloatTensor, x: torch.cuda.FloatTensor,
        y: torch.cuda.IntTensor) -> Tuple[float, float]:
    """
    Arguments:
        w: 1D (D) weights of linear estimator
        x: 2D (N x D) input data
        y: 1D (D) target labels
    """
    # predict
    y_hat = x.matmul(w)

    # for correct count, see how often the rounded predicted value matches gold
    corr: float = (y_hat.round().type(torch.cuda.IntTensor) == y).sum()

    # for l2 loss, compute sum of squared residuals
    l2_loss: float = (y_hat - y.type(torch.cuda.FloatTensor)).pow(2).sum()

    # report num correct and loss
    return corr, l2_loss


def report(method_name: str, w: torch.cuda.FloatTensor, x: torch.cuda.FloatTensor, y: torch.cuda.IntTensor):
    corr, l2_loss = regression_eval(w, x, y)
    total = len(y)

    print('{} accuracy: {}/{} ({}%)'.format(
        method_name, corr, total, round((corr/total)*100, 2)))
    print('{} l2 loss: {}'.format(method_name, l2_loss))


# execution starts here

# load
train_y_cpu, train_x_cpu = dataio.bin_to_tensors(constants.TRAIN_BIAS)
val_y_cpu, val_x_cpu = dataio.bin_to_tensors(constants.VAL_BIAS)

train_y: torch.cuda.IntTensor = train_y_cpu.cuda()
train_x: torch.cuda.FloatTensor = train_x_cpu.cuda()
val_y: torch.cuda.IntTensor = val_y_cpu.cuda()
val_x: torch.cuda.FloatTensor = val_x_cpu.cuda()

# analytic solution. uses CPU tensors to go to/from numpy for pseudoinverse.
w = least_squares(train_x_cpu, train_y_cpu)
report('least squares', w, train_x, train_y)
report('least squares', w, val_x, val_y)

# try gradient descent
w = gradient_descent_regression(train_x, train_y)
report('gradient descent linear regression', w, train_x, train_y)
report('gradient descent linear regression', w, val_x, val_y)
