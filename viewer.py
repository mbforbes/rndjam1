"""
For looking at data. Currently using 'visdom' web dashboard.
"""

# imports
# ---

# builtins
import code
import csv
import random
from typing import List, Dict

# 3rd party
import torch
import visdom

# local
import constants


# globals
# ---
vis = visdom.Visdom()


# code
# ---

def scale(orig: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Pixel-perfect upscaling

    Arguments:
        t: orig, should be 2D tensor n x m
        scale: must be >= 1

    Returns:
        scale*n x scale*m
    """
    # stupid implementation that doesn't utilize any special torch functions
    # for each input point, copy, to all output points
    n, m = orig.size()
    new_n, new_m = scale*n, scale*m
    new = torch.Tensor(new_n, new_m)
    for i in range(n):
        for j in range(m):
            for new_i in range(i*scale, (i+1)*scale):
                for new_j in range(j*scale, (j+1)*scale):
                    new[new_i, new_j] = orig[i, j]
    return new


def jitter(mag: float = 0.1) -> float:
    """
    TODO: elsewhere, guess jitter amt based on data.
    """
    return random.uniform(-mag, mag)


def plot_jitter(data: Dict[str, List[float]], win: str = 'my-scatter') -> None:
    """
    data is a map from named values to the list of their data points
    win is the visdom window name to plot into
    """
    n = sum(len(v) for v in data.values())
    t = torch.FloatTensor(n, 2)
    idx = 0
    keys = sorted(data.keys())
    for x, k in enumerate(keys):
        for y in data[k]:
            t[idx,0] = x + jitter()
            t[idx,1] = y
            idx += 1
    vis.scatter(t, win=win, env=constants.VISDOM_ENV, opts={
        'title': win,
        'xtickvals': list(range(len(keys))),
        'xticklabels': keys, # {i: k for i, k in enumerate(keys)}
    })


def plot_bar(
        x: torch.Tensor, legend: List[str] = [], win: str = 'my-bar',
        opts = {}) -> None:
    """
    Arguments:
        TODO
    """
    baseopts = dict(title=win, legend=legend)
    vis.bar(x, win=win, env=constants.VISDOM_ENV, opts={**baseopts, **opts})


def plot_line(
        x: torch.Tensor, ys: torch.Tensor, legend: List[str] = [],
        win: str = 'my-line', opts={}) -> None:
    """
    Arguments:
        x:  1d (N) x values
        ys: 1d (N) y values, or
            2d (M x N) y values for M lines, one row per line
    """
    if len(ys.size()) > 1:
        ys = ys.t()
    baseopts = dict(title=win, legend=legend)
    vis.line(ys, x, win=win, env=constants.VISDOM_ENV, opts={**baseopts, **opts})


def view_train_datum(n: int = 0):
    """
    Arguments:
        n: which datum to view (0-based indexing)
    """
    # read the desired row from the csv
    img_list = None
    with open(constants.TRAIN_RESPLIT, 'r') as f:
        for i, row in enumerate(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)):
            if i == n:
                img_list = row
                break

    if img_list is None:
        print('ERROR: n ({}) was too large. should be <= {}'.format(n, i))
        return

    # transform it to view it in its normal size
    # visdom takes C x H x W
    # - we only have 1 channel (b/w) so this is fine
    # - H = how many rows, W = how many columns
    # - the data is laid out as row0, then row1, ..., and that seems to be how
    #   view(...) creates the tensor, so this works.
    # - unsqueeze just creates a new dimension
    label = int(img_list[0])
    img_vector = torch.Tensor(img_list[1:])
    img_matrix = img_vector.view(28, 28)
    img_tensor = img_matrix.unsqueeze(0)
    vis.image(img_tensor, win='demo image', env=constants.VISDOM_ENV, opts={
            'caption': 'this should be a {}'.format(label),
    })

    # NOTE: could use vis.images.(...) to view 10 of them in a row. would use
    # torch.stack(...).

    # view it bigger
    bigger = scale(img_matrix, 10).unsqueeze(0)
    vis.image(
        bigger, win='demo image expanded', env=constants.VISDOM_ENV, opts={
            'caption': 'this should be a bigger {}'.format(label),
        }
    )


def main():
    view_train_datum(0)


if __name__ == '__main__':
    main()
