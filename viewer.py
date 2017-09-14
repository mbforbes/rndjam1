# imports

# builtins
import code
import csv

# 3rd party
import torch
import visdom

# local
import constants

# settings
constants.TRAIN_UNNORM = 'data/processed/resplit/mnist_train.csv'
constants.VISDOM_ENV = 'rndj1'

# globals
vis = visdom.Visdom()

# code

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


def view_train_datum(n: int = 0):
    """
    Arguments:
        n: which datum to view (0-based indexing)
    """
    # read the desired row from the csv
    img_list = None
    with open(constants.TRAIN_UNNORM, 'r') as f:
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
    vis.image(bigger, win='demo image expanded', env=constants.VISDOM_ENV, opts={
            'caption': 'this should be a bigger {}'.format(label),
    })


def main():
    view_train_datum(0)


if __name__ == '__main__':
    main()
