# imports

# builtins
import code
import csv

# 3rd party
import torch
import visdom


# settings
train_unnorm = 'data/processed/resplit/mnist_train.csv'
visdom_env = 'rndj1'

# globals
vis = visdom.Visdom()

# code

def view_train_datum(n: int = 0):
    """
    Arguments:
        n: which datum to view (0-based indexing)
    """
    # read the desired row from the csv
    img_list = None
    with open(train_unnorm, 'r') as f:
        for i, row in enumerate(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)):
            if i == n:
                img_list = row
                break

    if img_list is None:
        print('ERROR: n ({}) was too large. should be <= {}'.format(n, i))
        return


    # transform it to view it in its normal size
    label = int(img_list[0])
    img_vector = torch.Tensor(img_list[1:])
    # visdom takes C x H x W
    # - we only have 1 channel (b/w) so this is fine
    # - H = how many rows, W = how many columns
    # - the data is laid out as row0, then row1, ..., and that seems to be how
    #   view(...) creates the tensor, so this works.
    img_tensor = img_vector.view(1,28,28)
    vis.image(img_tensor, win='demo image', env=visdom_env, opts={
            'caption': 'this should be a {}'.format(label),
    })

    # TODO: view 10 of them in a row

    # TODO: blow it up to view it in bigger size (separate function for this
    # transform)


def main():
    view_train_datum()


if __name__ == '__main__':
    main()
