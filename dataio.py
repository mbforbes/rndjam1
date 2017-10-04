"""
For reading and writing crap.
"""

# imports
# ---

# builtins
import argparse
import csv
from typing import Tuple, List, Dict
import os
import time

# 3rd party
import torch
from tqdm import tqdm

# local
import constants  # for speed test script; not needed for data IO
import viewer


# lib functions
# ---

def which_exist(filenames: List[str]) -> List[str]:
    """
    Returns the subset of filenames that exist on disk.
    """
    return list(filter(lambda f: os.path.exists(f), filenames))


def split_tensor(data: torch.Tensor, label_cols: int = 1) -> Tuple[torch.IntTensor, torch.FloatTensor]:
    """
    Splits 'all data' tensor into labels and features tensors.

    Returns 2-tuple of:
        (1) either a 1d (N) vector (if label_cols == 1),
                or a 2d (N x L) matrix (if label_cols > 1)
        (2) a 2d (N x D) matrix of datums x features
    """
    # have to be careful when selecting one column:
    # - selecting data[:, 0]  gives a 1d (N) vector
    # - selecting data[:, :1] gives a 2d (N x 1) matrix
    if label_cols == 1:
        labels = data[:, 0].type(torch.IntTensor)
    else:
        labels = data[:, :label_cols].type(torch.IntTensor)
    features = data[:, label_cols:]
    return labels, features


def csv_to_tensor(filename: str) -> torch.Tensor:
    """
    Loads all data from filename in csv format; returns in a single tensor.
    """
    with open(filename, 'r') as f:
        rows = [r for r in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)]
    return torch.Tensor(rows)


def csv_to_tensors(
        filename: str) -> Tuple[torch.IntTensor, torch.FloatTensor]:
    """
    Loads data from filename in csv format; return label (col 0) and features
    (rest) tensors.
    """
    return split_tensor(csv_to_tensor(filename))


def tensor_to_csv(t: torch.Tensor, filename: str) -> None:
    """
    Writes tensor t to filename on disk in csv format, creating parent directories as needed.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerows(t)


def tensor_to_bin(t: torch.Tensor, filename: str) -> None:
    """
    Writes tensor t to filename on disk in torch.save(...) format, creating
    directories as needed.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(t, filename)


def bin_to_tensor(filename: str) -> torch.Tensor:
    """
    Loads tensor from disk at filename in torch.save(...) format.
    """
    return torch.load(filename)


def bin_to_tensors(filename: str, label_cols: int = 1) -> Tuple[torch.IntTensor, torch.FloatTensor]:
    """
    Loads data from filename in torch.save(...) format; return label
    (label_cols) and features (rest) tensors.

    Returns 2-tuple of:
        (1) either a 1d (N) vector (if label_cols == 1),
                or a 2d (N x L) matrix (if label_cols > 1)
        (2) a 2d (N x D) matrix of datums x features
    """
    return split_tensor(bin_to_tensor(filename), label_cols)


def bias_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Adds bias column (all 1s) to `t`.

    Arguments:
        t 2D (N x D) tensor

    Retuns:
          2D (N x D+1) tensor, with final bias column (all 1s)
    """
    n = len(t)
    bias_col = torch.ones(n)
    return torch.cat([t, bias_col], dim=1)


def labels_to_onehot(labels: torch.IntTensor, opts: int) -> torch.IntTensor:
    """
    Turns 1d (N) class-label tensor `labels` into 2d (N x opts) onehot tensor
    and returns it.

    Arguments:
        labels: 1d (N) vector of class labels
        opts: number of class label options (will be output cols)

    Returns:
        2d (N x opts) onehot label matrix
    """
    n, = labels.size()
    label_idx = labels.type(torch.LongTensor).view(-1,1)
    return torch.IntTensor(n, opts).zero_().scatter_(1, label_idx, 1)


# script
# --

def speedtest(n: int = 3) -> None:
    """
    Tests speed of loading csv vs torch.load(...) on a tensor. Does `n`
    iterations of tests on several different file sizes. Plots results to
    visdom.
    """
    tests = [
        ('big file, csv', csv_to_tensor, constants.TRAIN_NORM),
        ('big file, tensor', bin_to_tensor, constants.TRAIN_TENSOR),
        ('small file, csv', csv_to_tensor, constants.VAL_NORM),
        ('small file, tensor', bin_to_tensor, constants.VAL_TENSOR),
    ]
    results = {}  # type: Dict[str, List[float]]
    for desc, fn, arg in tqdm(tests):
        results[desc] = []
        for i in range(n):
            start = time.perf_counter()
            res = fn(arg)
            results[desc].append(time.perf_counter() - start)
            del res
    viewer.plot_jitter(results, 'Data Loading Speeds')


def convert() -> None:
    """
    Converts files from this project from csv to torch.save(...) format.
    """
    worklist = [
        (constants.TRAIN_NORM, constants.TRAIN_TENSOR),
        (constants.VAL_NORM, constants.VAL_TENSOR),
        (constants.TEST_NORM, constants.TEST_TENSOR),
    ]

    # pre-check: don't convert if any dest. files exist
    existing = which_exist([out for inp, out in worklist])
    if len(existing) > 0:
        print('ERROR: Not converting because the following files already '
            'exist: {}'.format(existing))
        return

    print('dataio.convert :: start')
    for csv_fn, bin_fn in worklist:
        print('\t Converting {} to {}...'.format(csv_fn, bin_fn))
        # This is the actual line of code that does the conversion.
        tensor_to_bin(csv_to_tensor(csv_fn), bin_fn)
    print('dataio.convert :: finish')


def bias() -> None:
    """
    Adds bias term to files from this project (output from convert()).
    """
    worklist = [
        (constants.TRAIN_TENSOR, constants.TRAIN_BIAS),
        (constants.VAL_TENSOR, constants.VAL_BIAS),
        (constants.TEST_TENSOR, constants.TEST_BIAS),
    ]

    existing = which_exist([out for inp, out in worklist])
    if len(existing) > 0:
        print('ERROR: Not adding bias because the following files already '
            'exist: {}'.format(existing))
        return

    print('dataio.bias :: start')
    for tensor_fn, bias_fn in worklist:
        print('\t Converting {} to {}...'.format(tensor_fn, bias_fn))
        # This is the actual line of code that does the biasing.
        tensor_to_bin(bias_tensor(bin_to_tensor(tensor_fn)), bias_fn)
    print('dataio.bias :: finish')


def onehot() -> None:
    """
    Transform labels to onehot for files from this project (output from
    bias()).
    """
    worklist = [
        (constants.TRAIN_BIAS, constants.TRAIN_ONEHOT),
        (constants.VAL_BIAS, constants.VAL_ONEHOT),
        (constants.TEST_BIAS, constants.TEST_ONEHOT),
    ]

    existing = which_exist([out for inp, out in worklist])
    if len(existing) > 0:
        print('ERROR: Not creating onehots because the following files '
            'already exist: {}'.format(existing))
        return

    print('dataio.onehot :: start')
    for bias_fn, onehot_fn in worklist:
        print('\t Converting {} to {}...'.format(bias_fn, onehot_fn))
        # This is the actual code that does the onehot'ing.
        class_labels, data = bin_to_tensors(bias_fn)
        onehot_labels = labels_to_onehot(class_labels, 10).type(torch.FloatTensor)
        tensor_to_bin(torch.cat([onehot_labels, data], dim=1), onehot_fn)
    print('dataio.onehot :: finish')


def main() -> None:
    """
    Runs a speed test of reading CSV vs torch.load(...)
    """
    parser = argparse.ArgumentParser(description='Data stuff.')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--speedtest', action='store_true')
    choice.add_argument('--convert', action='store_true')
    choice.add_argument('--bias', action='store_true')
    choice.add_argument('--onehot', action='store_true')
    args = parser.parse_args()
    if args.speedtest:
        speedtest()
    if args.convert:
        convert()
    if args.bias:
        bias()
    if args.onehot:
        onehot()


if __name__ == '__main__':
    main()
