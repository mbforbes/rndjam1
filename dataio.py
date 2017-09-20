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


def csv_to_tensor(filename: str) -> torch.Tensor:
    """
    Loads all data from filename; returns in a single tensor.
    """
    with open(filename, 'r') as f:
        rows = [r for r in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)]
    return torch.Tensor(rows)


def csv_to_tensors(
        filename: str) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Loads data from filename; return label (col 0) and features (rest) tensors.
    """
    data = csv_to_tensor(filename)

    # split off labels and features
    labels = data[:, 0].type(torch.IntTensor)
    features = data[:, 1:]

    return labels, features


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
        tensor_to_bin(csv_to_tensor(csv_fn), bin_fn)
    print('dataio.convert :: finish')


def main() -> None:
    """
    Runs a speed test of reading CSV vs torch.load(...)
    """
    parser = argparse.ArgumentParser(description='Data stuff.')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--speedtest', action='store_true')
    choice.add_argument('--convert', action='store_true')
    args = parser.parse_args()
    if args.speedtest:
        speedtest()
    if args.convert:
        convert()


if __name__ == '__main__':
    main()
