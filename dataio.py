"""
For reading and writing crap.
"""

# imports
# ---

# builtins
import argparse
import csv
from typing import Tuple, List
import os

# 3rd party
import torch

# local
import constants  # for speed test script; not needed for data IO


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
    Writes tensor t to filename on disk, creating parent directories as needed.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerows(t)


# script
# --

def speedtest() -> None:
    """
    Tests speed of loading csv vs torch.load(...) on a tensor. Reports results.
    """
    # TODO: this
    pass


def convert() -> None:
    """
    Converts files from this project from csv to torch.save(...) format.
    """
    # TODO: this
    pass


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
        print('speedtest')
    if args.convert:
        print('convert')


if __name__ == '__main__':
    main()
