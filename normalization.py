"""
Handles per-feature normalization to zero mean and unit variance.
"""

# imports
# ---

# builtins
import code
import csv
from typing import List, Tuple
import os

# 3rd party
import numpy as np  # for the epsilon definition and float comparison
import torch

# local
import constants
import dataio

# constants
# ---

# practically defined; this is as close as we can expect for checking float eq
# when normalizing.
CHECK_EPSILON = 1e-5


# lib functions
# ---

def normalize(
        features: torch.FloatTensor, means: torch.FloatTensor,
        stds: torch.FloatTensor, check: bool = False) -> torch.Tensor:
    """
    Arguments:
        features: N x D
        means: D
        stds: D
        check: whether to ensure that means are 0 and variances are 1 (or 0, if
            all data 0). Only makes sense for train (on which data comptued).

    Returns:
        N x D
    """
    # normalize: subtract mean and divide by std. std could be 0 in some
    # dimensions, which results in NaN after divison. Counteract by adding
    # epsilon to each standard deviation. It will have no effect if the
    # standard deviation is any "normal" number, and if it is 0, it will
    # prevent a divide by 0, but keep the result 0 (because every element was 0
    # anyway, so 0/epsilon = 0).
    #
    # NOTE: I don't totally trust this epsilon because GPU implementations may
    # use a different float representation (this is ~2e-16 on my machine).
    # Something like 1e-5 would likely work as well. However, none of this is
    # going on the GPU, so this is fine for now.
    epsilon = np.finfo(float).eps
    norm = (features - means) / (stds + epsilon)

    # only makes sense to check on train (as that's where the means/stds come
    # from)
    if check:
        # ensure each mean 0
        new_means = norm.mean(0)
        for i, m in enumerate(new_means):
            if not np.isclose(0.0, m, atol=CHECK_EPSILON):
                print('ERROR: Dimension {} has mean {}, wanted 0.0'.format(
                    i, m
                ))

        # ensure each variance entry 1 or 0
        variances = norm.var(0)
        for i, v in enumerate(variances):
            if not (np.isclose(0.0, v, atol=CHECK_EPSILON) or
                    np.isclose(1.0, v, atol=CHECK_EPSILON)):
                print(
                    'ERROR: Dimension {} has variance {},'
                    'wanted 0.0 or 1.0'.format(
                        i, v
                    )
                )

    # give normalized features
    return norm


# script functions
# ---

def normalize_and_save(
        labels: torch.Tensor, features: torch.Tensor, out_fn: str,
        means: torch.Tensor, stds: torch.Tensor,
        check: bool = False) -> None:
    """
    Helper
    """
    norm = normalize(features, means, stds, check)
    result = torch.Tensor(norm.size()[0], norm.size()[1] + 1)
    result[:, 0] = labels
    result[:, 1:] = norm
    dataio.tensor_to_csv(result, out_fn)


def normalize_data(
        train: Tuple[str, str], worklist: List[Tuple[str, str]]) -> None:
    train_unnorm_fn, train_norm_fn = train

    train_labels, train_unnorm_features = dataio.csv_to_tensors(
        train_unnorm_fn)

    # now, compute per-feature mean/std. dimension is 0 because averaging
    # *along* the 0th dimension (data rows). slightly counter-intuitive because
    # we *want* averages for dimension 1 (columns), but we specify this by
    # saying to average *along* the 0th dimension.
    means = train_unnorm_features.mean(0)
    stds = train_unnorm_features.std(0)

    # normalize and save train
    normalize_and_save(
        train_labels, train_unnorm_features, train_norm_fn, means, stds, True)

    # normalize others (probably val and test)
    for raw_fn, norm_fn in worklist:
        normalize_and_save(
            *dataio.csv_to_tensors(raw_fn), norm_fn, means, stds, False)


def main() -> None:
    # we provide tuples of: (
    #     where the unnormalized file exists,
    #     where we want the normalized file to go
    # )
    # train is special because (a) it's used to define the normalizeation, (b)
    # we don't want to load it twice.
    train = (constants.TRAIN_RESPLIT, constants.TRAIN_NORM)
    worklist = [
        (constants.VAL_RESPLIT, constants.VAL_NORM),
        (constants.TEST_RESPLIT, constants.TEST_NORM),
    ]

    # to be the safest, we'll check if any of the normalized files exist and
    # not overwrite them if so.
    existing = dataio.which_exist([train[1]] + [w[1] for w in worklist])
    if len(existing) > 0:
        print('ERROR: The following normalized files already exist: {}'.format(
            existing))
        return

    # now we can actually normalize
    normalize_data(train, worklist)


if __name__ == '__main__':
    main()
