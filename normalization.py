# imports

# builtins
import code
import csv

# 3rd party
import numpy as np  # just for the epsilon definition
import torch

# local
import constants

# load all data into a single tensor
with open(constants.TRAIN_UNNORM, 'r') as f:
    rows = [r for r in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)]
data = torch.Tensor(rows)

# split off labels and features
labels = data[:, 0]
features = data[:, 1:]

# now, compute per-feature mean/std. dimension is 0 because averaging *along*
# the 0th dimension (data rows). slightly counter-intuitive because we *want*
# averages for dimension 1 (columns), but we specify this by saying to average
# *along* the 0th dimension.
means = features.mean(0)
stds = features.std(0)

# normalize: subtract mean and divide by std. std could be 0 in some
# dimensions, which results in NaN after divison. Counteract by adding epsilon
# to each standard deviation. It will have no effect if the standard deviation
# is any "normal" number, and if it is 0, it will prevent a divide by 0, but
# keep the result 0 (because every element was 0 anyway, so 0/epsilon = 0).
#
# NOTE: I don't totally trust this epsilon because GPU implementations may use
# a different float representation (this is ~2e-16 on my machine). Something
# like 1e-5 would likely work as well. However, none of this is going on the
# GPU, so this is fine for now.
epsilon = np.finfo(float).eps
norm = (features - means) / (stds + epsilon)

# TODO: ensure each entry 1 or 0 and print some kind of check
variances = norm.var(0)

# TODO: do above for val and test, too, and save them. also likely make this
# script not overwrite.
