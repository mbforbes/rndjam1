# imports

# builtins
import code
import csv

# 3rd party
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

# now, compute per-feature norms dimension is 0 because averaging *along* the
# 0th dimension (data rows). slightly counter-intuitive because we *want*
# averages for dimension 1 (columns), but we specify this by saying to average
# *along* the 0th dimension.
means = features.mean(0)
stds = features.std(0)
variances = features.var(0)

# TODO: what to do with this! how are you "supposed" to normalize data? 0-mean
# and unit variance, right?
code.interact(local=dict(globals(), **locals()))
