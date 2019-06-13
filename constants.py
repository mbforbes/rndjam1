"""
Per project constants.

These aren't settings (like hyparameters) but per-project things that a user might want
to change.
"""

# data processing stages. Effects cumulative (e.g., normalized is also resplit):
# - (1) original   (downloaded)
# - (2) resplit    (pulled off some of train for val)
# - (3) normalized (zero mean, unit variance per feature)
# - (4) tensor     (using torch.save(...) instead of a CSV format)
# - (5) bias       (adding bias column)
# - (6) onehot     (turning labels from numbers to onehots)

# filenames
TRAIN_RESPLIT = "data/processed/resplit/mnist_train.csv"
VAL_RESPLIT = "data/processed/resplit/mnist_val.csv"
TEST_RESPLIT = "data/processed/resplit/mnist_test.csv"

TRAIN_NORM = "data/processed/normalized/mnist_train.csv"
VAL_NORM = "data/processed/normalized/mnist_val.csv"
TEST_NORM = "data/processed/normalized/mnist_test.csv"

TRAIN_TENSOR = "data/processed/tensor/mnist_train.tensor"
VAL_TENSOR = "data/processed/tensor/mnist_val.tensor"
TEST_TENSOR = "data/processed/tensor/mnist_test.tensor"

TRAIN_BIAS = "data/processed/bias/mnist_train.tensor"
VAL_BIAS = "data/processed/bias/mnist_val.tensor"
TEST_BIAS = "data/processed/bias/mnist_test.tensor"

TRAIN_ONEHOT = "data/processed/onehot/mnist_train.tensor"
VAL_ONEHOT = "data/processed/onehot/mnist_val.tensor"
TEST_ONEHOT = "data/processed/onehot/mnist_test.tensor"

# visdom
VISDOM_ENV = "rndj1"
