#!/bin/bash
mkdir -p data/processed/resplit/
head -n 50000 data/original/mnist_train.csv > data/processed/resplit/mnist_train.csv
tail -n 10000 data/original/mnist_train.csv > data/processed/resplit/mnist_val.csv
cp data/original/mnist_test.csv data/processed/resplit/mnist_test.csv
