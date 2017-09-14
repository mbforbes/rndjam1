#!/bin/bash
mkdir -p data/original/
curl https://pjreddie.com/media/files/mnist_train.csv > data/original/mnist_train.csv
curl https://pjreddie.com/media/files/mnist_test.csv > data/original/mnist_test.csv
