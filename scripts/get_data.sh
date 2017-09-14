#!/bin/bash
mkdir data/
curl https://pjreddie.com/media/files/mnist_train.csv > data/mnist_train.csv
curl https://pjreddie.com/media/files/mnist_test.csv > data/mnist_test.csv
