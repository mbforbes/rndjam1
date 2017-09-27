#!/bin/bash

# exit on error
set -e

# convert all tex files in math directory to equivalently-named svg files in
# the svg directory
for filename in math/*.tex; do
    bname=$(basename ${filename} .tex)
    ./node_modules/mathjax-node-cli/bin/tex2svg "$(< ${filename})" > svg/${bname}.svg
done
