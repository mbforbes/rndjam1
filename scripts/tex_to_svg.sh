#!/bin/bash

# convert all tex files in math directory to equivalently-named svg files in
# the svg directory
for filename in math/*.tex; do
    # just echo'ing so we can see where errors happen when they do
    echo ${filename}
    bname=$(basename ${filename} .tex)
    ./node_modules/mathjax-node-cli/bin/tex2svg "$(< ${filename})" > svg/${bname}.svg &
done
