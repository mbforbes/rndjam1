# R&Djam1

## Goal

**Build** linear and logistic regression for MNIST from scratch using pytorch.

## Running

python3. Setup:

```
# pyenv wraps virtualenv to let you pick a python version. global namespace.
pyenv virtualenv 3.6.2 rndj1
pyenv local rndj1
# get latest from pytorch.org
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
pip install torchvision
pip install visdom
# visdom should already be running. if it's not, run with:
# `python -m visdom.server`
# for vscode:
pip install prospector
# in vs code, use command `Python: Select Workspace Interpreter` and pick rndj1
```
