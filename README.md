# R&Djam1


## Goal

**Build** linear and logistic regression for MNIST from scratch using pytorch.


## Running

python3. Setup:

```bash
# pyenv wraps virtualenv to let you pick a python version. global namespace.
pyenv virtualenv 3.6.2 rndj1
pyenv local rndj1

# pytroch. (get latest from pytorch.org. this is what i used.)
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
pip install torchvision

# dashboard
pip install visdom
# visdom should already be running. if not, run with: `python -m visdom.server`

# editor setup. for vscode:
pip install prospector
# in vs code, use command `Python: Select Workspace Interpreter` and pick rndj1

# get data. writes to data/original/
./scripts/get_data.sh

# split data. writes to data/processed/resplit/
./scripts/split_data.sh

# normalize data. writes to data/processed/normalized
python normalization.py
```


## Data splits

MNIST ([csv version][mnist-csv]) has a 60k/10k train/test split.

I pulled the last 10k off of train for a val set.

My final splits are then 50k/10k/10k train/val/test.

[mnist-csv]: https://pjreddie.com/projects/mnist-in-csv/


## Viewing an image

Here's an MNIST image:

![the first mnist datum](images/example_normal.jpg)

Here it is expanded 10x:

![the first mnist datum, expanded](images/example_bloated.jpg)
