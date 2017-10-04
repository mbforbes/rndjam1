# imports
# ---

# builtins
import code

# 3rd party
import torch
import torch.nn.functional as F
import visdom

# local
import viewer

# test visdom basics
x = torch.rand(50, 100).type(torch.FloatTensor)
print(x)
y = torch.rand(50, 100).type(torch.cuda.FloatTensor)
print(y)

vis = visdom.Visdom()
vis.text('Hello from python3.6.2!', env='rndj1', win='testmsg')
# open browser to http://localhost:8097/env/rndj1


# test softshrink = "soft threshold"
lmb = 0.5
xs = torch.arange(-5, 5, 0.1)
ys = torch.FloatTensor(2, len(xs))
ys[0] = xs
ys[1] = F.softshrink(xs, lmb).data
# code.interact(local=dict(globals(), **locals()))
viewer.plot_line(xs, ys, ['y = x', 'y = s(x,{})'.format(lmb)], 'test-soft-thresh')
