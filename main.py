import torch
import visdom

x = torch.rand(50, 100).type(torch.FloatTensor)
print(x)
y = torch.rand(50, 100).type(torch.cuda.FloatTensor)
print(y)

vis = visdom.Visdom()
vis.text('Hello from python3.6.2!', env='rndj1', win='testmsg')
# open browser to http://localhost:8097/env/rndj1
