from typing import Type, Tuple

import torch


def foo() -> Tuple[int, str]:
    return 5, 'hello'

a, b = foo()
reveal_type(a)  # should be int, is int
reveal_type(b)  # should be str, is str

def bar() -> Tuple[torch.IntTensor, torch.FloatTensor]:
    return torch.IntTensor(1), torch.FloatTensor(1.0)

c, d = bar()
reveal_type(c)  # should be torch.IntTensor, is torch.IntTensor
reveal_type(d)  # should be torch.FloatTensor, is torch.FloatTensor

def baz(it: torch.IntTensor, ft: torch.FloatTensor) -> None:
    pass

baz(c, d)  # should be ok
baz(1, 2)  # should be bad, is bad
baz(c, c)  # should be bad, is ok  <-- current problem
baz(d, d)  # should be bad, is ok  <-- current problem
