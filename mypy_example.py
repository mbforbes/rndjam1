from typing import Tuple

import torch


def foo() -> Tuple[int, str]:
    return 5, 'hello'

a, b = foo()
reveal_type(a)
reveal_type(b)

def bar() -> Tuple[torch.IntTensor, torch.FloatTensor]:
    return torch.IntTensor(1), torch.FloatTensor(1.0)

c, d = bar()
reveal_type(c)
reveal_type(d)


def baz(it: torch.IntTensor, ft: torch.FloatTensor) -> None:
    pass

baz(c, d)  # should be ok
baz(1, 2)  # should be bad
baz(c, c)  # should be bad
baz(d, d)  # should be bad
