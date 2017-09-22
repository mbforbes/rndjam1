from typing import Tuple

import torch


def foo() -> Tuple[int, str]:
    return 5, 'hello'

a, b = foo()
reveal_type(a)
reveal_type(b)

def baz() -> Tuple[torch.IntTensor, torch.FloatTensor]:
    return torch.IntTensor(1), torch.FloatTensor(1.0)

c, d = baz()
reveal_type(c)
reveal_type(d)
