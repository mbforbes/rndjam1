import torch

i = torch.IntTensor()
f = torch.FloatTensor()
reveal_type(i)
reveal_type(f)

def foo(i: torch.IntTensor, f: torch.FloatTensor) -> None:
    pass

# these behave as expected
foo(i, f)
foo(1, 2)

# these typecheck OK, but I think should error
foo(i, i)
foo(f, f)
