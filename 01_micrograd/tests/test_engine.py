import torch
from micrograd.engine import Value


def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()    
    xpt, ypt = x, y

    assert ymg.data == ypt.data.item()
    assert xmg.grad == xpt.grad.item()

    assert 1 ==1

