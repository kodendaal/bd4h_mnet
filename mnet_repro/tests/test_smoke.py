import torch
from mnet.models.mnet_tiny import MNetTiny

def test_forward():
    net = MNetTiny(1, 2).eval()
    x = torch.randn(1,1,16,64,64)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1,2,16,64,64)
