import torch
from weekly_projects.models.ffn import FFN

def test_ffn_forward():
    model = FFN([256, 256])
    out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)
