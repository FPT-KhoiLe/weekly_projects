import torch
from torch import nn

# Basic ffn neuron network
class FFN(nn.Module):
    def __init__(self, hidden_layers: list = [512, 256, 128], out_features: int = 10):
        super().__init__()

        self.neuron_net = nn.Sequential(
            nn.Flatten()
        )

        for i in hidden_layers:
            self.neuron_net.add_module(f"layer_{i}", nn.LazyLinear(out_features=i))
            self.neuron_net.add_module(f"relu_{i}", nn.ReLU(inplace=True))

        self.neuron_net.add_module(f"output", nn.LazyLinear(out_features=out_features))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.neuron_net(x)

    #optional: count number of params to fast print on CLI
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
