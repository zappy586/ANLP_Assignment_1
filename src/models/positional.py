import torch
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, model_dim, N):
        super().__init__()
        self.d = model_dim
        self.N = N
        self.register_buffer('k', torch.arange(0, model_dim, step=2))

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).unsqueeze(1)
        sin = torch.sin(t/self.N ** (self.k/self.d))
        cos = torch.cos(t/self.N ** (self.k/self.d))
        total_angles = torch.cat([sin, cos], dim=-1).unsqueeze(0)
        return x + total_angles





# t = torch.arange(25).unsqueeze(1)
# N = 10000
# k_sin = torch.arange(0, 512, step=2)
# d = 512
# test = torch.sin(t/(N ** (k_sin/d)))
# print(test, test.shape)
