import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.beta = nn.Parameter(torch.zeros((self.model_dim)))
        self.gamma = nn.Parameter(torch.ones((self.model_dim)))

    def forward(self, x):
        eps = 1e-5
        var, u = torch.var_mean(x, dim=-1, keepdim=True)
        sigma = (var + eps) ** (1/2)
        y = (x - u) / sigma
        output = self.gamma * y + self.beta
        return output

class RMSNorm(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.gamma = nn.Parameter(torch.ones(model_dim))
        
    def forward(self, x):
        eps = 1e-5
        rms = (((torch.sum(torch.square(x), dim=-1, keepdim=True)) / self.model_dim) + eps) ** (1/2)
        norm = self.gamma * (x / rms)
        return norm

# test_ln = LayerNorm(model_dim=512)
# test_tensor = torch.randn([1, 20, 512])
# output = test_ln(test_tensor)
# print(output)

# test_tensor = torch.randn([1, 20, 512])
# square = torch.square(test_tensor)
# sumtorch = torch.sum(test_tensor, dim=-1)
# print(sumtorch, sumtorch.shape)