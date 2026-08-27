import torch
from torch import nn

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, model_dim, N=10000):
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

class MHA_RoPE(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        
        self.q = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.k = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.v = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.out_proj = nn.Linear(in_features=model_dim, out_features=model_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q_heads = q.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, N, H, Dk] -> [B, H, N, Dk]
        k_heads = k.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_heads = v.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # RoPE Logic starts from here
        q_heads_rope = q_heads.chunk(2, dim=-1)
        k_heads_rope = k_heads.chunk(2, dim=-1)

        i = torch.arange(self.head_dim // 2, device=x.device, dtype=x.dtype)
        m = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(-1)
        base_freq = 10000.0 ** (-2.0 * i / self.head_dim)
        theta = m * base_freq

        # rotating the qk vectors
        q1_rot = q_heads_rope[0] * torch.cos(theta) - q_heads_rope[1] * torch.sin(theta)
        q2_rot = q_heads_rope[0] * torch.sin(theta) + q_heads_rope[1] * torch.cos(theta)

        k1_rot = k_heads_rope[0] * torch.cos(theta) - k_heads_rope[1] * torch.sin(theta)
        k2_rot = k_heads_rope[0] * torch.sin(theta) + k_heads_rope[1] * torch.cos(theta) 

        # concatenating them
        q_heads = torch.cat((q1_rot, q2_rot), dim=-1)
        k_heads = torch.cat((k1_rot, k2_rot), dim=-1)

        attention_scores = torch.softmax((q_heads @ torch.transpose(k_heads, 3, 2)) / self.head_dim ** (1/2), dim=-1)
        self_attention = attention_scores @ v_heads

        mha_output = self_attention.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.model_dim)
        output = self.out_proj(mha_output)
        return output




# # t = torch.arange(25).unsqueeze(1)
# # N = 10000
# # k_sin = torch.arange(0, 512, step=2)
# # d = 512
# # test = torch.sin(t/(N ** (k_sin/d)))
# # print(test, test.shape)

# # test = torch.arange(20)
# # print(test.split(2))

# test = torch.randn([1, 14, 512])
# print(test.reshape([1, 14, ]))

# q1, q2 = torch.rand((1, 12, 15, 128)).split(int(128/2), dim=-1) 
# t = 15
# i = torch.arange(int(128/2))
# m = torch.arange(t).unsqueeze(-1)
# base_freq = 10000 ** (-2 * i/128)
# phi = m * base_freq

# q1_rot = q1 * torch.cos(phi) - q2 * torch.sin(phi)
# q2_rot = q1 * torch.sin(phi) + q2 * torch.cos(phi)

# q = torch.cat((q1_rot, q2_rot), dim=-1)

# print(m.shape, base_freq.shape, phi.shape, q.shape)