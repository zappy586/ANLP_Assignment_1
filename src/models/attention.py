import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, num_heads, model_dim):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        
        self.q = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.k = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.v = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.ff = nn.Linear(in_features=model_dim, out_features=model_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q_heads = q.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_heads = k.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_heads = v.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_scores = torch.softmax((q_heads @ torch.transpose(k_heads, 3, 2)) / self.head_dim ** (1/2), dim=-1)
        print(attention_scores.shape, attention_scores)
        self_attention = attention_scores @ v_heads

        mha_output = self_attention.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.model_dim)
        output = self.ff(mha_output)
        return output

class GQA(nn.Module):
    def __init__(self, num_query_heads, num_kv_heads, model_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_query_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.num_kv_heads = num_kv_heads

        self.q = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.k = nn.Linear(in_features=model_dim, out_features=num_kv_heads * self.head_dim)
        self.v = nn.Linear(in_features=model_dim, out_features=num_kv_heads * self.head_dim)
        self.ff = nn.Linear(in_features=model_dim, out_features=model_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q_heads = q.reshape(x.shape[0], x.shape[1], self.num_query_heads, self.head_dim).permute(0, 2, 1, 3)        

        
        k_heads = k.reshape(x.shape[0], x.shape[1], self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v_heads = v.reshape(x.shape[0], x.shape[1], self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # inflating the kv heads by consecutively copying the tensors along the num_kv_heads dimension, 
        # to match the query tensor shape, so that groups of queries get the same kv tensors
        k_heads = torch.repeat_interleave(k_heads, self.num_queries_per_kv, dim=-3)
        v_heads = torch.repeat_interleave(v_heads, self.num_queries_per_kv, dim=-3)

        attention_scores = torch.softmax((q_heads @ torch.transpose(k_heads, 3, 2)) / self.head_dim ** (1/2), dim=-1)
        self_attention = attention_scores @ v_heads

        mha_output = self_attention.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.model_dim)
        output = self.ff(mha_output)
        return output





# test_mha = MHA(10, 60)

# print(f"MHA output: {test_mha(torch.randn((1, 13, 60))).shape}")

# test = torch.tensor([1, 2, 3, 4, 5])
# print(torch.repeat_interleave(test, 2))

# test_gqa = GQA(10, 5, 60)

# print(f"GQA output : {test_gqa(torch.randn((1, 13, 60))).shape}")

        



        