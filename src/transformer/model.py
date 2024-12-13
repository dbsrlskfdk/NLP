import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(
        self,
    ):
        pass

    def forward(
        self,
    ):
        pass


class SDPSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
    ):
        super(SDPSelfAttention, self).__init__()
        self.q = nn.Linear(
            d_model, d_k
        )  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k)
        self.k = nn.Linear(
            d_model, d_k
        )  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k)
        self.v = nn.Linear(
            d_model, d_v
        )  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_v)
        self.d_k = d_k

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, seq_len, d_model)
    ):
        attn_scores = torch.matmul(
            self.q(x), self.k(x).transpose(1, 2)
        )  # (batch_size, seq_len, seq_len)
        scaled_scores = attn_scores / (self.d_k**0.5)  # (batch_size, seq_len, seq_len)
        attn_probs = F.softmax(scaled_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        x = torch.matmul(
            attn_probs, self.v(x)
        )  # (batch_size, seq_len, seq_len) * (batch_size, seq_len, d_v) -> (batch_size, seq_len, d_v)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int = 6,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
    ):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.attention_heads = nn.ModuleList(
            [SDPSelfAttention(d_model, d_k, d_v) for _ in range(n_head)]
        )
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, seq_len, d_model)
    ):
        self_attention_outputs = [
            attention_head(x) for attention_head in self.attention_heads
        ]  # [(batch_size, seq_len, d_v)] * n_head
        self_attention_outputs = torch.cat(self_attention_outputs, dim=-1)
        x = self.fc(self_attention_outputs)
        x = self.layer_norm(x)

        return x
