import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientAttention(nn.Module):
    """Efficient Attention: Attention with Linear Complexities (Shen et al., WACV 2021).

    Paper: https://arxiv.org/abs/1812.01243
    Code: https://github.com/cmsflash/efficient-attention
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None):
        # q, k, v: [L, N, D]
        Lq, N, _ = q.shape
        Lk, _, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(Lq, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # [N, H, Lq, d]
        k = k.view(Lk, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # [N, H, Lk, d]
        v = v.view(Lk, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # [N, H, Lk, d]

        # Apply softmax separately on K and Q
        k = k.softmax(dim=-2)  # softmax over sequence length
        v_prime = torch.einsum('nhld,nhle->nhde', k, v)  # [N, H, d, d]

        q = q.softmax(dim=-1)  # softmax over feature dimension
        attn_output = torch.einsum('nhld,nhde->nhle', q, v_prime)  # [N, H, Lq, d]

        attn_output = attn_output.permute(2, 0, 1, 3).reshape(Lq, N, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None


class LinearAttention(nn.Module):
    """Linear Attention from "Transformers are RNNs" (Katharopoulos et al., ICML 2020).

    Paper: https://arxiv.org/abs/2006.16236
    Code: https://github.com/idiap/fast-transformers
    """

    def __init__(self, embed_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.eps = eps

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _feature_map(x):
        return F.elu(x) + 1

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None):
        # q, k, v: [L, N, D]
        Lq, N, _ = q.shape
        Lk, _, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(Lq, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # [N, H, Lq, d]
        k = k.view(Lk, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # [N, H, Lk, d]
        v = v.view(Lk, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # [N, H, Lk, d]

        q = self._feature_map(q)
        k = self._feature_map(k)

        k_sum = k.sum(dim=2)  # [N, H, d]
        kv = torch.einsum('nhld,nhle->nhde', k, v)  # [N, H, d, d]
        z = 1.0 / (torch.einsum('nhld,nhd->nhl', q, k_sum) + self.eps)  # [N, H, Lq]
        attn_output = torch.einsum('nhld,nhde->nhle', q, kv)  # [N, H, Lq, d]
        attn_output = attn_output * z.unsqueeze(-1)

        attn_output = attn_output.permute(2, 0, 1, 3).reshape(Lq, N, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None
