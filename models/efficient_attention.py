# efficient_attention.py
# Collects efficient attention variants with consistent API and mask semantics.
# Masks follow PyTorch semantics: True = mask/ignore; False = keep.

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _to_key_padding_mask(attn_mask: Optional[torch.Tensor],
                         B: int, Tq: int, Tk: int) -> Optional[torch.Tensor]:
    """
    Normalize various mask shapes into a key-padding mask [B, Tk] with True = ignore.

    Accepted shapes:
      - [B, Tk]                 : already a key padding mask
      - [Tq, Tk]                : reduced by AND across queries -> [B, Tk] (broadcast to batch)
      - [B, Tq, Tk]             : reduced by AND across queries -> [B, Tk]
    """
    if attn_mask is None:
        return None
    if attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(torch.bool)
    if attn_mask.dim() == 2 and attn_mask.shape == (B, Tk):
        return attn_mask
    if attn_mask.dim() == 2 and attn_mask.shape == (Tq, Tk):
        return attn_mask.all(dim=0, keepdim=True).expand(B, -1)
    if attn_mask.dim() == 3 and attn_mask.shape == (B, Tq, Tk):
        return attn_mask.all(dim=1)
    raise ValueError(
        f"Unsupported attn_mask shape {tuple(attn_mask.shape)}. "
        f"Use [B,Tk], [Tq,Tk], or [B,Tq,Tk]."
    )


# ------------------------------------------------------------
# 1) Efficient Attention (Shen et al., WACV'21)
#     Paper: https://arxiv.org/abs/1812.01243
#     Code : https://github.com/cmsflash/efficient-attention
# ------------------------------------------------------------

class EfficientAttention(nn.Module):
    """
    Efficient Attention (Shen et al. 2018/2021 WACV) with learnable projections.
    Works on [B, T, C] when batch_first=True, else [T, B, C].

    forward args:
      q, k, v : [B,T,C] if batch_first else [T,B,C]
      attn_mask: PyTorch-style (True = ignore). Supports [B,Tk], [Tq,Tk], or [B,Tq,Tk].
      need_weights: "factors" -> return normalized (Q_norm, K_norm)
                    "full"    -> return full quadratic weights [B,H,Tq,Tk]
    """
    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 value_channels: int,
                 num_heads: int,
                 out_channels: Optional[int] = None,
                 dropout_p: float = 0.0,
                 batch_first: bool = False):
        super().__init__()
        assert key_channels % num_heads == 0, "key_channels must be divisible by num_heads"
        assert value_channels % num_heads == 0, "value_channels must be divisible by num_heads"

        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.q_proj = nn.Linear(in_channels, key_channels, bias=True)
        self.k_proj = nn.Linear(in_channels, key_channels, bias=True)
        self.v_proj = nn.Linear(in_channels, value_channels, bias=True)
        self.out_proj = nn.Linear(value_channels, out_channels or in_channels, bias=True)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.head_dim_k = key_channels // num_heads
        self.head_dim_v = value_channels // num_heads

    def _to_bhtd(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # [B,T,C] -> [B,H,T,D]
        B, T, C = x.shape
        H = self.num_heads
        D = dim // H
        return x.view(B, T, H, D).permute(0, 2, 1, 3).contiguous()

    def _from_bhtd(self, x: torch.Tensor) -> torch.Tensor:
        # [B,H,T,D] -> [B,T,H*D]
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: Optional[str] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # to [B,T,C]
        if not self.batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        B, Tq, Cq = q.shape
        Bk, Tk, Ck = k.shape
        Bv, Tv, Cv = v.shape
        assert B == Bk == Bv and Tk == Tv and Cq == Ck == Cv == self.in_channels

        # learned projections
        Q = self.q_proj(q)  # [B,Tq,Kc]
        K = self.k_proj(k)  # [B,Tk,Kc]
        V = self.v_proj(v)  # [B,Tk,Vc]

        # split heads
        Q = self._to_bhtd(Q, self.key_channels)    # [B,H,Tq,Dk]
        K = self._to_bhtd(K, self.key_channels)    # [B,H,Tk,Dk]
        V = self._to_bhtd(V, self.value_channels)  # [B,H,Tk,Dv]

        # logits to normalized forms
        # K_norm softmax over positions, Q_norm softmax over channels
        K_logits = K.permute(0, 1, 3, 2).contiguous()  # [B,H,Dk,Tk]
        Q_logits = Q.permute(0, 1, 3, 2).contiguous()  # [B,H,Dk,Tq]

        # key padding mask (True=ignore) over positions
        key_padding_mask = _to_key_padding_mask(attn_mask, B, Tq, Tk)
        if key_padding_mask is not None:
            # expand to [B,1,1,Tk]
            pos_mask = key_padding_mask[:, None, None, :]
            K_logits = K_logits.masked_fill(pos_mask, float("-inf"))

        K_norm = F.softmax(K_logits, dim=-1)  # over Tk
        Q_norm = F.softmax(Q_logits, dim=2)   # over Dk
        K_norm = self.dropout(K_norm)
        Q_norm = self.dropout(Q_norm)

        # efficient compute
        context = torch.matmul(K_norm, V)                               # [B,H,Dk,Dv]
        out_t = torch.matmul(context.transpose(-2, -1), Q_norm)         # [B,H,Dv,Tq]
        out = out_t.permute(0, 1, 3, 2).contiguous()                    # [B,H,Tq,Dv]

        out = self._from_bhtd(out)                                      # [B,Tq,Vc]
        out = self.out_proj(out)                                        # [B,Tq,out]
        if not self.batch_first:
            out = out.transpose(0, 1)

        attn = None
        if need_weights == "factors":
            attn = (
                Q_norm,   # [B,H,Dk,Tq]
                K_norm    # [B,H,Dk,Tk]
            )
        elif need_weights == "full":
            A = torch.matmul(Q_norm.transpose(-2, -1), K_norm)          # [B,H,Tq,Tk]
            if key_padding_mask is not None:
                A = A.masked_fill(key_padding_mask[:, None, None, :], 0.0)
            attn = A

        return out, attn


# ------------------------------------------------------------
# 2) Performer-style Fast Attention (ICLR'21)
#     Paper: https://arxiv.org/abs/2009.14794
#     Code : https://github.com/google-research/google-research/tree/master/performer/fast_attention
# ------------------------------------------------------------

def _gaussian_random_matrix(rows: int, cols: int, device=None, dtype=None):
    return torch.randn(rows, cols, device=device, dtype=dtype)

class FastAttention(nn.Module):
    """
    Fast softmax attention with random feature maps (nonnegative FAVOR+).
    Shapes: q,k,v -> [B,H,T,D]
    """
    def __init__(
        self,
        dim: int,
        nb_features: int = 256,
        ortho_features: bool = False,
        redraw_features: bool = False,
        causal: bool = False,      # NOTE: bidirectional in this minimal version
        renormalize: bool = True,
        nonnegative_features: bool = True,
        eps: float = 1e-6,
        device=None, dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.m = nb_features
        self.ortho = ortho_features
        self.redraw = redraw_features
        self.causal = causal
        self.renorm = renormalize
        self.nonneg = nonnegative_features
        self.eps = eps
        self.register_buffer("proj", self._draw_projection(device=device, dtype=dtype), persistent=False)

    def _draw_projection(self, device=None, dtype=None):
        # (Optionally implement orthogonal features if desired.)
        return _gaussian_random_matrix(self.m, self.dim, device=device, dtype=dtype)

    @torch.no_grad()
    def redraw_projection(self):
        self.proj = self._draw_projection(device=self.proj.device, dtype=self.proj.dtype)

    def _phi_nonnegative(self, x: torch.Tensor, is_query: bool) -> torch.Tensor:
        """
        Nonnegative softmax kernel features (JAX reference style).
        x: [B,H,T,D] -> returns [B,H,T,M]
        """
        B, H, T, D = x.shape
        proj = self.proj  # [M,D]
        data_normalizer = 1.0 / math.sqrt(math.sqrt(D))
        ratio = 1.0 / math.sqrt(self.m)

        x_proj = torch.einsum("bhtd,md->bhtm", x * data_normalizer, proj)   # [B,H,T,M]
        diag = 0.5 * (data_normalizer ** 2) * (x * x).sum(dim=-1, keepdim=True)  # [B,H,T,1]
        if is_query:
            mval = x_proj.amax(dim=-1, keepdim=True)
        else:
            mval = x_proj.amax(dim=(-2, -1), keepdim=True)
        features = ratio * torch.exp(x_proj - diag - mval) + self.eps
        return features

    def _phi(self, x: torch.Tensor, is_query: bool) -> torch.Tensor:
        return self._phi_nonnegative(x, is_query)

    def forward(self, q, k, v, key_padding_mask: Optional[torch.Tensor] = None):
        """
        q,k,v: [B,H,T,D]
        key_padding_mask: [B,T] with True = ignore
        """
        assert q.dim() == k.dim() == v.dim() == 4
        assert q.shape[:2] == k.shape[:2] == v.shape[:2]  # B,H
        assert q.size(-1) == k.size(-1) == self.dim

        if self.redraw:
            self.redraw_projection()

        q_prime = self._phi(q, is_query=True)    # [B,H,Tq,M]
        k_prime = self._phi(k, is_query=False)   # [B,H,Tk,M]

        # Apply key padding mask (True=ignore): zero contributions of masked keys
        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(k_prime.dtype)    # [B,Tk]
            keep = keep[:, None, :, None]                   # [B,1,Tk,1]
            k_prime = k_prime * keep
            v = v * keep

        # Bidirectional linear attention
        Z = torch.einsum("bhsm,bhsd->bhmd", k_prime, v)         # [B,H,M,Dv]
        W = torch.einsum("bhtm,bhmd->bhtd", q_prime, Z)         # [B,H,Tq,Dv]

        if not self.renorm:
            return W

        K_sum = k_prime.sum(dim=2)                              # [B,H,M]
        R = torch.einsum("bhtm,bhm->bht", q_prime, K_sum)       # [B,H,Tq]
        return W / (R.unsqueeze(-1) + self.eps)


class MultiheadFastAttention(nn.Module):
    """
    Multi-head wrapper around FastAttention with MHA-like API.
    Accepts [L,N,E] (batch_first=False) or [N,L,E] when batch_first=True.
    Returns (attn_output, None).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        nb_features: int = 256,
        dropout: float = 0.0,
        causal: bool = False,
        ortho_features: bool = False,
        redraw_features: bool = False,
        renormalize: bool = True,
        nonnegative_features: bool = True,
        eps: float = 1e-6,
        batch_first: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.dropout_p = dropout
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.core = FastAttention(
            dim=self.head_dim,
            nb_features=nb_features,
            ortho_features=ortho_features,
            redraw_features=redraw_features,
            causal=causal,
            renormalize=renormalize,
            nonnegative_features=nonnegative_features,
            eps=eps,
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # [L,N,E] -> [N,H,L,D]
        if self.batch_first:
            N, L, E = x.shape
            x = x.contiguous().view(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            return x  # [N,H,L,D]
        L, N, E = x.shape
        x = x.permute(1, 0, 2).contiguous().view(N, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [N,H,L,D]

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        # Projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Shapes -> [N,H,L,D]
        qh = self._shape(q)
        kh = self._shape(k)
        vh = self._shape(v)

        # Normalize mask to key padding mask [N, Lk]
        if self.batch_first:
            N, Lk, _ = key.shape
            Tq = query.shape[1]
        else:
            Lk, N, _ = key.shape
            Tq = query.shape[0]
        key_padding_mask = _to_key_padding_mask(attn_mask, N, Tq, Lk)

        # Core fast attention (returns [N,H,L,D])
        out = self.core(qh, kh, vh, key_padding_mask=key_padding_mask)

        # Merge heads back
        N, H, L, D = out.shape
        out = out.permute(2, 0, 1, 3).contiguous().view(L, N, H * D)  # [L,N,E]
        if self.batch_first:
            out = out.permute(1, 0, 2).contiguous()                   # [N,L,E]
        out = self.out_proj(self.dropout(out))
        return out, None


# ------------------------------------------------------------
# 3) Mobile Attention (ICML'24)
#     Paper: https://openreview.net/forum?id=VHtIDVaOKC
#     Code : https://github.com/thuml/MobileAttention
# ------------------------------------------------------------

class Mobile_Attention(nn.Module):
    """
    Mobile Attention with head competing mechanism.
    Inputs can be [T,B,C] (batch_first=False) or [B,T,C] when True.
    Supports cross-attention and key padding mask (True = ignore).

    need_weights:
      None        -> do not return weights
      'factors'   -> return components
      'full'      -> return full [B,H,Lq,Lk] attention (quadratic)
    """
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=1e-6, batch_first=False):
        super(Mobile_Attention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.batch_first = batch_first

        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection   = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)   # keep V in d_model
        self.out_projection   = nn.Linear(d_model, d_output)

        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv  = torch.einsum("bhld,bhlm->bhdm", k, v)     # [B,H,Dh,Dv]
        qkv = torch.einsum("bhld,bhdm->bhlm", q, kv)    # [B,H,Lq,Dv]
        return qkv

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor] = None, need_weights: Optional[str] = None):
        # layout to [B,T,C]
        if not self.batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        # linear projections
        q = self.query_projection(q).view(B, Lq, self.n_heads, self.d_head)
        k = self.key_projection(k).view(B, Lk, self.n_heads, self.d_head)
        v = self.value_projection(v).view(B, Lk, self.n_heads, self.d_head)

        # move heads forward
        q = q.permute(0, 2, 1, 3).contiguous()  # [B,H,Lq,Dh]
        k = k.permute(0, 2, 1, 3).contiguous()  # [B,H,Lk,Dh]
        v = v.permute(0, 2, 1, 3).contiguous()  # [B,H,Lk,Dh]

        # key padding mask (True = ignore) on positions
        key_padding_mask = _to_key_padding_mask(attn_mask, B, Lq, Lk)
        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(k.dtype)              # [B,Lk]
            keep = keep[:, None, :, None]                       # [B,1,Lk,1]
            k = k * keep
            v = v * keep

        # nonnegative projection
        q_nng = self.kernel_method(q)  # [B,H,Lq,Dh]
        k_nng = self.kernel_method(k)  # [B,H,Lk,Dh]

        # head competing mechanics
        sumK_hp = k_nng.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)  # [B,1,1,Dh]
        sumQ_hp = q_nng.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)  # [B,1,1,Dh]

        sink_incoming   = 1.0 / torch.sum((q_nng + self.eps) * (sumK_hp + self.eps), dim=-1)  # [B,H,Lq]
        source_outgoing = 1.0 / torch.sum((k_nng + self.eps) * (sumQ_hp + self.eps), dim=-1)  # [B,H,Lk]

        K_weighted   = k_nng * source_outgoing[:, :, :, None]
        sumKw_hp     = K_weighted.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        conserved_sink   = torch.sum((q_nng + self.eps) * (sumKw_hp + self.eps), dim=-1)      # [B,H,Lq]

        Q_weighted   = q_nng * sink_incoming[:, :, :, None]
        sumQw_hp     = Q_weighted.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        conserved_source = torch.sum((k_nng + self.eps) * (sumQw_hp + self.eps), dim=-1)      # [B,H,Lk]
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)

        if key_padding_mask is not None:
            conserved_source = conserved_source.masked_fill(key_padding_mask[:, None, :], float('-inf'))

        sink_allocation    = torch.sigmoid(conserved_sink * (float(Lq) / float(max(Lk, 1))))  # [B,H,Lq]
        source_competition = torch.softmax(conserved_source, dim=-1) * float(Lk)              # [B,H,Lk]

        # dot product with competition and allocation
        q_eff = q_nng * sink_incoming[:, :, :, None]
        v_eff = v     * source_competition[:, :, :, None]
        x = self.dot_product(q_eff, k_nng, v_eff)                    # [B,H,Lq,Dh]
        x = (x * sink_allocation[:, :, :, None]).transpose(1, 2)     # [B,Lq,H,Dh]

        # final projection
        x = x.reshape(B, Lq, self.n_heads * self.d_head)
        x = self.out_projection(x)
        x = self.dropout(x)

        if not self.batch_first:
            x = x.transpose(0, 1)

        attn = None
        if need_weights == 'factors':
            attn = {
                'q_nng': q_nng, 'k_nng': k_nng,
                'sink_incoming': sink_incoming, 'sink_allocation': sink_allocation,
                'source_competition': source_competition
            }
        elif need_weights == 'full':
            A = torch.einsum('bhld,bhsd->bhls', q_nng * sink_incoming[:, :, :, None], k_nng)
            A = A * source_competition[:, :, None, :]
            A = A * sink_allocation[:, :, :, None]
            if key_padding_mask is not None:
                A = A.masked_fill(key_padding_mask[:, None, None, :], 0.0)
            attn = A
        return x, attn


# ------------------------------------------------------------
# 4) Hydra Attention (ECCV W'22)
#     Paper: https://arxiv.org/abs/2209.07484
#     Code : https://github.com/robflynnyh/hydra-linear-attention
# ------------------------------------------------------------

def hydra_attention(Q, K, V, eps=1e-6):
    # Single-head functional version (for reference)
    Q = Q / (Q.norm(dim=-1, keepdim=True) + eps)
    K = K / (K.norm(dim=-1, keepdim=True) + eps)
    g = (K * V).sum(dim=-2, keepdim=True)
    out = Q * g
    return out

class HydraAttentionQKV(nn.Module):
    """
    Hydra Attention with many heads using global per-head vector.
    Accepts [T,B,D] (batch_first=False) or [B,T,D] when batch_first=True.
    """
    def __init__(
        self,
        qk_dim: int,
        num_heads: int = 1,
        head_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        phi_q: str = 'l2',
        phi_k: str = 'l2',                   # or 'softmax_token'
        dropout: float = 0.0,
        eps: float = 1e-6,
        batch_first: bool = False,
        return_attn_weights: bool = True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.return_attn_weights = return_attn_weights

        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (qk_dim // num_heads)
        assert self.num_heads * self.head_dim == self.qk_dim, "qk_dim must equal num_heads * head_dim"

        self.v_dim = v_dim or qk_dim
        self.out_dim = out_dim or qk_dim
        self.phi_q = phi_q
        self.phi_k = phi_k
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

        self.v_proj = nn.Identity() if self.v_dim == self.qk_dim else nn.Linear(self.v_dim, self.qk_dim, bias=False)
        self.out_proj = nn.Identity() if self.out_dim == self.qk_dim else nn.Linear(self.qk_dim, self.out_dim, bias=False)

    def _phi_l2(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + self.eps)

    def _phi_pointwise(self, x, kind: str):
        if kind == 'l2':
            return self._phi_l2(x)
        if kind == 'tanh':
            return torch.tanh(x)
        if kind == 'sigmoid':
            return torch.sigmoid(x)
        if kind == 'none':
            return x
        raise ValueError(f"Unsupported phi kind: {kind}")

    def _phi_k(self, K, key_padding_mask):
        # K: (B,H,Tk,Dh)
        if self.phi_k == 'softmax_token':
            B, H, Tk, Dh = K.shape
            Kt = K.permute(0, 1, 3, 2)  # (B,H,Dh,Tk)
            if key_padding_mask is not None:
                mask = key_padding_mask[:, None, None, :]  # (B,1,1,Tk)
                Kt = Kt.masked_fill(mask, float('-inf'))
            Kt = torch.softmax(Kt, dim=-1)
            Kt = torch.nan_to_num(Kt, nan=0.0)
            return Kt.permute(0, 1, 3, 2)       # (B,H,Tk,Dh)
        else:
            Kp = self._phi_pointwise(K, self.phi_k)
            if key_padding_mask is not None:
                Kp = Kp.masked_fill(key_padding_mask[:, None, :, None], 0.0)
            return Kp

    def _to_heads(self, x, T):
        # x: (B,T,D) -> (B,H,T,Dh)
        B = x.size(0)
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _from_heads(self, x):
        # x: (B,H,T,Dh) -> (B,T,D)
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).reshape(B, T, H * Dh)

    def forward(
        self,
        Q: torch.Tensor,   # (Tq,B,D) or (B,Tq,D)
        K: torch.Tensor,   # (Tk,B,D) or (B,Tk,D)
        V: torch.Tensor,   # (Tk,B,Dv) or (B,Tk,Dv)
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,  # alternative input
    ):
        if not self.batch_first:
            Q = Q.transpose(0, 1).contiguous()
            K = K.transpose(0, 1).contiguous()
            V = V.transpose(0, 1).contiguous()

        B, Tq, D = Q.shape
        _, Tk, Dk = K.shape
        assert D == self.qk_dim == Dk, "Q and K last dim must equal qk_dim"

        # Normalize mask to key padding mask
        if key_padding_mask is None:
            key_padding_mask = _to_key_padding_mask(attn_mask, B, Tq, Tk)

        # Heads
        Qh = self._to_heads(Q, Tq)        # (B,H,Tq,Dh)
        Kh = self._to_heads(K, Tk)        # (B,H,Tk,Dh)
        Vp = self.v_proj(V)               # (B,Tk,D)
        Vh = self._to_heads(Vp, Tk)       # (B,H,Tk,Dh)

        # Kernels
        Qp = self._phi_pointwise(Qh, self.phi_q)      # (B,H,Tq,Dh)
        Kp = self._phi_k(Kh, key_padding_mask)        # (B,H,Tk,Dh)

        kv = Kp * Vh                                   # (B,H,Tk,Dh)
        if self.dropout.p > 0:
            kv = self.dropout(kv.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        g = kv.sum(dim=2, keepdim=True)                # (B,H,1,Dh)

        out = Qp * g                                   # (B,H,Tq,Dh)
        out = self._from_heads(out)                    # (B,Tq,D)
        out = self.out_proj(out)                       # (B,Tq,out)

        if not self.batch_first:
            out = out.transpose(0, 1).contiguous()     # (Tq,B,D)

        attn_weights = None
        return (out, attn_weights) if self.return_attn_weights else out


# ------------------------------------------------------------
# 5) PolaLinearAttention (PolaFormer-style, arXiv'25)
#     Paper: https://arxiv.org/abs/2501.15061
#     Code : https://github.com/ZacharyMeng/PolaFormer
# ------------------------------------------------------------

class PolaLinearAttention(nn.Module):
    """
    Polarity-aware Linear Attention usable for self- or cross-attention.

    - batch_first supported.
    - If num_patches is None, positional encoding & SR are disabled (sr_ratio forced to 1).
    - Mask semantics: PyTorch-style (True = ignore).
    - For ViT-style SR, pass num_patches at init and (H/W for queries, H_k/W_k for keys) at call.
    """
    def __init__(
        self,
        dim: int,
        num_patches: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        kernel_size: int = 5,
        alpha: float = 4.0,
        batch_first: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "Pola requires head_dim to be even (for v split)."
        self.batch_first = batch_first

        # If num_patches is not provided, run in "generic" mode (no SR, no PE).
        self.use_spatial = (num_patches is not None)
        if not self.use_spatial:
            sr_ratio = 1  # force off

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)  # q and gate g
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)  # k and v

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.use_spatial and sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.dwc = nn.Conv2d(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=kernel_size,
            groups=self.head_dim,
            padding=kernel_size // 2,
        )

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

        if self.use_spatial:
            assert num_patches % (sr_ratio * sr_ratio) == 0, \
                "num_patches must be divisible by sr_ratio**2"
            self.n_after_sr = num_patches // (sr_ratio * sr_ratio)
            self.positional_encoding = nn.Parameter(
                torch.zeros(size=(1, self.n_after_sr, dim))
            )
        else:
            self.n_after_sr = None
            self.register_buffer("positional_encoding", None, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        H: Optional[int] = None, W: Optional[int] = None,
        H_k: Optional[int] = None, W_k: Optional[int] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # ----- to [B,T,C] -----
        if not self.batch_first:
            q = q.transpose(0, 1)
            if k is not None: k = k.transpose(0, 1)
            if v is not None: v = v.transpose(0, 1)

        if k is None: k = q
        if v is None: v = k

        B, Nq, C = q.shape
        Bk, Nk, Ck = k.shape
        Bv, Nv, Cv = v.shape
        assert B == Bk == Bv and Nk == Nv and C == Ck == Cv == self.dim

        # ----- projections -----
        q_lin, g = self.qg(q).reshape(B, Nq, 2, C).unbind(2)

        # SR on keys (if enabled)
        if self.use_spatial and self.sr is not None:
            if H_k is None or W_k is None or Nk != H_k * W_k:
                raise ValueError("Pola with sr_ratio>1 requires H_k and W_k with Nk==H_k*W_k.")
            k_2d = k.permute(0, 2, 1).reshape(B, C, H_k, W_k)
            k_2d = self.sr(k_2d)
            k_seq = k_2d.reshape(B, C, -1).permute(0, 2, 1)
            k_seq = self.norm(k_seq)
            kv = self.kv(k_seq).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(k).reshape(B, -1, 2, C).permute(2, 0, 1, 3)

        k_lin, v_lin = kv[0], kv[1]   # [B, n, C]
        n = k_lin.size(1)

        # Positional encoding if spatial mode
        if self.use_spatial:
            if n != self.n_after_sr:
                raise ValueError(f"positional_encoding length {self.n_after_sr} != n={n}")
            k_lin = k_lin + self.positional_encoding[:, :n, :]

        # ----- masks (True = ignore) -----
        key_padding_mask = _to_key_padding_mask(attn_mask, B, Nq, n)
        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(k_lin.dtype).unsqueeze(-1)  # [B,n,1]
            k_lin = k_lin * keep
            v_lin = v_lin * keep

        # ----- kernel + heads -----
        scale = F.softplus(self.scale)
        q_lin = q_lin / scale
        k_lin = k_lin / scale

        qh = q_lin.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        kh = k_lin.reshape(B, n,  self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        vh = v_lin.reshape(B, n,  self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        power = 1.0 + self.alpha * torch.sigmoid(self.power)
        relu = F.relu
        q_pos, q_neg = relu(qh) ** power, relu(-qh) ** power
        k_pos, k_neg = relu(kh) ** power, relu(-kh) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)  # [B,H,Nq,2Dh]
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k_feat = torch.cat([k_pos, k_neg], dim=-1) # [B,H,n,2Dh]

        v1, v2 = torch.chunk(vh, 2, dim=-1)       # [B,H,n,Dh/2] each

        k_mean = k_feat.mean(dim=-2, keepdim=True)                           # [B,H,1,2Dh]
        z_sim = 1.0 / (torch.matmul(q_sim, k_mean.transpose(-2, -1)) + 1e-6) # [B,H,Nq,1]
        kv_sim = torch.matmul(k_feat.transpose(-2, -1) * (n ** -0.5), v1 * (n ** -0.5))
        x_sim = torch.matmul(q_sim, kv_sim) * z_sim

        z_opp = 1.0 / (torch.matmul(q_opp, k_mean.transpose(-2, -1)) + 1e-6)
        kv_opp = torch.matmul(k_feat.transpose(-2, -1) * (n ** -0.5), v2 * (n ** -0.5))
        x_opp = torch.matmul(q_opp, kv_opp) * z_opp

        xh = torch.cat([x_sim, x_opp], dim=-1)                               # [B,H,Nq,Dh]
        x = xh.transpose(1, 2).contiguous().view(B, Nq, self.dim)            # [B,Nq,C]

        # Optional depthwise refinement only if spatial grids are known and match
        if self.use_spatial and (H is not None and W is not None and H_k is not None and W_k is not None
                                 and (H == H_k) and (W == W_k) and (Nq == n == H * W)):
            v_grid = vh.reshape(B * self.num_heads, n, self.head_dim).transpose(1, 2)
            v_grid = v_grid.reshape(B * self.num_heads, self.head_dim, H, W)
            v_grid = self.dwc(v_grid)
            v_res = v_grid.reshape(B, self.num_heads, self.head_dim, H * W)
            v_res = v_res.transpose(2, 3).reshape(B, H * W, self.dim)
            x = x + v_res

        x = x * g
        x = self.proj(x)
        x = self.proj_drop(x)

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x, (None if not need_weights else None)


# ------------------------------------------------------------
# Registry / factory utilities
# ------------------------------------------------------------

_CANONICAL_ATTENTION_NAMES = {
    "efficient": "efficient",
    "efficient_attention": "efficient",
    "ea": "efficient",
    "fast": "fast",
    "performer": "fast",
    "favar": "fast",
    "mobile": "mobile",
    "mobile_attention": "mobile",
    "hydra": "hydra",
    "hydra_attention": "hydra",
    "pola": "pola",
    "polaformer": "pola",
}


def list_efficient_attention_choices(include_none: bool = True) -> List[str]:
    """
    Returns the list of CLI-friendly attention names.
    """
    names = sorted(set(_CANONICAL_ATTENTION_NAMES.values()))
    if include_none:
        return ["none"] + names
    return names


def normalize_efficient_attention_choice(choice: Optional[str]) -> Optional[str]:
    """
    Normalizes raw user input into the canonical registry key.
    Returns None when vanilla PyTorch attention should be used.
    """
    if choice is None:
        return None
    key = str(choice).strip().lower()
    if key in ("", "none", "vanilla", "default", "standard"):
        return None
    normalized = _CANONICAL_ATTENTION_NAMES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unknown efficient attention '{choice}'. "
            f"Valid options: {', '.join(list_efficient_attention_choices())}"
        )
    return normalized


class MultiheadAttentionAdapter(nn.Module):
    """
    Thin wrapper that exposes a MultiheadAttention-style interface while delegating
    to one of the efficient attention implementations defined above.
    """

    def __init__(
        self,
        attn_type: Optional[str],
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = False,
    ):
        super().__init__()
        self.kind = normalize_efficient_attention_choice(attn_type)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.core = self._build_core()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            core = super().__getattribute__("core")
            if hasattr(core, name):
                return getattr(core, name)
            raise

    def _build_core(self) -> nn.Module:
        if self.kind is None:
            return nn.MultiheadAttention(
                self.embed_dim,
                self.num_heads,
                dropout=self.dropout,
                batch_first=self.batch_first,
            )

        if self.kind == "efficient":
            return EfficientAttention(
                in_channels=self.embed_dim,
                key_channels=self.embed_dim,
                value_channels=self.embed_dim,
                num_heads=self.num_heads,
                dropout_p=self.dropout,
                batch_first=self.batch_first,
            )

        if self.kind == "fast":
            return MultiheadFastAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                nb_features=256,
                dropout=self.dropout,
                batch_first=self.batch_first,
            )

        if self.kind == "mobile":
            return Mobile_Attention(
                d_input=self.embed_dim,
                d_model=self.embed_dim,
                d_output=self.embed_dim,
                n_heads=self.num_heads,
                drop_out=self.dropout,
                batch_first=self.batch_first,
            )

        if self.kind == "hydra":
            return HydraAttentionQKV(
                qk_dim=self.embed_dim,
                num_heads=self.num_heads,
                v_dim=self.embed_dim,
                out_dim=self.embed_dim,
                dropout=self.dropout,
                batch_first=self.batch_first,
                return_attn_weights=True,
            )

        if self.kind == "pola":
            return PolaLinearAttention(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                attn_drop=self.dropout,
                proj_drop=self.dropout,
                batch_first=self.batch_first,
            )

        raise ValueError(f"Unsupported efficient attention kind: {self.kind}")

    @staticmethod
    def _mask_to_bool(attn_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None
        if attn_mask.dtype == torch.bool:
            return attn_mask
        if attn_mask.is_floating_point():
            return attn_mask != 0
        return attn_mask.to(torch.bool)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.kind is None:
            return self.core(
                query,
                key,
                value,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )

        mask = self._mask_to_bool(attn_mask)
        if self.kind == "efficient":
            need_flag = "full" if need_weights else None
            out, attn = self.core(query, key, value, attn_mask=mask, need_weights=need_flag)
            return out, attn

        if self.kind == "fast":
            out, _ = self.core(query, key, value, attn_mask=mask, need_weights=need_weights)
            return out, None

        if self.kind == "mobile":
            need_flag = "full" if need_weights else None
            out, attn = self.core(query, key, value, attn_mask=mask, need_weights=need_flag)
            return out, attn

        if self.kind == "hydra":
            result = self.core(query, key, value, attn_mask=mask)
            if isinstance(result, tuple):
                return result
            return result, None

        if self.kind == "pola":
            out, attn = self.core(query, key, value, attn_mask=mask, need_weights=need_weights)
            return out, attn

        raise RuntimeError(f"Unhandled efficient attention type: {self.kind}")


def build_attention_module(
    attn_type: Optional[str],
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    batch_first: bool = False,
) -> MultiheadAttentionAdapter:
    return MultiheadAttentionAdapter(
        attn_type=attn_type,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=batch_first,
    )


# ------------------------------------------------------------
# Exports
# ------------------------------------------------------------

__all__ = [
    "EfficientAttention",
    "FastAttention",
    "MultiheadFastAttention",
    "Mobile_Attention",
    "HydraAttentionQKV",
    "PolaLinearAttention",
    "hydra_attention",
    "MultiheadAttentionAdapter",
    "build_attention_module",
    "list_efficient_attention_choices",
    "normalize_efficient_attention_choice",
]
