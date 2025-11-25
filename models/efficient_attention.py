# efficient_attention_extended.py
# Collects efficient attention variants with consistent API and mask semantics.
# Masks follow PyTorch semantics: True = mask/ignore; False = keep.

from typing import Optional, Tuple, List, Dict
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
# ------------------------------------------------------------
class EfficientAttention(nn.Module):
    """
    Paper: https://arxiv.org/abs/1812.01243
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
        assert key_channels % num_heads == 0
        assert value_channels % num_heads == 0

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

    def _to_bhtd(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        B, T, C = x.shape
        H = self.num_heads
        D = dim // H
        return x.view(B, T, H, D).permute(0, 2, 1, 3).contiguous()

    def _from_bhtd(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

    def forward(self, q, k, v, attn_mask=None, need_weights=None):
        if not self.batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        B, Tq, _ = q.shape
        Bk, Tk, _ = k.shape

        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)

        Q = self._to_bhtd(Q, self.key_channels)  # [B,H,Tq,Dk]
        K = self._to_bhtd(K, self.key_channels)  # [B,H,Tk,Dk]
        V = self._to_bhtd(V, self.value_channels) # [B,H,Tk,Dv]

        K_logits = K.permute(0, 1, 3, 2).contiguous()
        Q_logits = Q.permute(0, 1, 3, 2).contiguous()

        key_padding_mask = _to_key_padding_mask(attn_mask, B, Tq, Tk)
        if key_padding_mask is not None:
            pos_mask = key_padding_mask[:, None, None, :]
            K_logits = K_logits.masked_fill(pos_mask, float("-inf"))

        K_norm = F.softmax(K_logits, dim=-1)
        Q_norm = F.softmax(Q_logits, dim=2)
        K_norm = self.dropout(K_norm)
        Q_norm = self.dropout(Q_norm)

        context = torch.matmul(K_norm, V)
        out_t = torch.matmul(context.transpose(-2, -1), Q_norm)
        out = out_t.permute(0, 1, 3, 2).contiguous()

        out = self._from_bhtd(out)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)

        attn = None
        if need_weights == "full":
            attn = torch.matmul(Q_norm.transpose(-2, -1), K_norm)

        return out, attn


# ------------------------------------------------------------
# 2) Performer-style Fast Attention (ICLR'21)
# ------------------------------------------------------------
# Minimal implementation dependencies
def _gaussian_random_matrix(rows: int, cols: int, device=None, dtype=None):
    return torch.randn(rows, cols, device=device, dtype=dtype)

class FastAttention(nn.Module):
    """
    Paper: https://arxiv.org/abs/2009.14794
    """
    def __init__(self, dim, nb_features=256, redraw_features=False, renormalize=True, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.m = nb_features
        self.redraw = redraw_features
        self.renorm = renormalize
        self.eps = eps
        self.register_buffer("proj", _gaussian_random_matrix(self.m, self.dim, device=device, dtype=dtype), persistent=False)

    def _phi(self, x, is_query):
        # Non-negative FAVOR+ kernel
        B, H, T, D = x.shape
        ratio = 1.0 / math.sqrt(self.m)
        data_normalizer = 1.0 / math.sqrt(math.sqrt(D))
        x_proj = torch.einsum("bhtd,md->bhtm", x * data_normalizer, self.proj)
        diag = 0.5 * (data_normalizer ** 2) * (x * x).sum(dim=-1, keepdim=True)
        if is_query:
            mval = x_proj.amax(dim=-1, keepdim=True)
        else:
            mval = x_proj.amax(dim=(-2, -1), keepdim=True)
        features = ratio * torch.exp(x_proj - diag - mval) + self.eps
        return features

    def forward(self, q, k, v, key_padding_mask=None):
        if self.redraw:
            self.proj.copy_(_gaussian_random_matrix(self.m, self.dim, device=self.proj.device, dtype=self.proj.dtype))
        
        q_prime = self._phi(q, True)
        k_prime = self._phi(k, False)

        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(k_prime.dtype)[:, None, :, None]
            k_prime = k_prime * keep
            v = v * keep

        Z = torch.einsum("bhsm,bhsd->bhmd", k_prime, v)
        W = torch.einsum("bhtm,bhmd->bhtd", q_prime, Z)

        if not self.renorm:
            return W

        K_sum = k_prime.sum(dim=2)
        R = torch.einsum("bhtm,bhm->bht", q_prime, K_sum)
        return W / (R.unsqueeze(-1) + self.eps)

class MultiheadFastAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, nb_features=256, dropout=0.0, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.core = FastAttention(dim=self.head_dim, nb_features=nb_features)

    def _shape(self, x):
        if self.batch_first:
            N, L, E = x.shape
            return x.view(N, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        L, N, E = x.shape
        return x.view(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))
        
        # Determine shapes for mask
        if self.batch_first:
            B, Tq = query.shape[:2]
            Tk = key.shape[1]
        else:
            B = query.shape[1]
            Tq = query.shape[0]
            Tk = key.shape[0]

        key_padding_mask = _to_key_padding_mask(attn_mask, B, Tq, Tk)
        out = self.core(q, k, v, key_padding_mask=key_padding_mask)
        
        # Merge
        out = out.permute(2, 0, 1, 3).contiguous().view(-1, B, self.embed_dim) # [L, B, E]
        if self.batch_first:
            out = out.permute(1, 0, 2)
        out = self.out_proj(self.dropout(out))
        return out, None


# ------------------------------------------------------------
# 3) Mobile Attention (ICML'24)
# ------------------------------------------------------------
class Mobile_Attention(nn.Module):
    """
    Paper: https://openreview.net/forum?id=VHtIDVaOKC
    """
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, batch_first=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.batch_first = batch_first
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = 1e-6

    def forward(self, q, k, v, attn_mask=None, need_weights=None):
        if not self.batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        q = self.query_projection(q).view(B, Lq, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.key_projection(k).view(B, Lk, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.value_projection(v).view(B, Lk, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        key_padding_mask = _to_key_padding_mask(attn_mask, B, Lq, Lk)
        if key_padding_mask is not None:
            keep = (~key_padding_mask).to(k.dtype)[:, None, :, None]
            k = k * keep
            v = v * keep

        q_nng = torch.sigmoid(q)
        k_nng = torch.sigmoid(k)

        # Head competing mechanism
        sumK = k_nng.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        sumQ = q_nng.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        
        sink = 1.0 / torch.sum((q_nng + self.eps) * (sumK + self.eps), dim=-1)
        source = 1.0 / torch.sum((k_nng + self.eps) * (sumQ + self.eps), dim=-1)
        
        Kw = k_nng * source.unsqueeze(-1)
        sumKw = Kw.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        conserved_sink = torch.sum((q_nng + self.eps) * (sumKw + self.eps), dim=-1)
        
        Qw = q_nng * sink.unsqueeze(-1)
        sumQw = Qw.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        conserved_source = torch.sum((k_nng + self.eps) * (sumQw + self.eps), dim=-1)
        conserved_source = torch.clamp(conserved_source, -1.0, 1.0)
        
        if key_padding_mask is not None:
             conserved_source = conserved_source.masked_fill(key_padding_mask[:, None, :], float('-inf'))

        alloc = torch.sigmoid(conserved_sink * (float(Lq) / max(float(Lk), 1.0)))
        comp = torch.softmax(conserved_source, dim=-1) * float(Lk)

        q_eff = q_nng * sink.unsqueeze(-1)
        v_eff = v * comp.unsqueeze(-1)
        
        # Linear Attn
        kv = torch.einsum("bhld,bhlm->bhdm", k_nng, v_eff)
        x = torch.einsum("bhld,bhdm->bhlm", q_eff, kv)
        x = (x * alloc.unsqueeze(-1)).permute(0, 2, 1, 3).reshape(B, Lq, -1)
        
        x = self.dropout(self.out_projection(x))
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, None


# ------------------------------------------------------------
# 4) Hydra Attention (ECCV'22)
# ------------------------------------------------------------
class HydraAttentionQKV(nn.Module):
    """
    Paper: https://arxiv.org/abs/2209.07484
    """
    def __init__(self, qk_dim, num_heads=1, batch_first=False, dropout=0.0):
        super().__init__()
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.head_dim = qk_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6
        self.out_proj = nn.Linear(qk_dim, qk_dim)

    def forward(self, Q, K, V, attn_mask=None):
        if not self.batch_first:
            Q, K, V = Q.transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1)
        
        B, Tq, _ = Q.shape
        _, Tk, _ = K.shape

        Qh = Q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        Kh = K.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        Vh = V.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        key_padding_mask = _to_key_padding_mask(attn_mask, B, Tq, Tk)

        # Normalize (L2)
        Qp = Qh / (Qh.norm(dim=-1, keepdim=True) + self.eps)
        Kp = Kh / (Kh.norm(dim=-1, keepdim=True) + self.eps)

        if key_padding_mask is not None:
             Kp = Kp.masked_fill(key_padding_mask[:, None, :, None], 0.0)

        kv = Kp * Vh
        if self.dropout.p > 0:
            kv = self.dropout(kv)
        
        g = kv.sum(dim=2, keepdim=True) # Global aggregation
        out = Qp * g
        
        out = out.transpose(1, 2).reshape(B, Tq, -1)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None


# ------------------------------------------------------------
# 5) PolaLinearAttention (arXiv'25)
# ------------------------------------------------------------
class PolaLinearAttention(nn.Module):
    """
    Paper: https://arxiv.org/abs/2501.15061
    """
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, batch_first=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.batch_first = batch_first
        self.qg = nn.Linear(dim, 2 * dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.power = nn.Parameter(torch.zeros(size=(1, num_heads, 1, self.head_dim)))
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.alpha = 4.0

    def forward(self, q, k=None, v=None, attn_mask=None, need_weights=False):
        if not self.batch_first:
            q = q.transpose(0, 1)
            if k is not None: k = k.transpose(0, 1)
            if v is not None: v = v.transpose(0, 1)
        
        if k is None: k = q
        if v is None: v = k

        B, Nq, C = q.shape
        Bk, Nk, Ck = k.shape

        q_lin, g = self.qg(q).reshape(B, Nq, 2, C).unbind(2)
        kv = self.kv(k).reshape(B, Nk, 2, C).permute(2, 0, 1, 3)
        k_lin, v_lin = kv[0], kv[1]

        # Masking
        key_padding_mask = _to_key_padding_mask(attn_mask, B, Nq, Nk)
        if key_padding_mask is not None:
             keep = (~key_padding_mask).to(k_lin.dtype).unsqueeze(-1)
             k_lin = k_lin * keep
             v_lin = v_lin * keep

        scale = F.softplus(self.scale)
        q_lin = q_lin / scale
        k_lin = k_lin / scale

        qh = q_lin.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kh = k_lin.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vh = v_lin.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        power = 1.0 + self.alpha * torch.sigmoid(self.power)
        q_pos, q_neg = F.relu(qh) ** power, F.relu(-qh) ** power
        k_pos, k_neg = F.relu(kh) ** power, F.relu(-kh) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k_feat = torch.cat([k_pos, k_neg], dim=-1)
        v1, v2 = torch.chunk(vh, 2, dim=-1)

        k_mean = k_feat.mean(dim=-2, keepdim=True)
        z_sim = 1.0 / (torch.matmul(q_sim, k_mean.transpose(-2, -1)) + 1e-6)
        kv_sim = torch.matmul(k_feat.transpose(-2, -1) * (Nk**-0.5), v1 * (Nk**-0.5))
        x_sim = torch.matmul(q_sim, kv_sim) * z_sim
        
        z_opp = 1.0 / (torch.matmul(q_opp, k_mean.transpose(-2, -1)) + 1e-6)
        kv_opp = torch.matmul(k_feat.transpose(-2, -1) * (Nk**-0.5), v2 * (Nk**-0.5))
        x_opp = torch.matmul(q_opp, kv_opp) * z_opp

        x = torch.cat([x_sim, x_opp], dim=-1).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x * g)

        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, None


# ------------------------------------------------------------
# 6) Linear Attention (Transformers are RNNs, ICML'20)
# ------------------------------------------------------------

class LinearAttention(nn.Module):
    """
    Standard Linear Attention using the kernel trick Phi(x) = elu(x) + 1.
    Paper: https://arxiv.org/abs/2006.16236 ("Transformers are RNNs")
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6

    def _feature_map(self, x):
        # The standard 'elu + 1' feature map from the paper
        return F.elu(x) + 1.0

    def forward(self, query, key, value, attn_mask: Optional[torch.Tensor] = None, need_weights: bool = False):
        # Standard API handling
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, Tq, _ = query.shape
        _, Tk, _ = key.shape

        q = self.q_proj(query).view(B, Tq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(B, Tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(B, Tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply feature map
        Q = self._feature_map(q) # [B,H,Tq,D]
        K = self._feature_map(k) # [B,H,Tk,D]

        # Handle mask (True = ignore)
        key_padding_mask = _to_key_padding_mask(attn_mask, B, Tq, Tk)
        if key_padding_mask is not None:
            # Mask K and V by zeroing them out
            mask = (~key_padding_mask).view(B, 1, Tk, 1).to(K.dtype)
            K = K * mask
            v = v * mask

        # Linear Attention Logic:
        # Numerator:   Q * (K^T * V)
        # Denominator: Q * (K^T * 1) aka sum of K
        
        # 1. Compute global context KV: [B, H, D, D]
        #    Sum over sequence length Tk
        KV = torch.einsum("bhtd,bhte->bhde", K, v)

        # 2. Compute normalizer (K_sum): [B, H, D]
        K_sum = K.sum(dim=2) 

        # 3. Compute output
        #    Numer: [B, H, Tq, D] dot [B, H, D, D] -> [B, H, Tq, D]
        num = torch.einsum("bhtd,bhde->bhte", Q, KV)
        
        #    Denom: [B, H, Tq, D] dot [B, H, D] -> [B, H, Tq]
        den = torch.einsum("bhtd,bhd->bht", Q, K_sum)
        den = den.unsqueeze(-1) + self.eps

        out = num / den

        # Reshape and Project
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, Tq, self.embed_dim)
        out = self.out_proj(self.dropout(out))

        if not self.batch_first:
            out = out.transpose(0, 1)
        
        return out, None


# ------------------------------------------------------------
# 7) XCiT Attention (Cross-Covariance Image Transformer, NeurIPS'21)
# ------------------------------------------------------------

class XCiTAttention(nn.Module):
    """
    Cross-Covariance Attention (XCA). Computes attention across channels rather than tokens.
    Complexity is O(N * d^2 / h) linear in N.
    Paper: https://arxiv.org/abs/2106.09681
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask: Optional[torch.Tensor] = None, need_weights: bool = False):
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        B, Tq, _ = query.shape
        _, Tk, _ = key.shape

        q = self.q_proj(query).reshape(B, Tq, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B,H,T,D
        k = self.k_proj(key).reshape(B, Tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, Tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # L2 Normalize Q and K (vital for XCiT)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Handle mask
        key_padding_mask = _to_key_padding_mask(attn_mask, B, Tq, Tk)
        if key_padding_mask is not None:
             # For XCiT (channel mixing), we zero out masked tokens so they don't contribute to covariance
             mask = (~key_padding_mask).view(B, 1, Tk, 1).to(q.dtype)
             k = k * mask
             v = v * mask
             # Note: exact masking in XCiT is subtle; zeroing K usually sufficient for cross-covariance

        # Attention Map: Interactions between Channels (D x D), aggregated over Tokens (T)
        # [B, H, D, T] @ [B, H, T, D] -> [B, H, D, D]
        attn = torch.matmul(k.transpose(-2, -1), q)
        attn = attn * self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply to V: [B, H, T, D] @ [B, H, D, D] -> [B, H, T, D]
        out = torch.matmul(v, attn)
        
        out = out.permute(0, 2, 1, 3).reshape(B, Tq, self.embed_dim)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)
            
        return out, attn if need_weights else None


# ------------------------------------------------------------
# 8) SimA (Simple Softmax-free Attention, WACV'24)
# ------------------------------------------------------------

class SimA(nn.Module):
    """
    Simple Softmax-free Attention. 
    Uses ReLU + L2-Norm on Q/K, then standard Linear Attention aggregation.
    Paper: https://arxiv.org/abs/2206.08898
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask: Optional[torch.Tensor] = None, need_weights: bool = False):
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, Tq, _ = query.shape
        _, Tk, _ = key.shape

        q = self.q_proj(query).view(B, Tq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(B, Tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(B, Tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # SimA Logic: ReLU -> L2 Norm along head_dim
        q = F.relu(q)
        k = F.relu(k)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Masking
        key_padding_mask = _to_key_padding_mask(attn_mask, B, Tq, Tk)
        if key_padding_mask is not None:
            mask = (~key_padding_mask).view(B, 1, Tk, 1).to(k.dtype)
            k = k * mask
            v = v * mask

        # Linear Aggregation (Output = Q * (K^T * V))
        # Note: SimA usually doesn't strictly divide by denominator like Katharopoulos, 
        # relying on the L2 norm for stability, but scaling by 1/sqrt(D) is common.
        # We implement the raw matrix multiplication form.
        
        # [B, H, D, Tk] @ [B, H, Tk, D] -> [B, H, D, D]
        kv = torch.matmul(k.transpose(-2, -1), v)
        
        # [B, H, Tq, D] @ [B, H, D, D] -> [B, H, Tq, D]
        out = torch.matmul(q, kv)
        
        # Reshape
        out = out.permute(0, 2, 1, 3).reshape(B, Tq, self.embed_dim)
        out = self.out_proj(self.dropout(out))

        if not self.batch_first:
            out = out.transpose(0, 1)
            
        return out, None


# ------------------------------------------------------------
# Registry / factory utilities
# ------------------------------------------------------------

_CANONICAL_ATTENTION_NAMES: Dict[str, str] = {
    "efficient": "efficient",
    "ea": "efficient",
    "fast": "fast",
    "performer": "fast",
    "mobile": "mobile",
    "hydra": "hydra",
    "pola": "pola",
    "polaformer": "pola",
    "linear": "linear",      # New
    "xcit": "xcit",          # New
    "sima": "sima",          # New
}


def list_efficient_attention_choices(include_none: bool = True) -> List[str]:
    names = sorted(set(_CANONICAL_ATTENTION_NAMES.values()))
    if include_none:
        return ["none"] + names
    return names


def normalize_efficient_attention_choice(choice: Optional[str]) -> Optional[str]:
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
    def __init__(self,
                 attn_type: Optional[str],
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 batch_first: bool = False):
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
            return nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=self.batch_first)
        
        # Existing
        if self.kind == "efficient":
            return EfficientAttention(self.embed_dim, self.embed_dim, self.embed_dim, self.num_heads, dropout_p=self.dropout, batch_first=self.batch_first)
        if self.kind == "fast":
            return MultiheadFastAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=self.batch_first)
        if self.kind == "mobile":
            return Mobile_Attention(self.embed_dim, self.embed_dim, self.embed_dim, self.num_heads, drop_out=self.dropout, batch_first=self.batch_first)
        if self.kind == "hydra":
            return HydraAttentionQKV(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=self.batch_first)
        if self.kind == "pola":
            return PolaLinearAttention(self.embed_dim, self.num_heads, attn_drop=self.dropout, proj_drop=self.dropout, batch_first=self.batch_first)
        
        # New
        if self.kind == "linear":
            return LinearAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=self.batch_first)
        if self.kind == "xcit":
            return XCiTAttention(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=self.batch_first)
        if self.kind == "sima":
            return SimA(self.embed_dim, self.num_heads, dropout=self.dropout, batch_first=self.batch_first)

        raise ValueError(f"Unsupported efficient attention kind: {self.kind}")

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        # Dispatch with unified API
        if self.kind is None:
             # Vanilla torch MHA needs specific kwargs
             return self.core(query, key, value, key_padding_mask=None, attn_mask=attn_mask, need_weights=need_weights)
        
        # For our efficient layers, attn_mask generally maps to key_padding_mask inside the layer
        # if the layer supports it.
        # We pass `attn_mask` which our layers internally convert to key_padding_mask using `_to_key_padding_mask`
        res = self.core(query, key, value, attn_mask=attn_mask, need_weights=need_weights)
        if isinstance(res, tuple):
            return res
        return res, None

def build_attention_module(attn_type, embed_dim, num_heads, dropout=0.0, batch_first=False):
    return MultiheadAttentionAdapter(attn_type, embed_dim, num_heads, dropout, batch_first)

__all__ = [
    "EfficientAttention",
    "FastAttention",
    "MultiheadFastAttention",
    "Mobile_Attention",
    "HydraAttentionQKV",
    "PolaLinearAttention",
    "LinearAttention",
    "XCiTAttention",
    "SimA",
    "MultiheadAttentionAdapter",
    "build_attention_module",
    "list_efficient_attention_choices",
]