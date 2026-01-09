from jaxtyping import Float, Int, Bool
import torch
import math
from torch import Tensor, nn
from einops import einsum, rearrange, reduce
from cs336_basics.nn.layers import Linear



class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None
    ):
        super().__init__()

        self.theta = theta
        self.d_k = d_k 
        self.max_seq_len = max_seq_len

        self.cos: Tensor
        self.sin: Tensor

        assert d_k % 2 == 0, "the dimension d_k of rope must be even!"
        half = d_k // 2

        k = torch.arange(1, half + 1, device=device).to(torch.float32)
        inv_freq = 1 / (theta ** ((2 * k - 2) / d_k))
        pos = torch.arange(0, max_seq_len, device=device).to(torch.float32)

        angles = torch.outer(pos, inv_freq)
        cos: Float[Tensor, "max_seq_len, half"] = torch.cos(angles)
        sin: Float[Tensor, "max_seq_len, half"] = torch.sin(angles)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
            self,
            x: Float[Tensor, "... seq d_k"],
            token_position: Int[Tensor, " ... seq"]
    ) -> Float[Tensor, "... seq d_k"]:
        
        assert x.shape[-2] == token_position.shape[-1]
        assert x.shape[-1] == self.d_k
        assert token_position.max() < self.max_seq_len
        assert token_position.min() >= 0

        token_position = token_position.to(torch.long)
        token_position = token_position.to(self.cos.device)

        cos_pos: Float[Tensor, "... seq half"] = self.cos[token_position]
        sin_pos: Float[Tensor, "... seq half"]  = self.sin[token_position]

        x_even: Float[Tensor, "... seq half"] = x[..., 0::2]
        x_odd: Float[Tensor, "... seq half"] = x[..., 1::2]

        out_even: Float[Tensor, "... seq half"] = cos_pos * x_even - sin_pos * x_odd
        out_odd: Float[Tensor, "... seq half"] = sin_pos * x_even + cos_pos * x_odd

        out: Float[Tensor, "... seq d_k"] = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        
        return out
    
def softmax(
    x: Float[Tensor, "... d_model"],
    dim: int = -1
) -> Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x  = torch.exp(x - x_max)
    
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... n d_k"],
    K: Float[Tensor, " ... m d_k"],
    V: Float[Tensor, " ... m d_v"],
    mask: Bool[Tensor, " ... n m"] | None = None,       
) -> Float[Tensor, "... n d_v"]:
    d_k = Q.shape[-1]
    assert d_k == K.shape[-1] 
    assert K.shape[-2] == V.shape[-2] 

    scores: Float[Tensor, "... n m"] = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    if mask is not None:
        assert mask.shape[-1] == K.shape[-2]
        assert mask.shape[-2] == Q.shape[-2]

        scores: Float[Tensor, "... n m"] = scores.masked_fill(~mask, float("-inf")) 

    scores: Float[Tensor, "... n m"] = softmax(scores, dim=-1)
    result: Float[Tensor, "... n d_v"] = einsum(scores, V, "... n m, ... m d_v -> ... n d_v")
    
    return result

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            rope: RotaryPositionalEmbedding | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads
        self.rope = rope

        assert d_model % num_heads == 0
        self.d_h = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
            self, 
            x: Float[Tensor, "... seq d_model"],
            token_position: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq d_model"]:
        
        Q_all = self.q_proj(x)
        K_all = self.k_proj(x)
        V_all = self.v_proj(x)

        # rearange 
        Q_all = rearrange(Q_all, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)
        K_all = rearrange(K_all, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)
        V_all = rearrange(V_all, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)

        # apply RoPE
        seq_len = x.shape[-2]
        if self.rope is not None:
            if token_position is None:
                token_position = torch.arange(
                    seq_len,
                    device=x.device,
                )
            Q_all = self.rope(Q_all, token_position)
            K_all = self.rope(K_all, token_position)
        
        # attention   
        causal_mask = (
            torch.arange(seq_len, device=x.device)
            <= torch.arange(seq_len, device=x.device)[:, None]
        )
        out = scaled_dot_product_attention(Q_all, K_all, V_all, mask=causal_mask)

        # projection
        out = rearrange(out, "... h seq d_h -> ... seq (h d_h)")
        out: Float[Tensor, "... d_model"] = self.o_proj(out)

        return out


