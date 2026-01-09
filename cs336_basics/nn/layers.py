from jaxtyping import Float, Int, Bool
import torch
import math
from torch import Tensor, nn
from einops import einsum, rearrange, reduce

class Linear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
                 ):
        super().__init__()

        self.weight : Float[Tensor, "d_out d_in"] = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(
            self, 
            x: Float[Tensor, "... d_in"]
            ) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
            ):
        super().__init__()

        self.embedding : Float[Tensor, "vocab_size d_model"] = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1, a=-3, b=3)

    def forward(
            self, 
            token_ids: Int[Tensor, "..."]
                ) -> Float[Tensor, "... d_model"]:
        return self.embedding[token_ids]
    
class RMSNorm(nn.Module):      
    """
    rms(x) = sqrt(mean(x^2) + eps)
    y_i = x_i / rms(x) * g_i
    
    where g_i is a learned gain vector.
    """
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.eps = eps
        self.rms_g: Float[Tensor, "d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(
            self,
            x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:

        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms: Float[Tensor, "... 1"] = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result: Float[Tensor, "... d_model"] = x / rms * self.rms_g

        return result.to(in_dtype)

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x)⊙W3x)

    where SiLU(x) = x·sigmoid(x) = x / (1 + e^(-x))
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__() 
        
        self.w1: Linear  = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2: Linear  = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3: Linear  = Linear(d_model, d_ff, device=device, dtype=dtype)    

    def forward(
            self,
            x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        
        w1x: Float[Tensor, "... d_ff"] = self.w1(x)
        w3x: Float[Tensor, "... d_ff"] = self.w3(x)       
        silu_w1x: Float[Tensor, "... d_ff"] = w1x * torch.sigmoid(w1x)

        return self.w2(silu_w1x * w3x)
    