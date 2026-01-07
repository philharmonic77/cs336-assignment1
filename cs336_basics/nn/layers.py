from jaxtyping import Float, Int, Bool
import torch
import math
from torch import nn
from einops import einsum, rearrange, reduce

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.weight : Float[torch.Tensor, "d_out d_in"] = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
            )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, 
                x: Float[torch.Tensor, "... d_in"]
            ) -> Float[torch.Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()

        self.embedding : Float[torch.Tensor, "vocab_size d_model"] = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1, a=-3, b=3)

    def forward(self, 
                token_ids: Int[torch.Tensor, "... seq"]
                ) -> Float[torch.Tensor, "... seq d_model"]:
        return self.embedding[token_ids]


