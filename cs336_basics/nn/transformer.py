from jaxtyping import Float, Int
import torch
from torch import Tensor, nn
from cs336_basics.nn.layers import RMSNorm, PositionwiseFeedForward, Embedding, Linear
from cs336_basics.nn.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None            
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads   

        self.token_embeddings = Embedding(vocab_size, d_model, device =device, dtype=dtype)
        rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope=rope,
                    device=device,
                    dtype=dtype
                )
             for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
            self,
            x: Int[Tensor, "... seq"],
            token_position: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq vocab_size"]:
        
        if token_position is None:
            token_position = torch.arange(x.shape[-1], device=x.device, dtype=torch.long)
        
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x, token_position)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope: RotaryPositionalEmbedding | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads

        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
            self,
            x: Float[Tensor, "... seq d_model"],
            token_position: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq d_model"]:
        
        x = x + self.attn(self.ln1(x), token_position)
        x = x + self.ffn(self.ln2(x))
        return x
