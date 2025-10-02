import torch.nn as nn
import torch
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.swi_glu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
            use_rope=use_rope,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Sub-layer 1: RMSNorm -> MHA -> residual
        y = x + self.attn(self.ln1(x), token_positions=token_positions)
        # Sub-layer 2: RMSNorm -> FFN(SwiGLU) -> residual
        z = y + self.ffn(self.ln2(y))
        return z