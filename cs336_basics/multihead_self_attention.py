import torch.nn as nn
import torch
from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.utils import scaled_dot_product_attention
from einops import rearrange


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        kw = {"device": device, "dtype": dtype}

        # Q, K, V: project to (h * d_head)
        self.q_proj = Linear(d_model, num_heads * self.d_head, **kw)
        self.k_proj = Linear(d_model, num_heads * self.d_head, **kw)
        self.v_proj = Linear(d_model, num_heads * self.d_head, **kw)
        # Output: project back to d_model from concatenated heads
        self.output_proj = Linear(num_heads * self.d_head, d_model, **kw)
        # Optional RoPE
        self.rope = None
        self.use_rope = use_rope
        if use_rope:
            assert theta is not None and max_seq_len is not None, "theta and max_seq_len required when use_rope=True"
            self.rope = RoPE(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        def _causal_mask(seq_len: int, device) -> torch.Tensor:
            # True where key <= query (lower triangular)
            i = torch.arange(seq_len, device=device)
            return (i[:, None] >= i[None, :])  # (seq, seq)

        *batch, seq, _ = x.shape
        h = self.num_heads
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... s (h d) -> ... h s d", h=h)
        K = rearrange(K, "... s (h d) -> ... h s d", h=h)
        V = rearrange(V, "... s (h d) -> ... h s d", h=h)
        if self.use_rope:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)
            token_positions = token_positions.unsqueeze(-2)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = _causal_mask(seq, x.device) 
        Y = scaled_dot_product_attention(Q, K, V, mask=mask)
        Y = rearrange(Y, "... h s d -> ... s (h d)")  # (..., seq, h*d)
        return self.output_proj(Y)