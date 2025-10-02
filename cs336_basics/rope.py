import torch.nn as nn
import torch
from einops import einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.half = d_k // 2
        self.max_seq_len = max_seq_len

        k = torch.arange(self.half, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2.0 * k / float(d_k))
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(pos, inv_freq, "s, p -> s p")
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        dtype, device = x.dtype, x.device
        cos = self.cos_cache.to(device=device, dtype=dtype)[token_positions]
        sin = self.sin_cache.to(device=device, dtype=dtype)[token_positions]
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        y_even = einsum(x_even, cos, "... s p, ... s p -> ... s p") - \
                 einsum(x_odd,  sin, "... s p, ... s p -> ... s p")
        y_odd  = einsum(x_even, sin, "... s p, ... s p -> ... s p") + \
                 einsum(x_odd,  cos, "... s p, ... s p -> ... s p")
        y = torch.empty_like(x)
        y[..., 0::2] = y_even
        y[..., 1::2] = y_odd
        return y
