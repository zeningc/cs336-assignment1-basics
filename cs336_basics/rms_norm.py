import torch
import torch.nn as nn
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype =x.dtype
        x = x.to(torch.float32)
        ms = einsum(x, x, "... d, ... d -> ...") / self.d_model
        inv_rms = torch.rsqrt(ms + self.eps)
        y = einsum(x, inv_rms, "... d, ... -> ... d")
        y = einsum(y, self.weight.to(torch.float32), "... d, d -> ... d")
        return y.to(in_dtype)