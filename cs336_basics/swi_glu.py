import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)

        x3 = self.w3(x)
        gated = x1 * x3
        return self.w2(gated)