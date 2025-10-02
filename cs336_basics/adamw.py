# adamw.py
import math
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    r"""
    AdamW optimizer (Loshchilov & Hutter, 2019), decoupled weight decay.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate α
        betas (Tuple[float, float]): coefficients used for computing running averages of gradient and its square (β1, β2)
        eps (float): term added to the denominator to improve numerical stability (ε)
        weight_decay (float): weight decay λ (decoupled)
    """
    def __init__(self, params, lr: float=1e-3, betas=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # first and second moment running averages
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]  # starts at 1

                # m_t = β1 m_{t-1} + (1-β1) g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = β2 v_{t-1} + (1-β2) g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias-corrected learning rate:
                # α_t = α * sqrt(1 - β2^t) / (1 - β1^t)
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_c2) / bias_c1

                # parameter update:
                # θ ← θ - α_t * m_t / (sqrt(v_t) + ε)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # decoupled weight decay:
                # θ ← θ - α * λ * θ
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

        return loss
