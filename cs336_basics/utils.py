import torch
from einops import einsum
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # subtract the max along dim to avoid overflow
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp

def scaled_dot_product_attention(Q, K, V, mask):
    scores = einsum(Q, K, "... q d, ... k d -> ... q k")
    scores = scores / math.sqrt(Q.shape[-1])
    if mask is not None:
        if mask.ndim < scores.ndim:
            mask = mask.view((1,) * (scores.ndim - mask.ndim) + mask.shape)
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    out = einsum(attn, V, "... q k, ... k d -> ... q d")
    return out

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (..., vocab)  — unnormalized scores
    targets: (...)        — integer class indices (Long)
    returns: scalar tensor = mean cross-entropy over all leading dims
    """
    logits = logits.float()

    x_max = torch.amax(logits, dim=-1, keepdim=True)
    x = logits - x_max  

    logsumexp = torch.log(torch.sum(torch.exp(x), dim=-1))

    tgt = targets.long().unsqueeze(-1)
    x_y = torch.gather(x, dim=-1, index=tgt).squeeze(-1)

    nll = logsumexp - x_y

    # 6) average over all examples (all leading dims)
    return nll.mean()


def cosine_lr_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    Compute cosine learning rate with warmup.

    Args:
        t (int): current step (starting at 1).
        alpha_max (float): maximum learning rate.
        alpha_min (float): minimum learning rate.
        T_w (int): warmup steps.
        T_c (int): cosine decay steps.

    Returns:
        float: learning rate at step t.
    """
    if t < T_w:
        return alpha_max * t / T_w
    elif t >= T_w and t <= T_c:
        tau = (t - T_w) / (T_c-T_w)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * tau))
    else:
        return alpha_min
    
def run_gradient_clipping(parameters, max_l2_norm: float) -> None:
    eps = 1e-6
    grads = []

    # Collect gradients
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    if not grads:
        return  # no grads to clip

    all_grads = torch.cat(grads)
    total_norm = torch.norm(all_grads, p=2)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)  # in-place scaling