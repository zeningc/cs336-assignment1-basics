import torch
from einops import einsum
import math
import numpy as np
import numpy.typing as npt
from typing import Union, BinaryIO, IO
import os

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

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (inputs, labels) each of shape (batch_size, context_length), dtype Long, on `device`.
    Each input is a length-`context_length` slice, and its label is the same slice shifted by +1.
    """
    data = np.asarray(dataset, dtype=np.int64)
    n = data.shape[0]
    if n <= context_length:
        raise ValueError("dataset length must be > context_length")

    # Valid starts ensure labels i+context_length exists (last label index < n)
    # i ∈ [0, n - context_length - 1], and randint high is exclusive -> use high = n - context_length
    starts = np.random.randint(0, n - context_length, size=batch_size)

    # Build batches
    x_batch = np.stack([data[i : i + context_length] for i in starts], axis=0)
    y_batch = np.stack([data[i + 1 : i + 1 + context_length] for i in starts], axis=0)

    # To torch Long on device
    x = torch.from_numpy(x_batch).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_batch).to(device=device, dtype=torch.long)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> None:
    """
    Save model state, optimizer state, and iteration counter into a checkpoint file.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load checkpoint from file and restore model and optimizer states.

    Returns:
        iteration (int): The training iteration at the point of saving.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits and compute softmax.

    Args:
        logits: Tensor of shape (..., vocab_size) containing unnormalized scores
        temperature: Temperature parameter τ. Lower values make distribution more peaked,
                    higher values make it more uniform. Must be > 0.

    Returns:
        Probability distribution of shape (..., vocab_size)
    """
    scaled_logits = logits / temperature
    return softmax(scaled_logits, dim=-1)


def top_p_sampling(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Sample from a probability distribution using nucleus (top-p) sampling.

    Args:
        probs: Probability distribution of shape (..., vocab_size)
        p: Cumulative probability threshold (between 0 and 1)

    Returns:
        Sampled token indices of shape (...)
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the smallest set where cumulative probability >= p
    # We want to include the first token that pushes us over p
    mask = cumsum_probs - sorted_probs > p

    # Zero out probabilities outside the nucleus
    sorted_probs[mask] = 0.0

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from the truncated distribution
    # First, we need to map back to original indices
    # Sample from sorted distribution
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)

    # Map back to original vocabulary indices
    sampled_token = torch.gather(sorted_indices, -1, sampled_sorted_idx).squeeze(-1)

    return sampled_token


def generate(
    model: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int = None,
) -> torch.Tensor:
    """
    Generate text from a language model using autoregressive sampling.

    Args:
        model: Language model that takes token sequences and returns logits
        prompt_tokens: Initial token sequence of shape (batch_size, seq_len) or (seq_len,)
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for scaling logits (default 1.0 = no scaling)
        top_p: Threshold for nucleus sampling (default 1.0 = sample from full distribution)
        eos_token_id: Token ID for end-of-sequence. If generated, stop early.

    Returns:
        Generated token sequence of shape (batch_size, prompt_len + generated_len) or
        (prompt_len + generated_len,) matching input shape
    """
    model.eval()

    # Handle both batched and unbatched inputs
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    device = prompt_tokens.device
    batch_size = prompt_tokens.shape[0]

    # Start with the prompt
    generated = prompt_tokens.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model predictions
            # Model returns (batch_size, seq_len, vocab_size)
            logits = model(generated)

            # Get logits for the last position (predicting next token)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature scaling and softmax
            probs = softmax_with_temperature(next_token_logits, temperature)

            # Apply top-p sampling
            if top_p < 1.0:
                next_token = top_p_sampling(probs, top_p)
            else:
                # Sample from full distribution
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            # Check for end-of-sequence token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    if squeeze_output:
        generated = generated.squeeze(0)

    return generated