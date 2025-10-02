import torch.nn as nn
import torch
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rms_norm import RMSNorm

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = True,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_rope=use_rope,
                theta=theta,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,                    # (batch, seq) Long
        token_positions: torch.Tensor | None = None # (batch, seq) Long (absolute positions) for RoPE
    ) -> torch.Tensor:
        x = self.token_embeddings(token_ids)                 # (B, S, d_model)

        # 2) Default token positions if using RoPE and none are provided
        if self.use_rope and token_positions is None:
            B, S = token_ids.shape
            pos = torch.arange(S, device=token_ids.device)
            token_positions = pos.unsqueeze(0).expand(B, S)    # (B, S)

        # 3) Pass through blocks
        for block in self.layers:
            x = block(x, token_positions=token_positions)      # (B, S, d_model)

        # 4) Final norm + LM head
        x = self.ln_final(x)                                     # (B, S, d_model)
        logits = self.lm_head(x)                               # (B, S, vocab_size)
        return logits

