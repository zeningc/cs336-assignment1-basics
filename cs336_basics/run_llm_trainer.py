# train_lm.py
import os, math, time, argparse, json, random
import numpy as np
import torch
from torch import nn
from typing import Tuple
import numpy.typing as npt

# ---- your modules ----
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW              
from cs336_basics.utils import cosine_lr_schedule, save_checkpoint, load_checkpoint, run_get_batch, cross_entropy, run_gradient_clipping

# ---------- Cosine LR wrapper ----------
class CosineWithWarmup:
    def __init__(self, opt: torch.optim.Optimizer, max_lr, min_lr, warmup_iters, cosine_iters):
        self.opt = opt
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cosine_iters = cosine_iters
        self.t = 0

    def step(self):
        lr = cosine_lr_schedule(self.t, self.max_lr, self.min_lr, self.warmup_iters, self.cosine_iters)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.t += 1
        return lr

# ---------- Eval ----------
@torch.no_grad()
def evaluate(model: nn.Module, memmap: npt.NDArray[np.int64], batches: int, batch_size: int, ctx_len: int, device) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = run_get_batch(memmap, batch_size, ctx_len, device)
        logits = model(x)                          # (B, S, V)
        loss = cross_entropy(logits, y)  # implement or import your stable CE
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train_tokens_path", type=str, required=True)   # memmap .npy (int64)
    p.add_argument("--val_tokens_path", type=str, required=True)
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, default=1024)
    # model
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--num_layers", type=int, default=24)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=None)  # default None -> 4*d_model rounded to multiple of 64
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--use_rope", action="store_true", default=True)
    p.add_argument("--tie_weights", action="store_true", default=False)
    # opt & schedule
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=2000)
    p.add_argument("--cosine_iters", type=int, default=98000)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # loop
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--log_every", type=int, default=100)
    # io
    p.add_argument("--out_dir", type=str, default="runs/exp4")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="cs336-transformer")
    p.add_argument("--wandb_run", type=str, default="run")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # seed
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # data: memmap numpy arrays (1D int64 token stream)
    train_tokens = np.memmap(args.train_tokens_path, dtype=np.int64, mode="r")
    val_tokens   = np.memmap(args.val_tokens_path,   dtype=np.int64, mode="r")

    device = torch.device(args.device)

    # model
    d_ff = args.d_ff or ( (4*args.d_model + 63) // 64 * 64 )  # 4*d rounded to multiple of 64
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        use_rope=args.use_rope,
        theta=args.rope_theta,
        max_seq_len=args.context_length,
        # tie_weights=args.tie_weights,
        device=device,
        dtype=torch.float32,
    ).to(device)

    # optimizer
    opt = AdamW(model.parameters(), lr=args.lr_max, betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay)
    sched = CosineWithWarmup(opt, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_iters)

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # set enabled=True if you add autocast + BF16/FP16
    start_iter = 0

    # optional W&B
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # resume
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, opt)
        # If you stored scheduler step t, restore it; here we approximate by setting to start_iter
        sched.t = start_iter

    # training
    model.train()
    t0 = time.time()
    for it in range(start_iter, args.iters):
        lr = sched.step()  # set LR for this step

        x, y = run_get_batch(train_tokens, args.batch_size, args.context_length, args.device)

        # forward
        logits = model(x)  # (B, S, V)
        loss = cross_entropy(logits, y)

        # backward
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping
        if args.grad_clip and args.grad_clip > 0:
            run_gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)

        opt.step()

        if (it + 1) % args.log_every == 0:
            tok_per_step = args.batch_size * args.context_length
            dt = time.time() - t0
            tps = (tok_per_step * args.log_every) / max(dt, 1e-9)
            msg = f"it {it+1:7d} | lr {lr:.3e} | loss {loss.item():.4f} | toks/s {int(tps)}"
            print(msg, flush=True)
            if args.use_wandb:
                import wandb
                wandb.log({"train/loss": loss.item(), "lr": lr, "toks_per_sec": tps}, step=it+1)
            t0 = time.time()

        if (it + 1) % args.eval_every == 0:
            val_loss = evaluate(model, val_tokens, args.eval_batches, args.batch_size, args.context_length, device)
            print(f"[eval] it {it+1} | val_loss {val_loss:.4f}")
            if args.use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss}, step=it+1)

        if (it + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{it+1:07d}.pt")
            save_checkpoint(model, opt, it + 1, ckpt_path)
            print(f"[ckpt] saved to {ckpt_path}")

    # final save
    ckpt_path = os.path.join(args.out_dir, "ckpt_final.pt")
    save_checkpoint(model, opt, args.iters, ckpt_path)
    print(f"[ckpt] saved final to {ckpt_path}")

if __name__ == "__main__":
    main()
