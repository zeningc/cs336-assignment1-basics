#!/usr/bin/env python3
"""
Run inference with a trained language model.
"""
import argparse
import json
import torch
from pathlib import Path

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.utils import generate


def load_model_from_checkpoint(
    checkpoint_path: Path,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int = None,
    use_rope: bool = True,
    rope_theta: float = 10000.0,
    device: str = "cpu",
) -> TransformerLM:
    """
    Load a TransformerLM model from a checkpoint file.
    """
    # Default d_ff to 4*d_model rounded to multiple of 64, matching trainer
    if d_ff is None:
        d_ff = (4 * d_model + 63) // 64 * 64

    # Create model with same architecture
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        use_rope=use_rope,
        theta=rope_theta,
        max_seq_len=context_length,
        device=device,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained language model."
    )

    # Model checkpoint and tokenizer paths
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint file (.pt)",
    )
    parser.add_argument(
        "--vocab",
        required=True,
        help="Path to vocab.json file",
    )
    parser.add_argument(
        "--merges",
        required=True,
        help="Path to merges.txt file",
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default=None,
        help="Optional: Path to hparams.json to load model architecture params (overrides individual args)",
    )

    # Model architecture parameters (matching run_llm_trainer.py)
    parser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length / max sequence length")
    parser.add_argument("--d_model", type=int, default=1024, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None, help="Feed-forward dimension (default: 4*d_model rounded to 64)")
    parser.add_argument("--use_rope", action="store_true", default=True, help="Use RoPE (default: True)")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt to start generation from",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (default 1.0 = no truncation)",
    )
    parser.add_argument(
        "--special_token",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens (must match training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt repeatedly)",
    )

    args = parser.parse_args()

    # Load hyperparameters from hparams.json if provided
    if args.hparams:
        with open(args.hparams, "r") as f:
            hparams = json.load(f)

        # Set architecture parameters from hparams
        # (command line args can still override these)
        if args.vocab_size is None:
            args.vocab_size = hparams.get("vocab_size")
        if "context_length" in hparams:
            args.context_length = hparams["context_length"]
        if "d_model" in hparams:
            args.d_model = hparams["d_model"]
        if "num_layers" in hparams:
            args.num_layers = hparams["num_layers"]
        if "num_heads" in hparams:
            args.num_heads = hparams["num_heads"]
        if "d_ff" in hparams and args.d_ff is None:
            args.d_ff = hparams["d_ff"]
        if "use_rope" in hparams:
            args.use_rope = hparams["use_rope"]
        if "rope_theta" in hparams:
            args.rope_theta = hparams["rope_theta"]

        print(f"Loaded hyperparameters from {args.hparams}")

    # Ensure vocab_size is set
    if args.vocab_size is None:
        raise ValueError("vocab_size must be specified either via --vocab_size or --hparams")

    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab} and {args.merges}...")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=list(args.special_token),
    )

    # Get EOS token ID
    eos_token = args.special_token[0]  # Use first special token as EOS
    eos_token_id = tokenizer._id_for.get(eos_token.encode("utf-8"))

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    print(f"Model architecture:")
    print(f"  vocab_size={args.vocab_size}, context_length={args.context_length}")
    print(f"  d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}")
    print(f"  d_ff={args.d_ff or '(auto)'}, use_rope={args.use_rope}, rope_theta={args.rope_theta}")

    model = load_model_from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        device=args.device,
    )

    print(f"\nModel loaded successfully! Using device: {args.device}") 
    print(f"Generation parameters: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    print("-" * 80)

    def generate_text(prompt: str):
        """Generate text from a prompt."""
        # Tokenize prompt
        if prompt:
            prompt_tokens = tokenizer.encode(prompt)
        else:
            # If no prompt, start with EOS token
            prompt_tokens = [eos_token_id] if eos_token_id is not None else [0]

        prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=args.device)

        # Generate
        print(f"\nPrompt: {repr(prompt)}")
        print("\nGenerating...")

        generated_tokens = generate(
            model=model,
            prompt_tokens=prompt_tokens_tensor,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
        )

        # Decode
        generated_tokens_list = generated_tokens.cpu().tolist()
        generated_text = tokenizer.decode(generated_tokens_list)

        # Display result
        print("\n" + "=" * 80)
        print("GENERATED TEXT:")
        print("=" * 80)
        print(generated_text)
        print("=" * 80)
        print(f"Total tokens: {len(generated_tokens_list)}")
        print("-" * 80)

    # Interactive or single generation
    if args.interactive:
        print("\nEntering interactive mode. Type your prompt and press Enter.")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                prompt = input("Prompt: ")
                if prompt.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                generate_text(prompt)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    else:
        # Single generation
        generate_text(args.prompt)


if __name__ == "__main__":
    main()
