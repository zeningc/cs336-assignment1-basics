# train_and_encode_bpe.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np

from cs336_basics.bpe_tokenizer_trainer import BPETokenizerTrainer
from cs336_basics.bpe_tokenizer import BPETokenizer  # the runtime tokenizer you pasted


# ---- Helpers to export vocab/merges in GPT-2 style (like the reference test does) ----

def save_merges_txt(merges: List[Tuple[bytes, bytes]], out_path: Path) -> None:
    """
    Save merges as GPT-2-style merges.txt:
        <token1_str> <token2_str>
    one pair per line, in the order they were created.
    """
    b2u = BPETokenizer.gpt2_bytes_to_unicode()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for a, b in merges:
            ta = BPETokenizer.encode_token_bytes(a, b2u)
            tb = BPETokenizer.encode_token_bytes(b, b2u)
            f.write(f"{ta} {tb}\n")


def save_vocab_json(vocab: Dict[int, bytes], out_path: Path) -> None:
    """
    Save vocab as GPT-2-style vocab.json:
        { "<token_str>": <id>, ... }
    """
    b2u = BPETokenizer.gpt2_bytes_to_unicode()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # convert {id: bytes} -> {token_str: id}
    as_text = {BPETokenizer.encode_token_bytes(tok_bytes, b2u): tok_id for tok_id, tok_bytes in vocab.items()}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(as_text, f, ensure_ascii=False, indent=2)


# ---- Encoding the training set ----

def encode_corpus_to_bin(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_tokens_path: Path,
    chunk_size: int = 8192,
) -> int:
    """
    Stream-encode the training corpus to a flat int64 token file (.bin).
    Returns total number of tokens written.
    """
    output_tokens_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    buf: List[int] = []

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         output_tokens_path.open("wb") as fout:

        for line in fin:
            # You can also use encode_iterable([line]) if you want to stream per-line,
            # but encode(line) is fine here.
            ids = tokenizer.encode(line)
            buf.extend(ids)

            # Flush in chunks to avoid keeping everything in RAM
            while len(buf) >= chunk_size:
                chunk = np.asarray(buf[:chunk_size], dtype=np.int64)
                chunk.tofile(fout)
                total += chunk_size
                buf = buf[chunk_size:]

        # Flush remaining
        if buf:
            chunk = np.asarray(buf, dtype=np.int64)
            chunk.tofile(fout)
            total += len(chunk)

    return total


# ---- Main CLI ----

def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer and encode training/validation sets to token IDs."
    )
    parser.add_argument(
        "--train_input",
        required=True,
        help="Path to training text file (e.g. corpus.en).",
    )
    parser.add_argument(
        "--train_output",
        required=True,
        help="Output directory; will contain vocab.json, merges.txt, tokens.bin.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Total vocabulary size (including special tokens).",
    )
    parser.add_argument(
        "--special_token",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special token(s) that should be kept as single tokens.",
    )
    parser.add_argument(
        "--val_input",
        default=None,
        help="Optional: Path to validation text file for encoding with same tokenizer.",
    )
    parser.add_argument(
        "--val_output",
        default=None,
        help="Optional: Output path for validation tokens.bin.",
    )
    args = parser.parse_args()

    train_input_path = Path(args.train_input)
    train_out_dir = Path(args.train_output)
    train_out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = train_out_dir / "vocab.json"
    merges_path = train_out_dir / "merges.txt"
    tokens_path = train_out_dir / "tokens.bin"

    # 1) Train BPE on the input corpus
    trainer = BPETokenizerTrainer(
        input_path=train_input_path,
        vocab_size=args.vocab_size,
        special_tokens=list(args.special_token),
    )
    vocab, merges = trainer.merge()  # vocab: {id: bytes}, merges: list[(bytes, bytes)]

    # 2) Save vocab & merges in GPT-2 text formats
    save_vocab_json(vocab, vocab_path)
    save_merges_txt(merges, merges_path)
    print(f"Saved vocab.json -> {vocab_path}")
    print(f"Saved merges.txt -> {merges_path}")

    # 3) Build runtime tokenizer from the learned artifacts
    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=list(args.special_token))

    # 4) Encode the training corpus into a flat int64 token file
    total_tokens = encode_corpus_to_bin(tokenizer, train_input_path, tokens_path)
    print(f"Encoded training set -> {tokens_path} ({total_tokens} tokens, int64)")

    # 5) Optionally encode validation set with the same tokenizer
    if args.val_input and args.val_output:
        val_input_path = Path(args.val_input)
        val_output_path = Path(args.val_output) / "tokens.bin"
        print(f"Encoding validation set: {val_input_path} -> {val_output_path}")
        val_total_tokens = encode_corpus_to_bin(tokenizer, val_input_path, val_output_path)
        print(f"Encoded validation set -> {val_output_path} ({val_total_tokens} tokens, int64)")
    elif args.val_input or args.val_output:
        print("Warning: Both --val_input and --val_output must be specified to encode validation set")


if __name__ == "__main__":
    main()
