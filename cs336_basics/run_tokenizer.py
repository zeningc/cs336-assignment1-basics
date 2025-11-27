# train_and_encode_bpe.py
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np

from cs336_basics.bpe_tokenizer_trainer import BPETokenizerTrainer
from cs336_basics.bpe_tokenizer import BPETokenizer  # the runtime tokenizer you pasted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ---- Helpers to export vocab/merges in GPT-2 style (like the reference test does) ----

def save_merges_txt(merges: List[Tuple[bytes, bytes]], out_path: Path) -> None:
    """
    Save merges as GPT-2-style merges.txt:
        <token1_str> <token2_str>
    one pair per line, in the order they were created.
    """
    logging.info(f"Saving {len(merges)} merge rules to {out_path}")
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
    logging.info(f"Saving vocabulary of size {len(vocab)} to {out_path}")
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
    logging.info(f"Starting encoding of {input_path} to {output_tokens_path}")
    output_tokens_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    buf: List[int] = []
    lines_processed = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
         output_tokens_path.open("wb") as fout:

        for line in fin:
            # You can also use encode_iterable([line]) if you want to stream per-line,
            # but encode(line) is fine here.
            ids = tokenizer.encode(line)
            buf.extend(ids)
            lines_processed += 1

            # Flush in chunks to avoid keeping everything in RAM
            while len(buf) >= chunk_size:
                chunk = np.asarray(buf[:chunk_size], dtype=np.int64)
                chunk.tofile(fout)
                total += chunk_size
                buf = buf[chunk_size:]

            # Log progress every 10000 lines
            if lines_processed % 10000 == 0:
                logging.info(f"Processed {lines_processed} lines, {total} tokens written")

        # Flush remaining
        if buf:
            chunk = np.asarray(buf, dtype=np.int64)
            chunk.tofile(fout)
            total += len(chunk)

    logging.info(f"Encoding complete: {lines_processed} lines processed, {total} tokens total")
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Number of parallel workers for pre-tokenization (default: 20).",
    )
    args = parser.parse_args()

    train_input_path = Path(args.train_input)
    train_out_dir = Path(args.train_output)
    train_out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = train_out_dir / "vocab.json"
    merges_path = train_out_dir / "merges.txt"
    tokens_path = train_out_dir / "tokens.bin"

    # 1) Train BPE on the input corpus
    logging.info("="*60)
    logging.info("Starting BPE tokenizer training")
    logging.info(f"Input file: {train_input_path}")
    logging.info(f"Target vocab size: {args.vocab_size}")
    logging.info(f"Special tokens: {args.special_token}")
    logging.info(f"Number of workers: {args.num_workers}")
    logging.info("="*60)

    trainer = BPETokenizerTrainer(
        input_path=train_input_path,
        vocab_size=args.vocab_size,
        special_tokens=list(args.special_token),
        num_workers=args.num_workers,
    )
    logging.info("BPETokenizerTrainer initialized, starting merge process...")
    logging.info("This may take a while for large corpora...")

    vocab, merges = trainer.merge()  # vocab: {id: bytes}, merges: list[(bytes, bytes)]

    logging.info(f"BPE training complete! Final vocab size: {len(vocab)}")

    # 2) Save vocab & merges in GPT-2 text formats
    save_vocab_json(vocab, vocab_path)
    save_merges_txt(merges, merges_path)
    print(f"Saved vocab.json -> {vocab_path}")
    print(f"Saved merges.txt -> {merges_path}")

    # 3) Build runtime tokenizer from the learned artifacts
    logging.info("Building runtime BPE tokenizer from learned vocabulary and merges")
    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=list(args.special_token))

    # 4) Encode the training corpus into a flat int64 token file
    logging.info("="*60)
    logging.info("Encoding training corpus to token IDs")
    logging.info("="*60)
    total_tokens = encode_corpus_to_bin(tokenizer, train_input_path, tokens_path)
    print(f"Encoded training set -> {tokens_path} ({total_tokens} tokens, int64)")

    # 5) Optionally encode validation set with the same tokenizer
    if args.val_input and args.val_output:
        val_input_path = Path(args.val_input)
        val_output_path = Path(args.val_output) / "tokens.bin"
        logging.info("="*60)
        logging.info("Encoding validation corpus to token IDs")
        logging.info("="*60)
        val_total_tokens = encode_corpus_to_bin(tokenizer, val_input_path, val_output_path)
        print(f"Encoded validation set -> {val_output_path} ({val_total_tokens} tokens, int64)")
    elif args.val_input or args.val_output:
        logging.warning("Both --val_input and --val_output must be specified to encode validation set")

    logging.info("="*60)
    logging.info("All tasks completed successfully!")
    logging.info("="*60)


if __name__ == "__main__":
    main()
