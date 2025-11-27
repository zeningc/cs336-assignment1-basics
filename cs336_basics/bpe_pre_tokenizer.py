import os
import logging
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO, Optional, List, Tuple
import regex as re
from cs336_basics.bpe_tokenizer import BPETokenizer

class BPEPreTokenizer:
    NUM_PROCESSES = 4

    def __init__(
            self,
            input_path: str | os.PathLike,
            special_tokens: Optional[List[str]] = None,
            split_token_for_chunking: Optional[str] = "<|endoftext|>",
    ):
        """
        special_tokens: list of strings to remove before pre-tokenization (kept separate for training).
        split_token_for_chunking: special token used to align chunk boundaries.
        """
        self.input_path = input_path
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]
        self.special_tokens = special_tokens
        self.split_token_for_chunking = split_token_for_chunking or special_tokens[0]

        # Compile string-level regexes
        self._pat_re = re.compile(BPETokenizer.PAT)
        # Special-token alternation (string regex)
        esc = [re.escape(tok) for tok in self.special_tokens]
        self._specials_alt = "(" + "|".join(esc) + ")"  # capturing to keep them if needed

    @staticmethod
    def find_chunk_boundaries(
            file: BinaryIO,
            desired_num_chunks: int,
            split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    @staticmethod
    def _bytes_tuple_from_str_token(s: str) -> Tuple[bytes, ...]:
        """
        Encode a string pre-token to UTF-8 bytes, then split into a tuple of single-byte objects.
        Example: 'low' -> b'low' -> (b'l', b'o', b'w')
        """
        b = s.encode("utf-8")
        return tuple(bytes([x]) for x in b)

    def _count_chunk_pre_tokens(
            self,
            filename: str,
            start: int,
            end: int,
    ) -> Counter:
        """
        Worker-safe function: re-compile regexes inside the process (avoids pickling compiled objects).
        Streams through the chunk in smaller sub-chunks to avoid loading entire chunk into RAM.
        1) Read chunk bytes in sub-chunks
        2) Decode to str
        3) Remove special tokens (split and drop specials)
        4) finditer(PAT) over remaining text
        5) Convert each match to tuple[bytes] and count
        """
        pat_re = re.compile(BPETokenizer.PAT)
        specials_re = re.compile(self._specials_alt)

        # Process in sub-chunks to avoid loading entire chunk into memory
        # For a 2GB chunk, this processes 100MB at a time
        sub_chunk_size = 100 * 1024 * 1024  # 100 MB sub-chunks
        counts = Counter()

        total_size = end - start
        current_pos = start
        leftover = b""  # Buffer for incomplete tokens at chunk boundaries
        sub_chunks_processed = 0

        # Log chunk processing start
        chunk_mb = total_size / (1024 * 1024)
        logging.info(f"Worker processing chunk {start}-{end} ({chunk_mb:.1f} MB)")

        with open(filename, "rb") as f:
            while current_pos < end:
                # Calculate how much to read
                bytes_to_read = min(sub_chunk_size, end - current_pos)

                f.seek(current_pos)
                chunk_bytes = f.read(bytes_to_read)

                # Combine with leftover from previous iteration
                chunk_bytes = leftover + chunk_bytes

                # Try to decode; if we end mid-character, save the incomplete bytes
                try:
                    text = chunk_bytes.decode("utf-8")
                    leftover = b""
                except UnicodeDecodeError:
                    # Find the last valid UTF-8 character boundary
                    # Keep trying to decode smaller chunks from the end
                    for i in range(1, min(5, len(chunk_bytes)) + 1):  # UTF-8 chars are max 4 bytes
                        try:
                            text = chunk_bytes[:-i].decode("utf-8")
                            leftover = chunk_bytes[-i:]
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If we can't decode at all, skip this problematic section
                        text = chunk_bytes.decode("utf-8", errors="ignore")
                        leftover = b""

                # Remove special tokens by splitting on them and keeping only non-special parts
                parts = specials_re.split(text)  # [text, tok, text, tok, ...]
                content_parts = [p for p in parts if p not in self.special_tokens and p != ""]

                for piece in content_parts:
                    for m in pat_re.finditer(piece):
                        token_str = m.group(0)
                        key = self._bytes_tuple_from_str_token(token_str)
                        counts[key] += 1

                current_pos += bytes_to_read
                sub_chunks_processed += 1

                # Log progress every 10 sub-chunks (every ~1GB for 100MB sub-chunks)
                if sub_chunks_processed % 10 == 0:
                    progress_pct = ((current_pos - start) / total_size) * 100
                    logging.info(f"Worker chunk {start}-{end}: {progress_pct:.1f}% complete ({sub_chunks_processed} sub-chunks processed)")

        # Process any leftover bytes at the end
        if leftover:
            try:
                text = leftover.decode("utf-8")
                parts = specials_re.split(text)
                content_parts = [p for p in parts if p not in self.special_tokens and p != ""]

                for piece in content_parts:
                    for m in pat_re.finditer(piece):
                        token_str = m.group(0)
                        key = self._bytes_tuple_from_str_token(token_str)
                        counts[key] += 1
            except UnicodeDecodeError:
                # Ignore incomplete UTF-8 at very end
                pass

        logging.info(f"Worker chunk {start}-{end}: Complete! Found {len(counts)} unique token types")
        return counts

    def pre_tokenize_count_parallel(
            self,
            num_workers: int = 20,
    ) -> Counter:
        """
        Parallel pre-tokenization count:
        - Removes special tokens
        - Splits using GPT-2 PAT
        - Returns Counter[tuple[bytes], int]
        """
        # Boundaries
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(
                f,
                num_workers,
                self.split_token_for_chunking.encode('utf-8')
            )

        jobs = list(zip(boundaries[:-1], boundaries[1:]))

        if not jobs:
            return Counter()

        with Pool(processes=min(num_workers, len(jobs))) as pool:
            results = pool.starmap(
                self._count_chunk_pre_tokens,
                [
                    (
                        self.input_path,
                        s,
                        e,
                    )
                    for (s, e) in jobs
                ],
            )

        total = Counter()
        for c in results:
            total.update(c)
        return total
