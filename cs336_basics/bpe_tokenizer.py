from typing import Iterable, Dict, Iterator, List, Tuple
import regex as re
import json


class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    REPLACEMENT_BYTES = "\uFFFD".encode("utf-8")

    @staticmethod
    def gpt2_bytes_to_unicode() -> Dict[int, str]:
        """
        GPT-2's reversible byte<->unicode mapping.
        Maps raw bytes to printable unicode strings for vocab.json/merges.txt.
        """
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))

    @staticmethod
    def encode_token_bytes(b: bytes, b2u: Dict[int, str]) -> str:
        """Encode a raw bytes token into GPT-2-style unicode string."""
        return "".join(b2u[x] for x in b)

    @staticmethod
    def decode_gpt2_token_str(s: str, u2b: Dict[str, int]) -> bytes:
        """Decode a GPT-2-style unicode string back to raw bytes."""
        return bytes(u2b[c] for c in s)

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self._id_for: Dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # Precompile regexes
        self._pat_re = re.compile(self.PAT)

        if self.special_tokens:
            specials_sorted = sorted(set(self.special_tokens), key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(s) for s in specials_sorted) + ")"
            self._specials_re = re.compile(pattern)
        self._id_for: Dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # intern single-byte tokens to avoid re-allocating bytes([x])
        self._single_bytes = [bytes([i]) for i in range(256)]
        # optional small cache for repeated words
        self._word_cache: Dict[bytes, List[int]] = {}
        # build (a,b) -> id(a+b) so we don't concat bytes in the hot loop
        self._pair2id: Dict[tuple[bytes, bytes], int] = {}
        for a, b in self.merges:
            ab = a + b
            tid = self._id_for.get(ab)
            if tid is not None:
                self._pair2id[(a, b)] = tid

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Load BPETokenizer from GPT-2 style vocab.json and merges.txt files."""
        # Create unicode-to-byte mapping for decoding GPT-2 format
        b2u = cls.gpt2_bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}

        # Load vocab.json and decode from GPT-2 unicode format to bytes
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)  # {"token_str": id}
        vocab = {}
        for token_str, token_id in raw.items():
            token_bytes = cls.decode_gpt2_token_str(token_str, u2b)
            vocab[token_id] = token_bytes

        # Load merges.txt and decode from GPT-2 unicode format to bytes
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # skip blanks/comments
                parts = line.split()
                if len(parts) != 2:
                    continue
                a_str, b_str = parts
                a_bytes = cls.decode_gpt2_token_str(a_str, u2b)
                b_bytes = cls.decode_gpt2_token_str(b_str, u2b)
                merges.append((a_bytes, b_bytes))

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pre: str) -> List[int]:
        b = pre.encode("utf-8")
        if not b:
            return []

        # cache by raw bytes of the pretoken
        cached = self._word_cache.get(b)
        if cached is not None:
            return cached[:]  # copy to avoid external mutation

        # start from 1-byte tokens using interned table
        seq: List[bytes] = [self._single_bytes[x] for x in b]
        if len(seq) == 1:
            tid = self._id_for.get(seq[0])
            out = [tid] if tid is not None else []
            self._word_cache[b] = out[:]
            return out

        # Iteratively merge: at each step, pick neighbor pair with the smallest id
        while len(seq) > 1:
            best_id = None
            best_pos = -1

            # lookup pair ids via tuple (no bytes concatenation here)
            for i in range(len(seq) - 1):
                tid = self._pair2id.get((seq[i], seq[i + 1]))
                if tid is not None and (best_id is None or tid < best_id):
                    best_id = tid
                    best_pos = i

            if best_pos < 0:
                break

            # merge in place (we do need the merged bytes once)
            merged = seq[best_pos] + seq[best_pos + 1]
            seq[best_pos:best_pos + 2] = [merged]

        # map to ids, skip unknowns
        out_ids: List[int] = []
        for tok in seq:
            tid = self._id_for.get(tok)
            if tid is not None:
                out_ids.append(tid)

        self._word_cache[b] = out_ids[:]
        return out_ids


    def _split_on_specials(self, text: str) -> Iterator[Tuple[bool, str]]:
        """
        Split text so that special tokens appear as standalone elements in the list.
        """
        if not self.special_tokens:
            yield False, text
            return

        pos = 0
        for m in self._specials_re.finditer(text):
            start, end = m.span()
            if start > pos:
                yield False, text[pos:start]  # non-special span
            s = m.group(0)
            yield True, s  # special token
            pos = end
        if pos < len(text):
            yield False, text[pos:]


    def encode(self, text: str) -> list[int]:

        ids: List[int] = []
        for is_special, piece in self._split_on_specials(text):
            if is_special:
                # Emit special id directly
                sid = self._id_for[piece.encode("utf-8")]
                ids.append(sid)
            else:
                for m in self._pat_re.finditer(piece):
                    pre = m.group(0)
                    ids.extend(self._encode_pretoken(pre))
        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for is_special, piece in self._split_on_specials(text):
                if is_special:
                    yield self._id_for[piece.encode("utf-8")]
                else:
                    for m in self._pat_re.finditer(piece):
                        pre = m.group(0)
                        for tid in self._encode_pretoken(pre):
                            yield tid

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for i in ids:
            b = self.vocab.get(i)
            if b is None:
                # Unknown id: skip or insert replacement bytes; we skip bytes and add U+FFFD char
                buf.extend(self.REPLACEMENT_BYTES)
            else:
                buf.extend(b)
        return bytes(buf).decode("utf-8", errors="replace")
