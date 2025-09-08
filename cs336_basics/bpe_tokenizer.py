from typing import Iterable, Dict, Iterator, List
import regex as re
import json

class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    REPLACEMENT_BYTES = "\uFFFD".encode("utf-8")

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

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        vocab = {idx: tok.encode("utf-8") for tok, idx in raw.items()}
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # skip blanks/comments
                parts = line.split()
                a, b = parts
                merges.append((a.encode("utf-8"), b.encode("utf-8")))
        return BPETokenizer(vocab, merges, special_tokens)

    def _is_special(self, s: str) -> bool:
        if not self.special_tokens:
            return False
        # exact match against any special (string form)
        return s in self.special_tokens

    def _encode_pretoken(self, pre: str) -> List[int]:
        """
        Encode a single pre-token:
          - start as a tuple of single-byte tokens (b'x' length-1 each)
          - apply merges in *the order they were created* (assignment requirement)
          - map resulting byte-chunks to ids
        """
        # Start as a sequence of one-byte tokens (bytes objects length==1)
        b = pre.encode("utf-8")
        seq: List[bytes] = [bytes([x]) for x in b]

        if not seq:
            return []

        # Apply merges in creation order, greedily replacing all occurrences each time
        for a, c in self.merges:
            if not seq:
                break
            if len(seq) == 1:
                # A single token cannot contain a pair
                continue
            ab = a + c
            # Scan-and-rewrite: replace every (a,c) with (ab)
            out: List[bytes] = []
            i = 0
            changed = False
            n = len(seq)
            while i < n:
                if i + 1 < n and seq[i] == a and seq[i + 1] == c:
                    out.append(ab)
                    i += 2
                    changed = True
                else:
                    out.append(seq[i])
                    i += 1
            seq = out

        # Finally map byte-chunks to ids
        out_ids: List[int] = []
        for tok in seq:
            tid = self._id_for.get(tok)
            out_ids.append(tid)
        return out_ids

    def _split_on_specials(self, text: str) -> List[str]:
        """
        Split text so that special tokens appear as standalone elements in the list.
        """
        if not self.special_tokens:
            return [text]

        parts = self._specials_re.split(text)  # [text, special, text, special, ...]
        # keep empty pieces? Typically harmless, PAT will ignore.
        return [p for p in parts if p != ""]


    def encode(self, text: str) -> list[int]:

        ids: List[int] = []
        for piece in self._split_on_specials(text):
            if self._is_special(piece):
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
            for piece in self._split_on_specials(text):
                if self._is_special(piece):
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
