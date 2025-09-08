from collections import Counter, defaultdict
import os

from cs336_basics.bpe_pre_tokenizer import BPEPreTokenizer


class BPETokenizerTrainer:

    def __init__(self, input_path: str | os.PathLike,
                 vocab_size: int,
                 special_tokens: list[str],
                 **kwargs
                 ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pre_tokenizer = BPEPreTokenizer(self.input_path, self.special_tokens)

    def prepare_input(self) -> Counter:
        return self.pre_tokenizer.pre_tokenize_count_parallel()


    def merge(self):

        def _merge_tuple_all_occurrences(tok: tuple[bytes, ...], a: bytes, b: bytes) -> tuple[bytes, ...]:
            """Replace every occurrence of (a,b) in tok with (a+b)."""
            out = []
            i = 0
            ab = a + b
            n = len(tok)
            while i < n:
                if i + 1 < n and tok[i] == a and tok[i + 1] == b:
                    out.append(ab)
                    i += 2
                else:
                    out.append(tok[i])
                    i += 1
            return tuple(out)

        def _pairs_of(token: tuple[bytes, ...]) -> Counter:
            """Unweighted per-token pair multiset."""
            c = Counter()
            for a, b in zip(token, token[1:]):
                c[(a, b)] += 1
            return c

        freq = self.prepare_input()
        merges = []
        vocab = {}

        for i, special_token in enumerate(self.special_tokens):
            vocab[i] = special_token.encode('utf-8')
        for i in range(256):
            vocab[i + len(self.special_tokens)] = bytes([i])

        pair2tokens = defaultdict(set)
        tok2pairs = defaultdict(Counter)
        pair_cnt = Counter()

        for tok, cnt in freq.items():
            pc = _pairs_of(tok)  # unweighted per-token occurrences
            tok2pairs[tok] = pc
            for p, c in pc.items():
                pair_cnt[p] += c * cnt
                pair2tokens[p].add(tok)

        while len(vocab) < self.vocab_size:
            if not pair_cnt:
                break

            (a, b), _ = max(pair_cnt.items(), key=lambda kv: (kv[1], kv[0]))

            # add merged token to vocab
            merged_token = a + b
            vocab[len(vocab)] = merged_token
            merges.append((a, b))

            affected_tokens = pair2tokens.pop((a, b), set())
            for old_tok in affected_tokens:
                old_pairs = tok2pairs.pop(old_tok, {})
                cnt = freq.pop(old_tok, 0)
                for p, c in old_pairs.items():
                    pair_cnt[p] -= cnt * c
                    s = pair2tokens.get(p)
                    if s is not None:
                        s.discard(old_tok)
                        if not s:
                            pair2tokens.pop(p, None)
                    if pair_cnt[p] <= 0:
                        pair_cnt.pop(p, None)

                new_tok = _merge_tuple_all_occurrences(old_tok, a, b)
                freq[new_tok] += cnt
                pc = _pairs_of(new_tok)
                for p, c in pc.items():
                    pair_cnt[p] += c * cnt
                    tok2pairs[new_tok][p] += c
                    pair2tokens[p].add(new_tok)

        return vocab, merges







        return vocab, merges








