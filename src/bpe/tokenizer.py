# src/bpe/tokenizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple, Optional

# Public factory
def get_tokenizer(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    special_tokens: Optional[List[str]] = None,
):
    """
    Build a bytes-level BPE tokenizer compatible with GPT-2 style vocab/merges.

    Args:
        vocab: mapping from token id -> token bytes
        merges: list of (left_bytes, right_bytes) merge rules in priority order
        special_tokens: optional list of strings that should be treated as indivisible tokens

    Returns:
        An object with methods: encode(text: str) -> List[int],
        encode_iterable(iterable: Iterable[str]) -> Iterator[int],
        decode(ids: List[int]) -> str
    """
    return _Tokenizer(vocab, merges, special_tokens or [])


@dataclass
class _Tokenizer:
    id_to_bytes: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]
    special_tokens: List[str]

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], specials: List[str]) -> None:
        self.id_to_bytes = dict(vocab)  # copy
        self._bytes_to_id: Dict[bytes, int] = {v: k for k, v in self.id_to_bytes.items()}
        # Merge ranks: lower index = higher priority
        self._rank: Dict[Tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        self.merges = merges
        # Precompute special tokens as bytes, map to id (append must have ensured id exists)
        self.special_tokens = specials
        self._special_bytes: List[bytes] = [s.encode("utf-8") for s in self.special_tokens]
        self._special_id: Dict[bytes, int] = {}
        for sb in self._special_bytes:
            tid = self._bytes_to_id.get(sb)
            if tid is not None:
                self._special_id[sb] = tid
        # Greedy longest-match
        self._special_sorted = sorted(self._special_bytes, key=len, reverse=True)

    # ------------------------ public API ------------------------

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        b = text.encode("utf-8")
        return list(self._encode_bytes(b))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Stream over chunks to keep memory low
        for chunk in iterable:
            if not chunk:
                continue
            b = chunk.encode("utf-8")
            yield from self._encode_bytes(b)

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        b = b"".join(self.id_to_bytes[i] for i in ids)
        return b.decode("utf-8", errors="strict")

    # ------------------------ internal helpers ------------------------

    def _encode_bytes(self, b: bytes) -> Iterator[int]:
        """
        Encode a single byte sequence, honoring special tokens if configured.
        Strategy: left-to-right scan; on a special-token match, flush current buffer
        via BPE, then emit the special token id; otherwise accumulate bytes.
        """
        if not b:
            return
        pos = 0
        buf = bytearray()
        n = len(b)
        while pos < n:
            matched = False
            # Try to match any special token at current position (longest-first)
            if self._special_sorted:
                for sb in self._special_sorted:
                    if not sb:
                        continue
                    L = len(sb)
                    if pos + L <= n and b[pos:pos+L] == sb:
                        # Flush buf
                        if buf:
                            yield from self._bpe_segment(bytes(buf))
                            buf.clear()
                        # Emit special token id
                        tid = self._special_id.get(sb)
                        if tid is None:
                            # If not in vocab as bytes, fall back to normal BPE on its bytes
                            yield from self._bpe_segment(sb)
                        else:
                            yield tid
                        pos += L
                        matched = True
                        break
            if matched:
                continue
            # Accumulate a normal byte
            buf.append(b[pos])
            pos += 1
        if buf:
            yield from self._bpe_segment(bytes(buf))

    def _bpe_segment(self, b: bytes) -> Iterator[int]:
        """
        Greedy BPE merge on a bytes sequence using merge ranks.
        Start with list of single-byte tokens; repeatedly merge the lowest-rank
        adjacent pair that appears in ranks; stop when no merge applies.
        """
        if not b:
            return
        # Represent as list of bytes objects
        tokens: List[bytes] = [bytes([ch]) for ch in b]
        # Fast path: length 1
        if len(tokens) == 1:
            tid = self._bytes_to_id.get(tokens[0])
            if tid is None:
                raise KeyError(f"Unknown token bytes: {tokens[0]!r}")
            yield tid
            return

        # Helper: get rank or sentinel
        BIG = 1 << 30
        def get_rank_pair(a: bytes, c: bytes) -> int:
            r = self._rank.get((a, c))
            return r if r is not None else BIG

        # Initial ranks for adjacent pairs
        ranks = [get_rank_pair(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

        while True:
            best_rank = min(ranks) if ranks else BIG
            if best_rank == BIG:
                break  # no merges
            i = ranks.index(best_rank)  # position to merge
            # Merge tokens[i] and tokens[i+1]
            merged = tokens[i] + tokens[i+1]
            tokens[i:i+2] = [merged]
            # Update ranks around i (list length reduced by 1)
            if i-1 >= 0:
                ranks[i-1] = get_rank_pair(tokens[i-1], tokens[i])
            if i < len(tokens)-1:
                # replace/append rank at i
                if i < len(ranks):
                    ranks[i] = get_rank_pair(tokens[i], tokens[i+1])
                else:
                    ranks.append(get_rank_pair(tokens[i], tokens[i+1]))
                # remove the extra rank slot after merge
                if i+1 < len(ranks):
                    del ranks[i+1]
            else:
                # merged at end; cut off any trailing rank
                if i < len(ranks):
                    del ranks[i:]

        # Map to ids
        for t in tokens:
            tid = self._bytes_to_id.get(t)
            if tid is None:
                raise KeyError(f"Unknown token bytes after BPE merge: {t!r}")
            yield tid
