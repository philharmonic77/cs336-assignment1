import regex as re
import json
from typing import Iterable, Iterator, Self
from .pretokenize import compile_special_pattern, split_by_special_tokens


class Tokenizer(object):
    def  __init__(self,
                vocab: dict[int, bytes], # Invariant: vocab must not contain empty byte token b""
                merges: list[tuple[bytes, bytes]], 
                special_tokens: list[str] | None = None) -> None:
        
        max_id = max(vocab.keys())
        self.id_to_bytes: list[bytes] = [b""] * (max_id + 1)
        self.bytes_to_id: dict[bytes, int] = {}

        for i, b in vocab.items():
            self.id_to_bytes[i] = b
            self.bytes_to_id[b] = i

        if any(x == b"" for x in self.id_to_bytes):
            raise ValueError("vocab ids must be contiguous from 0..max_id")

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        for r, merge in enumerate(merges):
            self.merge_ranks[merge]  = r

        self.special_token_to_id: dict[str, int] = {}
        if special_tokens:
            for st in special_tokens:
                sb = st.encode("utf-8")
                if sb in self.bytes_to_id:
                    self.special_token_to_id[st] = self.bytes_to_id[sb]
                else:
                    new_id = len(self.id_to_bytes)
                    self.id_to_bytes.append(sb)
                    self.bytes_to_id[sb] = new_id
                    self.special_token_to_id[st] = new_id

        self.special_tokens = special_tokens or []
        self.special_pattern = compile_special_pattern(self.special_tokens)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pretoken_pattern = re.compile(PAT)

        self.bpe_cache: dict[bytes, list[bytes]] = {}
        self.byte_tokens: list[bytes] = [bytes([i]) for i in range(256)]

    @classmethod
    def from_file(cls, 
                  vocab_filepath: str, 
                  merges_file_path: str, 
                  special_tokens: list[str] | None = None) -> Self:
        # -------- vocab.json --------
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError("vocab json must be an object mapping id->token")

        vocab: dict[int, bytes] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                raise ValueError("vocab json keys must be strings")
            if not isinstance(v, str):
                raise ValueError("vocab json values must be strings")
            tid = int(k)
            vocab[tid] = cls._token_str_to_bytes(v)

        # -------- merges.txt --------
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue

                # Preserve leading spaces in token1: split by the LAST space.
                j = line.rfind(" ")
                if j == -1:
                    raise ValueError(f"bad merges line {line_no}: {line!r}")

                a = line[:j]
                b = line[j + 1 :]
                if b == "":
                    raise ValueError(f"bad merges line {line_no} (empty second token): {line!r}")

                merges.append((cls._token_str_to_bytes(a), cls._token_str_to_bytes(b)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        pieces = split_by_special_tokens(text, self.special_pattern)

        ids: list[int] = []
        for p in pieces:
            sp_id = self.special_token_to_id.get(p)
            if sp_id is not None:
                ids.append(sp_id)
            else:
                for m in self.pretoken_pattern.finditer(p):
                    bs = m.group(0).encode("utf-8")
                    for tb in self._bpe(bs):
                        try:
                            ids.append(self.bytes_to_id[tb])
                        except KeyError:
                            raise ValueError(f"token bytes not in vocab: {tb!r}") from None
        return ids

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for tid in self.encode(text):
                yield tid
    
    def decode(self, ids: list[int]) -> str:
        bs = b"".join(self.id_to_bytes[i] for i in ids)
        return bs.decode("utf-8", errors="replace")
    
    @staticmethod
    def _token_str_to_bytes(tok: str) -> bytes:
        try:
            return tok.encode("latin-1")
        except UnicodeEncodeError:
            return tok.encode("utf-8")
        
    def _bpe(self, bs: bytes) -> list[bytes]:
        """
        apply merges on a single pre-token's bytes
        """
        cached = self.bpe_cache.get(bs)
        if cached is not None:
            return cached
        
        byte_list = [self.byte_tokens[b] for b in bs]       

        if len(byte_list) <= 1:
            self.bpe_cache[bs] = byte_list
            return byte_list
        
        while True:
            best_rank, best_pos = -1, -1

            for i in range(len(byte_list) - 1):
                pair = (byte_list[i], byte_list[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank == -1 or rank < best_rank:
                    best_rank = rank 
                    best_pos = i
            if best_rank == -1:
                break 
            new_byte = byte_list[best_pos] + byte_list[best_pos + 1]
            byte_list[best_pos: best_pos + 2] = [new_byte]

        self.bpe_cache[bs] = byte_list
        return byte_list
