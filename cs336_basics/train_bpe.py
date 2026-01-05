from __future__ import annotations

import time
import os
import regex as re
import heapq
from typing import Optional
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.chunking import _find_chunk_boundaries
from cs336_basics.pretokenize import _chunk_to_word_freq, compile_special_pattern


@dataclass(frozen=True)
class ParallelConfig:
    desired_num_chunks: int
    num_workers: int | None = None
    boundary_token: str = "<|endoftext|>"

HeapItem = tuple[int, int, int]  # (-freq, a, b)


def train_byte_level_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    *,
    encoding: str = "utf-8",
    parallel: ParallelConfig | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer from a text corpus stored in a file.

    Parameters
    ----------
    input_path : str
        Path to a text file containing BPE tokenizer training data.
    vocab_size : int
        Positive integer defining the maximum final vocabulary size, including:
          - the initial byte vocabulary (0..255),
          - vocabulary items produced from merges,
          - and any provided special tokens.
    special_tokens : list[str]
        List of strings to add to the vocabulary as special tokens.
        Special tokens do not otherwise affect BPE training (i.e., they should not
        influence merge statistics).

    Returns
    -------
    vocab : dict[int, bytes]
        Tokenizer vocabulary mapping token ID (int) -> token bytes (bytes).
    merges : list[tuple[bytes, bytes]]
        Ordered list of BPE merges produced during training. Each merge is a tuple:
            (<token1>, <token2>)
        indicating <token1> was merged with <token2>. The list must be ordered
        by creation time (earliest first).
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_pattern = re.compile(PAT)

    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens contains duplicates")

    word_freq = _build_word_freq(
        input_path,
        special_tokens=special_tokens,
        pretoken_pattern=pretoken_pattern,
        parallel=parallel
    )

    vocab, words = _init_symbol_vocab_and_words(
        word_freq,
        special_tokens=special_tokens,
        encoding=encoding
    )

    merges: list[tuple[bytes, bytes]] = []
    next_id = max(vocab.keys()) + 1

    last_report = time.perf_counter()
    start_time = time.perf_counter()
    last_report = start_time

    while len(vocab) < vocab_size:       
        pair_freq = _count_pairs(words)
        if not pair_freq:
            break
        best_pair = _select_pair(pair_freq, vocab)

        left_bytes, right_bytes = vocab[best_pair[0]], vocab[best_pair[1]]

        merges.append((left_bytes, right_bytes))
        new_id = next_id
        next_id += 1
        vocab[new_id] = left_bytes + right_bytes

        new_words = _apply_merge(words, best_pair, new_id)
        words = new_words
        
        if len(vocab) % 100 == 0:
            now = time.perf_counter()
            print(
                f"[bpe] vocab={len(vocab)} merges={len(merges)} "
                f"elapsed={now - start_time:.1f}s "
                f"delta={now - last_report:.1f}s"
            )
            last_report = now

    return vocab, merges

def train_byte_level_bpe_incremental(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    *,
    encoding: str = "utf-8",
    parallel: ParallelConfig | None = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_pattern = re.compile(PAT)

    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens contains duplicates")

    word_freq = _build_word_freq(
        input_path,
        special_tokens=special_tokens,
        pretoken_pattern=pretoken_pattern,
        parallel=parallel
    )

    vocab, words = _init_symbol_vocab_and_words(
        word_freq,
        special_tokens=special_tokens,
        encoding=encoding
    )

    # init word_seq(wid->seq), word_freq(wid->freq)  
    wid_seq : dict[int, tuple[int, ...]] = {}
    wid_freq : dict[int, int] = {}
    for i, (word, freq) in enumerate(words.items()):
        wid_seq[i] = word
        wid_freq[i] = freq

    # init pair_freq and pair_to_word
    pair_freq, pair_to_word = _init_pair_freq_and_pair_to_word(wid_seq, wid_freq) 

    # init pair heap
    heap: list = []
    for p in pair_freq:
        _heap_push_max_lazy(heap, p, pair_freq)

    merges: list[tuple[bytes, bytes]] = []
    next_id = max(vocab.keys()) + 1

    last_report = time.perf_counter()
    start_time = time.perf_counter()
    last_report = start_time

    while len(vocab) < vocab_size:
        # 完全没有pair可以merge，break
        if not pair_freq: 
            break
        
        best_pair = _heap_pop_best_pair_max_lazy(heap, pair_freq, vocab)
        if best_pair is None:
            break
        left_bytes, right_bytes = vocab[best_pair[0]], vocab[best_pair[1]]

        merges.append((left_bytes, right_bytes))
        new_id = next_id
        next_id += 1
        vocab[new_id] = left_bytes + right_bytes
   
        affected_words = pair_to_word.get(best_pair, set()).copy()

        for wid in affected_words:
            new_seq = _update_one_word(wid,
                                       new_id,
                                       best_pair,
                                       wid_freq[wid], 
                                       wid_seq[wid], 
                                       pair_freq, 
                                       pair_to_word,
                                       heap) 
            wid_seq[wid] = new_seq
        
        # 轮次结束后清理
        if not pair_to_word.get(best_pair):
            pair_to_word.pop(best_pair, None)

        if pair_freq.get(best_pair, 0) == 0:
            pair_freq.pop(best_pair, None)

        if len(vocab) % 100 == 0:
            now = time.perf_counter()
            print(
                f"[bpe] vocab={len(vocab)} merges={len(merges)} "
                f"elapsed={now - start_time:.1f}s "
                f"delta={now - last_report:.1f}s "
                f"pair_freq={len(pair_freq)} "
                f"pair_to_word={len(pair_to_word)} "
                f"affected_words={len(affected_words)}"
            )
            last_report = now

    return vocab, merges


def _update_one_word(wid: int,
                     new_id: int,
                     best_pair: tuple[int, int],
                     freq: int, 
                     seq: tuple[int, ...], 
                     pair_freq: dict[tuple[int, int], int], 
                     pair_to_word: dict[tuple[int, int], set[int]],
                     heap: list[HeapItem]) -> tuple[int, ...]:

    # 撤销old seq的影响
    L = len(seq)
    seen_pair = set()
    for i in range(L - 1):
        p = (seq[i], seq[i + 1])

        pair_freq[p] = pair_freq.get(p, 0) - freq 
        if pair_freq[p] == 0:
            pair_freq.pop(p, None)

        _heap_push_max_lazy(heap, p, pair_freq)

        if p not in seen_pair:
            # 1.如果 p 这个 key 存在，就把 wid 从集合里删掉（discard）
	        # 2.删完后如果集合变空，就把 key 从 dict 里删掉（pop）
            s = pair_to_word.get(p)
            if s is not None:
                pair_to_word[p].discard(wid)
                if not s:
                    pair_to_word.pop(p, None)
            seen_pair.add(p)

    # merge new seq
    out = []
    i = 0
    while i < L:
        if i < L - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    new_seq = tuple(out) 

    # 统计new seq的影响
    new_L = len(new_seq)
    new_seen_pair = set()

    for i in range(new_L - 1):  
        p = (new_seq[i], new_seq[i + 1])
        pair_freq[p] = pair_freq.get(p, 0) + freq
        
        _heap_push_max_lazy(heap, p, pair_freq)

        if p not in new_seen_pair:
            pair_to_word.setdefault(p, set()).add(wid)
            new_seen_pair.add(p)

    return new_seq

def _init_pair_freq_and_pair_to_word(word_seq: dict[int, tuple[int, ...]], 
                                     word_freq: dict[int, int]
                                     ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:

    pair_freq: dict[tuple[int, int], int] = {}
    pair_to_word: dict[tuple[int, int], set[int]] = {}

    for wid, seq in word_seq.items():
        if len(seq) < 2:
            continue
        
        seen_pair: set[tuple[int, int]] = set()

        prev = seq[0]
        for x in seq[1:]:
            p = (prev, x)
            pair_freq[p] = pair_freq.get(p, 0) + word_freq[wid]
            seen_pair.add(p)
            prev = x

        for p in seen_pair:
            if p in pair_to_word:
                pair_to_word[p].add(wid)
            else:
                pair_to_word[p] = {wid}
    return pair_freq, pair_to_word

def _build_chunk_for_worker(
    input_path: str | os.PathLike[str],
    start: int,
    end: int,
    special_tokens: list[str],
    special_pattern,
    pretoken_pattern,
) -> Counter[str]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    return _chunk_to_word_freq(
        chunk,
        special_tokens=special_tokens,
        special_pattern=special_pattern,
        pretoken_pattern=pretoken_pattern,
    )


def _build_word_freq(
    input_path: str | os.PathLike[str],
    *,
    special_tokens:  list[str],
    pretoken_pattern,
    parallel: ParallelConfig | None = None,
) -> Counter[str]:
    special_pattern = compile_special_pattern(special_tokens)

    if parallel is None:
        with open(input_path, "r") as f:
            data = f.read()

        return _chunk_to_word_freq(
            data,
            special_tokens=special_tokens,
            special_pattern=special_pattern,
            pretoken_pattern=pretoken_pattern,
        )

    if parallel.desired_num_chunks <= 0:
        raise ValueError("parallel.desired_num_chunks must be positive")
    if parallel.boundary_token not in special_tokens:
        raise ValueError("parallel.boundary_token must be included in special_tokens")

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(
            f,
            parallel.desired_num_chunks,
            parallel.boundary_token.encode("utf-8"),
        )

    total = Counter()
    spans = zip(boundaries[:-1], boundaries[1:])

    with ProcessPoolExecutor(max_workers=parallel.num_workers) as ex:
        futures = [
            ex.submit(
                _build_chunk_for_worker,
                input_path,
                start,
                end,
                special_tokens,
                special_pattern,
                pretoken_pattern,
            )
            for start, end in spans
        ]
        for f in futures:
            total.update(f.result())
    return total


def _init_symbol_vocab_and_words(
    word_freq: Counter[str],
    special_tokens: list[str],
    *,
    encoding: str = "utf-8",
) -> tuple[dict[int, bytes], dict[tuple[int, ...], int]]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode(encoding)
        next_id += 1

    words = {}
    for word, freq in word_freq.items():
        if word in special_tokens:
            continue

        seq = tuple(word.encode(encoding))
        words[seq] = words.get(seq, 0) + freq

    return vocab, words


def _count_pairs(words: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pair_freq: dict[tuple[int, int], int] = {} # 不选Counter：Counter的内部方法/路径比手写 dict.get 更重
    for seq, freq in words.items():
        if len(seq) < 2:
            continue
        prev = seq[0]
        for x in seq[1:]:
            p = (prev, x)
            pair_freq[p] = pair_freq.get(p, 0) + freq
            prev = x
    return pair_freq


def _select_pair(pair_freq: dict[tuple[int, int], int],
                 vocab: dict[int, bytes]) -> tuple[int, int]:
    if not pair_freq:
        raise ValueError("pair_freq is empty!")

    return max(
        pair_freq.items(),
        key=lambda kv: (kv[1], vocab[kv[0][0]], vocab[kv[0][1]]),
    )[0]


def _apply_merge(
    words: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
    *,
    verify: bool = False,
) -> dict[tuple[int, ...], int]:
    a, b = pair
    total_sum_before = sum(words.values()) if verify else 0

    new_words = {}
    for seq, freq in words.items():
        length = len(seq)
        if length < 2:
            new_words[seq] = new_words.get(seq, 0) + freq
            continue

        # 先做存在性检查（一次线性扫，几乎零分配）
        prev = seq[0]
        has = False
        for x in seq[1:]:
            if prev == a and x == b:
                has = True
                break
            prev = x

        if not has:
            new_words[seq] = new_words.get(seq, 0) + freq
            continue

        out = []
        i = 0
        while i < length:
            if i < length - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        new_seq = tuple(out)
        new_words[new_seq] = new_words.get(new_seq, 0) + freq

    if verify:
        assert total_sum_before == sum(new_words.values()), (
            f"old freq: {total_sum_before}, new freq: {sum(new_words.values())}"
        )

        for seq in new_words.keys():
            for i in range(len(seq) - 1):
                assert not (seq[i] == a and seq[i + 1] == b)

    return new_words

def _heap_push_max_lazy(
    heap: list[HeapItem],
    pair: tuple[int, int],
    pair_freq: dict[tuple[int, int], int],
) -> None:
    f = pair_freq.get(pair, 0)
    if f > 0:
        a, b = pair
        heapq.heappush(heap, (-f, a, b))


def _heap_pop_best_pair_max_lazy(
    heap: list[HeapItem],
    pair_freq: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> Optional[tuple[int, int]]:
    """
    Pop the best pair under:
      - max frequency
      - tie-break: lexicographically greater (vocab[a], vocab[b])
    Uses lazy deletion: stale heap entries are discarded.
    """
    while heap:
        neg_f, a, b = heapq.heappop(heap)
        f = -neg_f

        # lazy 验证：如果这个条目不是当前真值，丢弃
        if pair_freq.get((a, b), 0) != f or f <= 0:
            continue

        # 收集所有 “同频 f” 的有效候选，做 tie-break（取 bytes 字典序最大的）
        best = (a, b)
        best_key = (vocab[a], vocab[b])

        same_freq_valid: list[HeapItem] = [(neg_f, a, b)]

        # 候选lazy验证+找best
        while heap and -heap[0][0] == f:
            neg2, a2, b2 = heapq.heappop(heap)
            if pair_freq.get((a2, b2), 0) != f:
                continue  # stale
            same_freq_valid.append((neg2, a2, b2))
            k2 = (vocab[a2], vocab[b2])
            if k2 > best_key:
                best = (a2, b2)
                best_key = k2

        # 把同频里没选中的有效候选放回去（保持后续可用）
        for item in same_freq_valid:
            _, a2, b2 = item
            if (a2, b2) != best:
                heapq.heappush(heap, item)

        return best

    return None


if __name__ == "__main__":
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_pattern = re.compile(PAT)

    cfg = ParallelConfig(
        desired_num_chunks=8,
        num_workers=8, 
        boundary_token="<|endoftext|>",
    )

    word_freq = _build_word_freq(
        "data/TinyStoriesV2-GPT4-valid.txt",
        special_tokens=["<|endoftext|>"],
        pretoken_pattern=pretoken_pattern,
        parallel=cfg
    )

    vocab, words = _init_symbol_vocab_and_words(
        word_freq,
        special_tokens=["<|endoftext|>"],
    )

    pair_freq = _count_pairs(words)
    best_pair = _select_pair(pair_freq, vocab)

    # print(vocab[best_pair[0]], vocab[best_pair[1]])

    new_words = _apply_merge(words, best_pair, max(vocab.keys()) + 1, verify=True)
