from __future__ import annotations

import time
import regex as re
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.chunking import _find_chunk_boundaries
from cs336_basics.pretokenize import _chunk_to_word_freq, _compile_special_pattern


@dataclass(frozen=True)
class ParallelConfig:
    desired_num_chunks: int
    num_workers: int | None = None
    boundary_token: str = "<|endoftext|>"


def train_byte_level_bpe(
    input_path: str,
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
        # TODO: optimize by maintaining pair_freq incrementally instead of full recompute.
        pair_freq = _count_pairs(words)
        if not pair_freq:
            break
        best_pair = _select_pair(pair_freq, vocab)

        a, b = vocab[best_pair[0]], vocab[best_pair[1]]

        merges.append((a, b))
        new_id = next_id
        next_id += 1
        vocab[new_id] = a + b

        new_words = _apply_merge(words, best_pair, new_id)
        words = new_words
        
        if len(vocab) % 500 == 0:
            now = time.perf_counter()
            print(
                f"[bpe] vocab={len(vocab)} merges={len(merges)} "
                f"elapsed={now - start_time:.1f}s "
                f"delta={now - last_report:.1f}s"
            )
            last_report = now

    return vocab, merges


def _build_chunk_for_worker(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
    special_pattern: re.Pattern,
    pretoken_pattern: re.Pattern,
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
    input_path: str,
    *,
    special_tokens: list[str],
    pretoken_pattern: re.Pattern[str],
    parallel: ParallelConfig | None = None,
) -> Counter[str]:
    special_pattern = _compile_special_pattern(special_tokens)

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
