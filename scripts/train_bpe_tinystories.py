import time
import cProfile
import pstats
import resource
from pathlib import Path
import json
import sys

from cs336_basics.train_bpe import train_byte_level_bpe_incremental, ParallelConfig
from scripts.bpe_verify import verify_tokenizer_roundtrip

def _bytes_to_str(b: bytes) -> str:
    """
    •	Latin-1 和 UTF-8 是不同体系
	•	ASCII 区间它们碰巧一致
	•	UTF-8 会“扩展字节”，Latin-1 永不扩展
	•	tokenizer 的 vocab/merges 必须用 Latin-1 才能保持 byte-level 语义
    """
    return b.decode("latin-1")

def _bytes_to_escaped_str(b: bytes) -> str:
    return b.decode("latin-1").encode("unicode_escape").decode("ascii")

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = ParallelConfig(
        desired_num_chunks=8,
        num_workers=8,
        boundary_token="<|endoftext|>",
    )

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    vocab, merges = train_byte_level_bpe_incremental(
        input_path=str(repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"),
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        parallel=cfg
    )
    end = time.perf_counter()
    print(f"\nElapsed time: {end - start:.6f} seconds")

    output_dir = repo_root / "artifacts" / "bpe"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_out = {str(idx): _bytes_to_str(tok) for idx, tok in vocab.items()}
    with open(output_dir / "tinystories_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=True, separators=(",", ":"))

    longest_id, longest_tok = max(vocab.items(), key=lambda kv: len(kv[1]))
    print(
        f"Longest token: id={longest_id} len={len(longest_tok)} "
        f"token={_bytes_to_str(longest_tok)!r}"
    )

    with open(output_dir / "tinystories_merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_bytes_to_escaped_str(a)} {_bytes_to_escaped_str(b)}\n")

    verify_tokenizer_roundtrip(
        vocab=vocab,
        merges=merges,
        vocab_path=output_dir / "tinystories_vocab.json",
        merges_path=output_dir / "tinystories_merges.txt",
        special_tokens=["<|endoftext|>"],
    )

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Platform: {sys.platform}, ru_maxrss: {peak}")
    if sys.platform == "darwin":
        peak_gb = peak / (1024 ** 3)
    else:
        peak_gb = (peak * 1024) / (1024 ** 3)
    print(f"Peak RSS: {peak_gb:.3f} GB")

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
