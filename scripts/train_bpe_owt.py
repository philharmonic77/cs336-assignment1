import time
import cProfile
import pstats
import resource
from pathlib import Path
import json
import sys

from cs336_basics.train_bpe import train_byte_level_bpe_incremental, ParallelConfig
from cs336_basics.gpt2_bytes import bytes_to_gpt2_str
from scripts.bpe_verify import verify_tokenizer_roundtrip

def _bytes_to_str(b: bytes) -> str:
    return bytes_to_gpt2_str(b)

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
        input_path=str(repo_root / "data" / "owt_train.txt"),
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        parallel=cfg
    )
    end = time.perf_counter()
    print(f"\nElapsed time: {end - start:.6f} seconds")

    output_dir = repo_root / "artifacts" / "bpe"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_out = {str(idx): _bytes_to_str(tok) for idx, tok in vocab.items()}
    with open(output_dir / "owt_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=True, separators=(",", ":"))

    longest_id, longest_tok = max(vocab.items(), key=lambda kv: len(kv[1]))
    print(
        f"Longest token: id={longest_id} len={len(longest_tok)} "
        f"token={_bytes_to_str(longest_tok)!r}"
    )

    with open(output_dir / "owt_merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_bytes_to_str(a)} {_bytes_to_str(b)}\n")

    verify_tokenizer_roundtrip(
        vocab=vocab,
        merges=merges,
        vocab_path=output_dir / "owt_vocab.json",
        merges_path=output_dir / "owt_merges.txt",
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
