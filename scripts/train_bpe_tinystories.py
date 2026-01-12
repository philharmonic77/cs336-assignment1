import time
import cProfile
import pstats
import resource
from pathlib import Path
import sys

from cs336_basics.text.bpe_runner import train_bpe_and_save
from cs336_basics.text.train_bpe import ParallelConfig

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
    train_bpe_and_save(
        input_path=repo_root / "data" / "TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        output_dir=repo_root / "artifacts" / "bpe",
        vocab_filename="tinystories_vocab.json",
        merges_filename="tinystories_merges.txt",
        special_tokens=["<|endoftext|>"],
        parallel=cfg,
    )
    end = time.perf_counter()
    print(f"\nElapsed time: {end - start:.6f} seconds")

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
