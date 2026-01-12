import regex as re
import time
import statistics

from cs336_basics.text.train_bpe import _build_word_freq, ParallelConfig

DATA_PATH = "data/TinyStoriesV2-GPT4-valid.txt"

def timed_run(fn, *, repeat: int = 3, warmup: int = 1):
    # warmup（让文件缓存/regex/JIT-like effects更稳定）
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times, out

def main() -> None:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_pattern = re.compile(PAT)   
    special_tokens = ["<|endoftext|>"]

    serial = _build_word_freq(
        DATA_PATH,
        special_tokens=special_tokens,
        pretoken_pattern=pretoken_pattern,
        parallel=None,
    )

    parallel = _build_word_freq(
        DATA_PATH,
        special_tokens=special_tokens,
        pretoken_pattern=pretoken_pattern,
        parallel=ParallelConfig(
            desired_num_chunks=4,
            num_workers=4,
            boundary_token="<|endoftext|>",
        ),
    )

    print("serial unique:", len(serial))
    print("parallel unique:", len(parallel))
    print("serial total:", sum(serial.values()))
    print("parallel total:", sum(parallel.values()))

    if serial != parallel:
        diff = serial - parallel
        diff2 = parallel - serial
        print("Mismatch!")
        print("serial-parallel (top 20):", diff.most_common(20))
        print("parallel-serial (top 20):", diff2.most_common(20))
        raise SystemExit(1)

    print("OK: serial == parallel")
    print("Top 20:", serial.most_common(20))


    print("----------- 以下测试提速 -------------")
    serial_fn = lambda: _build_word_freq(
        DATA_PATH,
        special_tokens=special_tokens,
        pretoken_pattern=pretoken_pattern,
        parallel=None,
    )

    parallel_cfg = ParallelConfig(
        desired_num_chunks=8,     # 可改：16/32/64
        num_workers=8,            # 可改：4/8/None
        boundary_token="<|endoftext|>",
    )

    parallel_fn = lambda: _build_word_freq(
        DATA_PATH,
        special_tokens=special_tokens,
        pretoken_pattern=pretoken_pattern,
        parallel=parallel_cfg,
    )

    serial_times, serial_out = timed_run(serial_fn, repeat=3, warmup=1)
    parallel_times, parallel_out = timed_run(parallel_fn, repeat=3, warmup=1)

    # 正确性护栏（避免“测到的是错的快”）
    assert serial_out == parallel_out

    def summarize(name, times):
        print(f"{name}:")
        print(f"  runs: {times}")
        print(f"  mean: {statistics.mean(times):.4f}s")
        print(f"  min : {min(times):.4f}s")

    summarize("serial", serial_times)
    summarize("parallel", parallel_times)

    speedup_mean = statistics.mean(serial_times) / statistics.mean(parallel_times)
    speedup_min = min(serial_times) / min(parallel_times)
    print(f"speedup (mean): {speedup_mean:.2f}x")
    print(f"speedup (min) : {speedup_min:.2f}x")

if __name__ == "__main__":
    main()
