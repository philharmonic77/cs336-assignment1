from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
import random
import time
import numpy as np

RANDOM_SEED = 20260101

def sample_docs(
    file_path: str,
    n: int = 10,
    boundary_token: str = "<|endoftext|>",
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    seed: int = RANDOM_SEED,
) -> list[str]:
    random.seed(seed)

    docs: list[str] = []
    buffer: list[str] = []
    read_bytes = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            read_bytes += len(line.encode("utf-8"))

            parts = line.split(boundary_token)
            for i, chunk in enumerate(parts):
                if i == 0:
                    buffer.append(chunk)
                else:
                    doc = "".join(buffer).strip()
                    if doc:
                        docs.append(doc)
                    buffer = [chunk]

            if read_bytes > max_bytes:
                break

    # flush last (possibly truncated) doc within the 100MB window
    tail = "".join(buffer).strip()
    if tail:
        docs.append(tail)

    if len(docs) < n:
        raise ValueError(
            f"Insufficient documents in the first 100MB: found {len(docs)}, required {n}."
        )

    return random.sample(docs, n)

def print_compression_ratio(
    texts: list[str],
    tokenizer,
    encoding: str = "utf-8",
    name: str = ""
) -> None:
    if not texts:
        raise ValueError("texts can't be empty!")

    total_bytes = 0
    total_tokens = 0

    for text in texts:
        if not text:
            continue

        total_bytes += len(text.encode(encoding))
        total_tokens += len(tokenizer.encode(text))

    if total_tokens == 0:
        raise ValueError("tokenizer returned 0 tokens!")
    
    ratio = total_bytes / total_tokens
    print(f"{name} total_bytes: {total_bytes}, total_tokens: {total_tokens} => compression_ratio: {ratio:.4f} bytes/token")

    return None

def measure_tokenizer_throughput(
    texts: list[str],
    tokenizer,
    encoding: str = "utf-8",
    repeats: int = 3,
) -> tuple[float, float]:
    """
    返回:
      - bytes_per_sec
      - tokens_per_sec
    """
    if not texts:
        raise ValueError("texts can't be empty!")

    total_bytes = sum(len(t.encode(encoding)) for t in texts if t)
    total_tokens = 0

    # warm-up
    for _ in tokenizer.encode_iterable(texts[:10]):
        pass

    start = time.perf_counter()
    for _ in range(repeats):
        for _ in tokenizer.encode_iterable(texts):
            total_tokens += 1
    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        raise ValueError("Timer resolution issue")

    bytes_per_sec = (total_bytes * repeats) / elapsed
    tokens_per_sec = total_tokens / elapsed

    return bytes_per_sec, tokens_per_sec


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    TS_train_data_path = str(repo_root / "data" / "TinyStoriesV2-GPT4-train.txt")
    TS_valid_data_path = str(repo_root / "data" / "TinyStoriesV2-GPT4-valid.txt")
    TS_vocab_path = str(repo_root / "artifacts" / "bpe" / "tinystories_vocab.json")
    TS_merges_path = str(repo_root / "artifacts" / "bpe" / "tinystories_merges.txt")

    OWT_train_data_path = str(repo_root / "data" / "owt_train.txt")
    OWT_valid_data_path = str(repo_root / "data" / "owt_valid.txt")
    OWT_vocab_path = str(repo_root / "artifacts" / "bpe" / "owt_vocab.json")
    OWT_merges_path = str(repo_root / "artifacts" / "bpe" / "owt_merges.txt")

    tinystories_samples = sample_docs(TS_train_data_path, n=10)
    owt_samples = sample_docs(OWT_train_data_path, n=10)

    tinystories_tokenizer = Tokenizer.from_file(TS_vocab_path,
                                                TS_merges_path,
                                                special_tokens=["<|endoftext|>"])
    owt_tokenizer = Tokenizer.from_file(OWT_vocab_path,
                                        OWT_merges_path,
                                        special_tokens=["<|endoftext|>"])
    
    owt_samples_10k = sample_docs(OWT_train_data_path, n=10_000, max_bytes=300 * 1024 * 1024)
    
    print("================ problem (a) ================")
    print_compression_ratio(tinystories_samples, tinystories_tokenizer, name="TS data + TS tok:")
    print_compression_ratio(owt_samples, owt_tokenizer, name="OWT data + OWT tok:")

    print("================ problem (b) ================")
    print_compression_ratio(owt_samples, tinystories_tokenizer, name="OWT data + TS tok:")
    print_compression_ratio(tinystories_samples, owt_tokenizer, name="TS data + OWT tok")

    print("================ problem (c) ================")
    bps, tps = measure_tokenizer_throughput(owt_samples_10k, owt_tokenizer, repeats=3)

    pile_bytes = 825 * 1_000_000_000
    seconds = pile_bytes / bps

    print(f"OWT tokenizer throughput: {bps/1e6:.2f} MB/s")
    print(f"Estimated time for 825GB: {seconds/3600:.2f} hours ({seconds/3600/24:.2f} days)")
    print(f"(optional) tokens/sec: {tps/1e6:.2f} M tokens/s")

    print("================ problem (d) ================")

    
if __name__ == "__main__":
    main()