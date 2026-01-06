from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
import random

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
    buffer = []
    read_bytes = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            read_bytes += len(line.encode("utf-8"))
            if read_bytes > max_bytes:
                break

            if boundary_token in line:
                parts = line.split(boundary_token)

                # boundary 前的内容属于当前 document
                buffer.append(parts[0])
                doc = "".join(buffer).strip()
                if doc:
                    docs.append(doc)

                # boundary 后的内容属于下一个 document
                buffer = [parts[1]] if len(parts) > 1 else []
            else:
                buffer.append(line)

    if len(docs) < n:
        raise ValueError(f"Insufficient documents in the first 100MB: found {len(docs)}, required {n}.")

    return random.sample(docs, n)

def get_compression_ratio(text: str, tokenizer, encoding: str = "utf-8") -> float:
    if not text:
        raise ValueError("text can't be empty!")

    num_bytes = len(text.encode(encoding))

    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)

    if num_tokens == 0:
        raise ValueError("tokenizer returns 0 token!")

    return num_bytes / num_tokens

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    tinystories_samples = sample_docs(str(repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"))
    # owt_samples = sample_docs(str(repo_root / "data" / "owt_train.txt"))

    tinystories_tokenizer = Tokenizer.from_file(vocab_filepath=str(repo_root / "artifacts" / "bpe" / "tinystories_vocab.json"),
                                                merges_file_path=str(repo_root / "artifacts" / "bpe" / "tinystories_merges.txt"),
                                                special_tokens=["<|endoftext|>"])
    # owt_tokenizer = Tokenizer.from_file(vocab_filepath=str(repo_root / "artifacts" / "bpe" / "owt_vocab.json"),
    #                                     merges_file_path=str(repo_root / "artifacts" / "bpe" / "owt_merges.txt"),
    #                                     special_tokens=["<|endoftext|>"])
    
    print(len(tinystories_samples), tinystories_samples[9])
    print(tinystories_tokenizer)
    # print(owt_tokenizer)
    
if __name__ == "__main__":
    main()