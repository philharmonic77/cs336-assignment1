import json
from collections import Counter

from cs336_basics.gpt2_bytes import bytes_to_gpt2_str, gpt2_str_to_bytes

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {int(k): gpt2_str_to_bytes(v) for k, v in vocab.items()}

ts_vocab = load_vocab("artifacts/bpe/tinystories_vocab.json")
owt_vocab = load_vocab("artifacts/bpe/owt_vocab.json")

def summarize(name, vocab):
    lengths = Counter(len(b) for b in vocab.values())
    ascii_only = sum(all(x < 128 for x in b) for b in vocab.values())

    print(f"\n== {name} ==")
    print("vocab size:", len(vocab))
    print("max token length:", max(lengths))
    print("ASCII-only ratio:", ascii_only / len(vocab))

    longest = sorted(vocab.values(), key=len, reverse=True)[:5]
    print("longest tokens:")
    for b in longest:
        print(" ", bytes_to_gpt2_str(b))

summarize("TinyStories", ts_vocab)
summarize("OpenWebText", owt_vocab)

# overlap
ts_set = set(ts_vocab.values())
owt_set = set(owt_vocab.values())

print("\n== Overlap ==")
print("shared tokens:", len(ts_set & owt_set))
print("only TinyStories:", len(ts_set - owt_set))
print("only OpenWebText:", len(owt_set - ts_set))
