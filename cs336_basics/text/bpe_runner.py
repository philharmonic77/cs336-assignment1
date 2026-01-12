from __future__ import annotations

import json
from pathlib import Path

from cs336_basics.text.train_bpe import ParallelConfig, train_byte_level_bpe_incremental
from cs336_basics.text.codec import bytes_to_gpt2_str


def _bytes_to_str(b: bytes) -> str:
    return bytes_to_gpt2_str(b)


def _expected_id_to_bytes(
    vocab: dict[int, bytes], special_tokens: list[str] | None
) -> list[bytes]:
    max_id = max(vocab.keys()) if vocab else -1
    id_to_bytes: list[bytes] = [b""] * (max_id + 1)
    bytes_to_id: dict[bytes, int] = {}

    for i, b in vocab.items():
        id_to_bytes[i] = b
        bytes_to_id[b] = i

    if any(x == b"" for x in id_to_bytes):
        raise ValueError("vocab ids must be contiguous from 0..max_id")

    if special_tokens:
        for st in special_tokens:
            sb = st.encode("utf-8")
            if sb not in bytes_to_id:
                new_id = len(id_to_bytes)
                id_to_bytes.append(sb)
                bytes_to_id[sb] = new_id

    return id_to_bytes


def _merges_from_tokenizer(tokenizer) -> list[tuple[bytes, bytes]]:
    return [
        pair
        for pair, _ in sorted(tokenizer.merge_ranks.items(), key=lambda kv: kv[1])
    ]


def verify_tokenizer_roundtrip(
    *,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | Path,
    merges_path: str | Path,
    special_tokens: list[str] | None = None,
) -> None:
    """
    Verify that a tokenizer reconstructed from files matches the in-memory vocab/merges.
    Raises ValueError on the first mismatch.
    """
    from cs336_basics.text.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_file(
        vocab_filepath=str(vocab_path),
        merges_file_path=str(merges_path),
        special_tokens=special_tokens,
    )

    expected_id_to_bytes = _expected_id_to_bytes(vocab, special_tokens)
    if tokenizer.id_to_bytes != expected_id_to_bytes:
        for i, (got, exp) in enumerate(
            zip(tokenizer.id_to_bytes, expected_id_to_bytes), start=0
        ):
            if got != exp:
                raise ValueError(
                    f"vocab mismatch at id {i}: file={got!r} expected={exp!r}"
                )
        raise ValueError(
            f"vocab size mismatch: file={len(tokenizer.id_to_bytes)} "
            f"expected={len(expected_id_to_bytes)}"
        )

    loaded_merges = _merges_from_tokenizer(tokenizer)
    if loaded_merges != merges:
        for i, (got, exp) in enumerate(zip(loaded_merges, merges), start=0):
            if got != exp:
                raise ValueError(
                    f"merge mismatch at index {i}: file={got!r} expected={exp!r}"
                )
        raise ValueError(
            f"merge count mismatch: file={len(loaded_merges)} expected={len(merges)}"
        )

    print(f"verify ok: {Path(vocab_path).name}, {Path(merges_path).name}")


def train_bpe_and_save(
    *,
    input_path: str | Path,
    vocab_size: int,
    output_dir: str | Path,
    vocab_filename: str,
    merges_filename: str,
    special_tokens: list[str],
    parallel: ParallelConfig | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab, merges = train_byte_level_bpe_incremental(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        parallel=parallel,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab_out = {str(idx): _bytes_to_str(tok) for idx, tok in vocab.items()}
    with open(output_path / vocab_filename, "w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=True, separators=(",", ":"))

    longest_id, longest_tok = max(vocab.items(), key=lambda kv: len(kv[1]))
    print(
        f"Longest token: id={longest_id} len={len(longest_tok)} "
        f"token={_bytes_to_str(longest_tok)!r}"
    )

    with open(output_path / merges_filename, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_bytes_to_str(a)} {_bytes_to_str(b)}\n")

    verify_tokenizer_roundtrip(
        vocab=vocab,
        merges=merges,
        vocab_path=output_path / vocab_filename,
        merges_path=output_path / merges_filename,
        special_tokens=special_tokens,
    )

    return vocab, merges
