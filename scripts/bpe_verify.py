from __future__ import annotations

from pathlib import Path
from typing import Iterable

from cs336_basics.text.tokenizer import Tokenizer


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


def _merges_from_tokenizer(tokenizer: Tokenizer) -> list[tuple[bytes, bytes]]:
    return [pair for pair, _ in sorted(tokenizer.merge_ranks.items(), key=lambda kv: kv[1])]


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
