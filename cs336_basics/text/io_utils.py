from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cs336_basics.text.tokenizer import Tokenizer


def encode_file_to_npy(
    input_path: str,
    output_path: str,
    tokenizer: Tokenizer,
    encoding: str = "utf-8",
    chunk_size: int = 1_000_000,
    log_every_tokens: int = 50_000_000,
) -> None:
    max_id = len(tokenizer.id_to_bytes) - 1
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(f"tokenizer vocab too large for uint16: max_id={max_id}")

    print(f"[encode] counting tokens in {input_path}")
    total_tokens = 0
    with open(input_path, "r", encoding=encoding) as f:
        for tid in tokenizer.encode_iterable(f):
            total_tokens += 1

    print(f"[encode] total tokens: {total_tokens}, writing to {output_path}")
    arr = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.uint16, shape=(total_tokens,)
    )

    offset = 0
    buffer: list[int] = []
    next_log = log_every_tokens
    with open(input_path, "r", encoding=encoding) as f:
        for tid in tokenizer.encode_iterable(f):
            buffer.append(tid)
            if len(buffer) >= chunk_size:
                end = offset + len(buffer)
                arr[offset:end] = np.asarray(buffer, dtype=np.uint16)
                offset = end
                if log_every_tokens > 0 and offset >= next_log:
                    print(f"[encode] wrote {offset} tokens...")
                    next_log += log_every_tokens
                buffer.clear()

    if buffer:
        end = offset + len(buffer)
        arr[offset:end] = np.asarray(buffer, dtype=np.uint16)

    print(f"[encode] done: {output_path}")

