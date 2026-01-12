from __future__ import annotations

from functools import lru_cache


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Map every byte 0..255 to a printable unicode character (GPT-2 scheme).
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


@lru_cache
def gpt2_unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


def bytes_to_gpt2_str(b: bytes) -> str:
    mapping = gpt2_bytes_to_unicode()
    return "".join(mapping[x] for x in b)


def gpt2_str_to_bytes(s: str) -> bytes:
    mapping = gpt2_unicode_to_bytes()
    out = bytearray()
    for ch in s:
        try:
            out.append(mapping[ch])
        except KeyError as exc:
            raise ValueError(f"invalid gpt2 byte char: {ch!r}") from exc
    return bytes(out)
