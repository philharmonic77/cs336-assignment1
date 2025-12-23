from collections import Counter
import re


def _compile_special_pattern(special_tokens: list[str]) -> re.Pattern[str]:
    if not special_tokens:
        return re.compile('(?!x)x') # a regexp that never match
    escaped_tokens = [re.escape(tok) for tok in special_tokens]
    pattern_str = "(" + "|".join(escaped_tokens) + ")"
    return re.compile(pattern_str)

def _split_by_special_tokens(text:str, pat: re.Pattern[str]) -> list[str]:
    return [p for p in pat.split(text) if p != '']

def _chunk_to_word_freq(
        chunk: str, 
        *,
        special_tokens: list[str],
        special_pattern: re.Pattern[str],
        pretoken_pattern: re.Pattern[str]
        ) -> Counter[str]:
    counter = Counter()
    pieces = _split_by_special_tokens(chunk, special_pattern)

    for p in pieces:
        if p in special_tokens:
            continue
        else:
            for m in pretoken_pattern.finditer(p):
                counter[m.group(0)] += 1
    return counter






if __name__ == '__main__':

    special_tokens = {"<|endoftext|>"}
    special_pattern = _compile_special_pattern(list(special_tokens))
    pretoken_pattern = re.compile(r"[A-Za-z]+")

    text = "<|endoftext|>!!!<|endoftext|>"

    freq = _chunk_to_word_freq(
        text,
        special_pattern=special_pattern,
        special_tokens=special_tokens,
        pretoken_pattern=pretoken_pattern,
    )

    print(freq)