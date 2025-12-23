import cProfile
import pstats

from cs336_basics.train_bpe import train_byte_level_bpe
from tests.common import FIXTURES_PATH


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    train_byte_level_bpe(
        input_path=FIXTURES_PATH / "corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        parallel=None,
    )
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
