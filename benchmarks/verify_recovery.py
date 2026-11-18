"""Reproduce confirmation joins and verify the recorded gate without rescoring."""

from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.recovery_confirm import JOINED, RAW, verify
from benchmarks.recovery_data import prepare, sha


def main():
    verify()
    with TemporaryDirectory(prefix="jax-unroll-confirmation-") as directory:
        reproduction = Path(directory) / JOINED.name
        prepare(RAW, "confirmation", reproduction)
        for path in (JOINED, JOINED.with_suffix(".meta.json")):
            if sha(path) != sha(reproduction.with_name(path.name)):
                raise ValueError(f"confirmation join did not reproduce: {path}")
    print("Reproduced the confirmation dataset and metadata byte for byte, without model scoring")


if __name__ == "__main__":
    main()
