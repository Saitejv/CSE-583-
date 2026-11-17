from __future__ import annotations

import jax

from . import __version__


def main() -> None:
    print(f"jax-unroll {__version__}")
    print(f"jax {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")


if __name__ == "__main__":
    main()
