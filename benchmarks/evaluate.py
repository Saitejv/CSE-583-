"""Held-out end-to-end evaluation entry point (Phase 4)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="evaluation config YAML/JSON")
    parser.parse_args(argv)
    raise SystemExit("evaluate: implemented in Phase 4 (T4.2-T4.3)")


if __name__ == "__main__":
    raise SystemExit(main())
