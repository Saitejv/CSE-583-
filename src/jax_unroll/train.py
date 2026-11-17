"""Training entry point for the Phase 3 cost model."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="raw or joined sweep CSV")
    parser.add_argument("--out", required=True, help="output directory for artifacts")
    parser.add_argument("--cv", default="leave-one-family-out")
    parser.parse_args(argv)
    raise SystemExit("train: implemented in Phase 3 (T3.1-T3.2)")


if __name__ == "__main__":
    raise SystemExit(main())
