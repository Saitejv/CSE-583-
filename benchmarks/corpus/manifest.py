"""Write the machine-readable corpus manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmarks.corpus.registry import (
    body_hashes,
    manifest,
    validate_environment,
    validate_structure,
)

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.json"


def write_manifest(path: Path, *, include_hashes: bool = True) -> dict[str, object]:
    validate_environment()
    validate_structure()
    hashes = body_hashes() if include_hashes else None
    data = manifest(hashes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-hashes", action="store_true")
    args = parser.parse_args(argv)
    data = write_manifest(args.out, include_hashes=not args.no_hashes)
    print(
        f"wrote {args.out}: {data['body_count']} bodies, "
        f"{data['family_count']} families, "
        f"smoke={data['tiers']['smoke']['measurements']} measurements, "
        f"full={data['tiers']['full']['measurements']} measurements"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
