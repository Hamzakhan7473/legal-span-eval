#!/usr/bin/env python3
"""Write or verify data/eval_set/MANIFEST.json.

    python scripts/eval_set_manifest.py write
    python scripts/eval_set_manifest.py verify
"""

from __future__ import annotations

import sys

from lse.data.eval_set import MANIFEST_PATH, verify_manifest, write_manifest


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "verify"
    if command == "write":
        print(f"wrote {write_manifest()}")
        return 0
    if command == "verify":
        problems = verify_manifest()
        if problems:
            print(f"{MANIFEST_PATH} does not match the committed files:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{MANIFEST_PATH} matches the committed files")
        return 0
    print(f"unknown command {command!r}; use 'write' or 'verify'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
