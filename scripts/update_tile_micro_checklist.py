#!/usr/bin/env python3

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptodsl.lib.a5.tile_micro_coverage import coverage_markdown


def main():
    target = _REPO_ROOT / "ptodsl" / "lib" / "a5" / "TILE_MICRO_CHECKLIST.md"
    target.write_text(coverage_markdown(), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
