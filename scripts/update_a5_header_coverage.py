#!/usr/bin/env python3

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptodsl.lib.a5.a5_header_coverage import a5_header_coverage_markdown


def main():
    target = _REPO_ROOT / "ptodsl" / "lib" / "a5" / "A5_HEADER_COVERAGE.md"
    target.write_text(a5_header_coverage_markdown(), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
