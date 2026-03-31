#!/usr/bin/env python3

import pathlib
import sys


_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ptodsl.lib import a5


_DEFAULT_OUTPUT_DIR = _ROOT / "ptodsl" / "lib" / "a5" / "generated" / "tile_ops"


def emit_tile_ops(*, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for op_name, builder in a5.TILE_OP_KERNEL_BUILDERS.items():
        module = builder()
        path = output_dir / f"{op_name}.pto"
        path.write_text(f"{module}\n", encoding="utf-8")
        generated.append(path)

    index_path = output_dir / "TILE_OP_GENERATION_INDEX.md"
    index_path.write_text(a5.tile_op_generation_index_markdown(), encoding="utf-8")
    generated.append(index_path)
    return generated


def main():
    generated = emit_tile_ops(output_dir=_DEFAULT_OUTPUT_DIR)
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
