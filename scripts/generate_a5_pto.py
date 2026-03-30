#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import sys


_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ptodsl.lib import a5


_DEFAULT_OUTPUT_DIR = _ROOT / "ptodsl" / "lib" / "a5" / "generated"
_DEFAULT_PTOAS = _ROOT.parent / "PTOAS" / "build-src312" / "tools" / "ptoas" / "ptoas"


def emit_kernels(*, output_dir, ptoas_bin=None, emit_cpp=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for kernel_name, builder in a5.KERNEL_BUILDERS.items():
        module = builder()
        pto_path = output_dir / f"{kernel_name}.pto"
        pto_path.write_text(f"{module}\n", encoding="utf-8")
        generated.append(pto_path)

        if emit_cpp:
            if ptoas_bin is None:
                raise ValueError("`emit_cpp=True` requires `ptoas_bin`.")
            cpp_path = output_dir / f"{kernel_name}.cpp"
            try:
                subprocess.run(
                    [str(ptoas_bin), str(pto_path), "-o", str(cpp_path)],
                    check=True,
                    cwd=str(output_dir),
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"warning: failed to lower {pto_path.name} to C++ with ptoas: {exc}",
                    file=sys.stderr,
                )
    return generated


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PTODSL A5 translation artifacts as `.pto` files."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to write generated artifacts. Default: {_DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--ptoas",
        type=pathlib.Path,
        default=_DEFAULT_PTOAS,
        help=f"ptoas binary to use when `--emit-cpp` is set. Default: {_DEFAULT_PTOAS}",
    )
    parser.add_argument(
        "--emit-cpp",
        action="store_true",
        help="Also run ptoas and write `.cpp` files next to the generated `.pto` files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    ptoas_bin = args.ptoas if args.emit_cpp else None
    generated = emit_kernels(
        output_dir=args.output_dir,
        ptoas_bin=ptoas_bin,
        emit_cpp=args.emit_cpp,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
