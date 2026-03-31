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


def _run_ptoas(ptoas_bin, args, *, output_dir, warning):
    try:
        subprocess.run(
            [str(ptoas_bin), *args],
            check=True,
            cwd=str(output_dir),
        )
    except subprocess.CalledProcessError as exc:
        print(f"warning: {warning}: {exc}", file=sys.stderr)


def emit_kernels(
    *,
    output_dir,
    ptoas_bin=None,
    emit_cpp=False,
    emit_hivm_llvm=False,
    kernel_names=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    selected = set(kernel_names) if kernel_names is not None else None
    hivm_selected = (
        set(a5.HIVM_LLVM_KERNELS)
        if selected is None
        else set(a5.HIVM_LLVM_KERNELS).intersection(selected)
    )
    for kernel_name, builder in a5.KERNEL_BUILDERS.items():
        if selected is not None and kernel_name not in selected:
            continue
        module = builder()
        pto_path = output_dir / f"{kernel_name}.pto"
        pto_path.write_text(f"{module}\n", encoding="utf-8")
        generated.append(pto_path)

        if emit_cpp or emit_hivm_llvm:
            if ptoas_bin is None:
                raise ValueError(
                    "`emit_cpp=True` and `emit_hivm_llvm=True` require `ptoas_bin`."
                )

        if emit_cpp:
            cpp_path = output_dir / f"{kernel_name}.cpp"
            _run_ptoas(
                ptoas_bin,
                [str(pto_path), "-o", str(cpp_path)],
                output_dir=output_dir,
                warning=f"failed to lower {pto_path.name} to C++ with ptoas",
            )

        if emit_hivm_llvm and kernel_name in hivm_selected:
            llvm_path = output_dir / f"{kernel_name}.ll"
            _run_ptoas(
                ptoas_bin,
                [
                    "--pto-arch=a5",
                    "--pto-level=level3",
                    "--pto-backend=vpto",
                    "--vpto-emit-hivm-llvm",
                    str(pto_path),
                    "-o",
                    str(llvm_path),
                ],
                output_dir=output_dir,
                warning=f"failed to lower {pto_path.name} to HIVM LLVM with ptoas",
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
    parser.add_argument(
        "--emit-hivm-llvm",
        action="store_true",
        help="Also run PTOAS VPTO lowering and write `.ll` files for kernels that are marked HIVM-lowerable.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    ptoas_bin = args.ptoas if args.emit_cpp or args.emit_hivm_llvm else None
    generated = emit_kernels(
        output_dir=args.output_dir,
        ptoas_bin=ptoas_bin,
        emit_cpp=args.emit_cpp,
        emit_hivm_llvm=args.emit_hivm_llvm,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
