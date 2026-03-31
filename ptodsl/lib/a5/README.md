# A5 Library Layer

This directory contains a first PTODSL library-style translation layer for the
`pto-isa/include/pto/npu/a5` surface.

The scope of this pass is:

- Pythonic wrappers over PTO tile ops and selected micro instructions
- A5-flavored compatibility aliases such as `TLoad`, `TAdd`, `TMatmul`, and `TStore`
- Translated builder kernels that emit `.pto` through PTODSL
- A checked-in generation flow for reproducible `.pto` artifacts

Entry points:

- [`ops.py`](./ops.py): reusable A5-style helpers built on PTODSL and PTO dialect ops
- [`kernels.py`](./kernels.py): translated example kernels
- [`generated`](./generated): emitted `.pto` artifacts from `scripts/generate_a5_pto.py`

Regenerate the current artifacts with:

```bash
PYTHONPATH=/Users/zhoubot/github/.llvm-19.1.7/build-mlir-py312/tools/mlir/python_packages/mlir_core:/Users/zhoubot/github/pto-org/PTOAS/install-src312:/Users/zhoubot/github/pto-org/PTOAS/build-src312/python \
/Users/zhoubot/github/.venv-ptoas-src312/bin/python scripts/generate_a5_pto.py
```

`--emit-cpp` is best-effort: the tile-based kernels lower through local `ptoas`,
while the direct micro-only kernel currently remains `.pto`-only in this environment.
