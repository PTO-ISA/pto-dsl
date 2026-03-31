# A5 Library Layer

This directory contains a PTODSL library-style translation layer for the
`pto-isa/include/pto/npu/a5` surface, organized around the PTO tile opcode that
each file is re-expressing with PTO micro instructions.

The scope of this layout is:

- Small, readable files that show how a tile helper is written from PTO micro
  opcodes such as `pto.vlds`, `pto.vadd`, and `pto.vsts`
- A5-flavored aliases such as `TLoad`, `TAdd`, `TMatmul`, and `TStore`
- Example builder kernels that emit `.pto` through PTODSL
- A checked-in generation flow for reproducible `.pto` artifacts

Entry points:

- [`tbinary.py`](./tbinary.py): tile binary helpers such as `tadd`, `tsub`, `tmul`,
  `tdiv`, and `tor_`, written with PTO vector micro ops
- [`tunary.py`](./tunary.py): tile unary helpers such as `texp`, `tlog`, `trelu`,
  `tsqrt`, `trsqrt`, and `trecip`
- [`texpand.py`](./texpand.py): row and column broadcast helpers
- [`treduce.py`](./treduce.py): row and column reduction helpers
- [`tsort.py`](./tsort.py): gather and sort helpers
- [`native.py`](./native.py): helpers that still map directly to tile/cube ops
- [`ops.py`](./ops.py): the public A5 surface that re-exports the split helpers
- [`kernels.py`](./kernels.py): translated example kernels
- [`generated`](./generated): emitted `.pto` artifacts from `scripts/generate_a5_pto.py`

Regenerate the current artifacts with:

```bash
PYTHONPATH=/Users/zhoubot/github/.llvm-19.1.7/build-mlir-py312/tools/mlir/python_packages/mlir_core:/Users/zhoubot/github/pto-org/PTOAS/install-src312:/Users/zhoubot/github/pto-org/PTOAS/build-src312/python \
/Users/zhoubot/github/.venv-ptoas-src312/bin/python scripts/generate_a5_pto.py
```

`--emit-cpp` is best-effort: the tile-based kernels lower through local `ptoas`,
while the direct micro-only kernel currently remains `.pto`-only in this environment.
