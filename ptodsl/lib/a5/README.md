# A5 Library Layer

This directory contains a PTODSL library-style translation layer for the
`pto-isa/include/pto/npu/a5` surface, organized around the PTO tile opcode that
each file is re-expressing with PTO micro instructions.

The scope of this layout is:

- Small, readable files that show how a tile helper is written from PTO micro
  opcodes such as `pto.vlds`, `pto.vadd`, and `pto.vsts`
- Canonical tile helper names such as `tadd`, `trow_sum`, and `tgather`, plus
  matching A5 header aliases such as `TAdd` where parity with `pto-isa` matters
- Example builder kernels that emit `.pto` through PTODSL
- A checked-in generation flow for reproducible `.pto` artifacts and HIVM LLVM
  sidecars for pure micro kernels

Entry points:

- [`tbinary.py`](./tbinary.py): tile binary helpers such as `tadd`, `tsub`, `tmul`,
  `tdiv`, `tmax`, `tmin`, and `tor_`, written with PTO vector micro ops
- [`tscalar.py`](./tscalar.py): scalar tile helpers such as `tadds`, `tmaxs`,
  `tlrelu`, and `taxpy`
- [`tunary.py`](./tunary.py): tile unary helpers such as `texp`, `tlog`, `trelu`,
  `tsqrt`, `trsqrt`, and `trecip`
- [`texpand.py`](./texpand.py): row and column broadcast helpers, including add/sub/div/mul/max/min variants
- [`treduce.py`](./treduce.py): row and column reduction helpers
- [`tsort.py`](./tsort.py): gather and sort helpers
- [`native.py`](./native.py): helpers that still map directly to tile/cube ops
- [`ops.py`](./ops.py): the public A5 surface that re-exports the split helpers
- [`a5_header_coverage.py`](./a5_header_coverage.py): tracked status for the wider A5 header inventory
- [`kernels.py`](./kernels.py): translated example kernels, including the
  no-section `build_hivm_vadd_demo()` flow that lowers through PTOAS VPTO into
  `llvm.hivm.*` intrinsics
- [`generated`](./generated): emitted `.pto` artifacts from `scripts/generate_a5_pto.py`

Regenerate the current artifacts with:

```bash
PYTHONPATH=/Users/zhoubot/github/.llvm-19.1.7/build-mlir-py312/tools/mlir/python_packages/mlir_core:/Users/zhoubot/github/pto-org/PTOAS/install-src312:/Users/zhoubot/github/pto-org/PTOAS/build-src312/python \
/Users/zhoubot/github/.venv-ptoas-src312/bin/python scripts/generate_a5_pto.py
```

To also emit HIVM LLVM for the pure micro kernels:

```bash
PYTHONPATH=/Users/zhoubot/github/.llvm-19.1.7/build-mlir-py312/tools/mlir/python_packages/mlir_core:/Users/zhoubot/github/pto-org/PTOAS/install-src312:/Users/zhoubot/github/pto-org/PTOAS/build-src312/python \
/Users/zhoubot/github/.venv-ptoas-src312/bin/python scripts/generate_a5_pto.py --emit-hivm-llvm
```

`--emit-cpp` and `--emit-hivm-llvm` are intentionally asymmetric:
- pure micro kernels such as `a5_hivm_vadd_demo` now lower end-to-end through
  PTOAS VPTO into `llvm.hivm.*`
- `--emit-hivm-llvm` only writes `.ll` sidecars for kernels listed in
  `a5.HIVM_LLVM_KERNELS`
- tensor-view and tile-buffer frontend examples remain useful PTODSL coverage,
  but they are not yet the canonical PTOAS VPTO/HIVM path
