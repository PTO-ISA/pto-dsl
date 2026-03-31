# PTO-DSL

Python DSL for PTO-ISA kernels, with a public `pto` surface for tensor/tile
authoring, a raw `micro` surface for direct PTO micro instructions, and an A5
library layer that rewrites tile-style helpers in terms of those micro ops.

The current repo targets three authoring levels:

- `ptodsl.pto`: ergonomic tensor, view, tile, sync, and control-flow helpers
- `ptodsl.micro`: raw PTO micro instruction access such as `vlds`, `vadd`,
  `vsts`, `pset_b32`, and vector register types
- `ptodsl.lib.a5`: readable A5 helper implementations that show how tile-style
  operations are written with PTO micro instructions

## Recent Upgrade

The recent PTODSL upgrade changed the repo in four important ways:

1. `pto.ptr(dtype, space=...)` is now the preferred pointer constructor for
   explicit memory spaces such as `GM`, `VEC`, `LEFT`, `RIGHT`, and `ACC`.
2. The public `pto` namespace now includes pythonic builders:
   `make_tensor(...)`, `TensorView.slice(...)`, `make_tile_buffer(...)`,
   `TileBufferSpec.alloc()`, and `TileBuffer.load_from()/store_to()`.
3. The package root now exposes `ptodsl.micro` as the raw micro-op surface.
4. The A5 library under [`ptodsl/lib/a5`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5)
   is organized around tile helpers implemented with PTO micro instructions,
   and selected pure-micro kernels are validated through PTOAS into
   `llvm.hivm.*` intrinsics.

Detailed API notes are in
[`docs/latest_api.md`](/Users/zhoubot/github/pto-org/pto-dsl/docs/latest_api.md).

## Install

PTODSL depends on the PTO dialect Python bindings from PTOAS and an MLIR Python
environment. For a reproducible setup, start with
[`docker/README.md`](/Users/zhoubot/github/pto-org/pto-dsl/docker/README.md).

For local development:

```bash
git clone https://github.com/PTO-ISA/pto-dsl.git
cd pto-dsl
pip install -e .
```

Typical local testing in this repo also needs PTOAS and MLIR on `PYTHONPATH`,
for example:

```bash
PYTHONPATH=/path/to/mlir_core:/path/to/PTOAS/install:/path/to/PTOAS/build/python \
python -m pytest -q tests/frontend tests/regression
```

## Public API

### 1. Pythonic `pto`

Use `ptodsl.pto` for tensor/view/tile construction:

```python
from mlir.ir import IndexType
from ptodsl import pto, tile, to_ir_module


def meta_data():
    return {
        "ptr_t": pto.ptr(pto.float32),
        "index_t": IndexType.get(),
    }


@to_ir_module(meta_data=meta_data)
def add_tile(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t", valid_row: "index_t", valid_col: "index_t") -> None:
    lhs = pto.make_tensor(src0, shape=[8, 64], dtype=pto.float32)
    rhs = pto.make_tensor(src1, shape=[8, 64], dtype=pto.float32)
    out = pto.make_tensor(dst, shape=[8, 64], dtype=pto.float32)

    lhs_tile = lhs.slice([0, 0], [8, 64])
    rhs_tile = rhs.slice([0, 0], [8, 64])
    out_tile = out.slice([0, 0], [8, 64])

    with pto.vector_section():
        lhs_buf = pto.make_tile_buffer(
            pto.float32,
            [8, 64],
            space="VEC",
            valid_shape=[-1, -1],
        ).alloc(valid_row=valid_row, valid_col=valid_col)
        rhs_buf = pto.make_tile_buffer(
            pto.float32,
            [8, 64],
            space="VEC",
            valid_shape=[-1, -1],
        ).alloc(valid_row=valid_row, valid_col=valid_col)
        out_buf = pto.make_tile_buffer(
            pto.float32,
            [8, 64],
            space="VEC",
            valid_shape=[-1, -1],
        ).alloc(valid_row=valid_row, valid_col=valid_col)

        lhs_buf.load_from(lhs_tile)
        rhs_buf.load_from(rhs_tile)
        tile.add(lhs_buf, rhs_buf, out_buf)
        out_buf.store_to(out_tile)
```

This still emits native PTO tensor/tile IR such as `pto.make_tensor_view`,
`pto.partition_view`, `pto.alloc_tile`, `pto.tload`, `pto.tadd`, and
`pto.tstore`.

### 2. Raw `micro`

Use `ptodsl.micro` when you want to write the micro instruction sequence
directly:

```python
from mlir.ir import IndexType
from ptodsl import micro, pto, to_ir_module
from ptodsl.api.scalar import _unwrap


def meta_data():
    return {
        "ptr_t": pto.ptr(pto.float32, space="VEC"),
        "index_t": IndexType.get(),
    }


@to_ir_module(meta_data=meta_data)
def vadd_demo(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t", offset: "index_t") -> None:
    v64f32 = micro.VRegType.get(64, pto.float32)
    mask = micro.pset_b32(micro.MaskType.get(), "PAT_ALL")
    lhs = micro.vlds(v64f32, _unwrap(src0), _unwrap(offset))
    rhs = micro.vlds(v64f32, _unwrap(src1), _unwrap(offset))
    out = micro.vadd(v64f32, lhs, rhs, mask)
    micro.vsts(out, _unwrap(dst), _unwrap(offset), mask)
```

This is the most direct PTODSL surface for VPTO/PTOAS lowering.

### 3. A5 Library

The A5 layer under [`ptodsl/lib/a5`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5)
shows how tile-style helpers map to micro instructions:

- `tadd` is written with `pto.vlds`, `pto.vadd`, and `pto.vsts`
- `trow_sum` is written with `pto.vcadd` plus vector combine/store logic
- `tcol_expand`, `tgather`, `tmrgsort`, and `tsort32` are expressed directly in
  terms of PTO micro opcodes where supported

See [`ptodsl/lib/a5/README.md`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/README.md)
for the file layout and generation flow.

## End-to-End Flow

The repo currently tracks two useful flows:

- PTODSL frontend coverage:
  tensor/view/tile and A5 examples emit correct `.pto`
- PTODSL -> PTOAS -> HIVM proof path:
  pure micro kernels such as
  [`a5_hivm_vadd_demo.pto`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/generated/a5_hivm_vadd_demo.pto)
  lower through PTOAS into
  [`a5_hivm_vadd_demo.ll`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/generated/a5_hivm_vadd_demo.ll)

Generated examples live in
[`ptodsl/lib/a5/generated`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/generated).

## Tests

The repo currently uses:

- [`tests/frontend`](/Users/zhoubot/github/pto-org/pto-dsl/tests/frontend) for
  frontend IR construction
- [`tests/regression`](/Users/zhoubot/github/pto-org/pto-dsl/tests/regression)
  for A5 library coverage, generated artifact expectations, and public-surface
  regressions

Run them with:

```bash
PYTHONPATH=/path/to/mlir_core:/path/to/PTOAS/install:/path/to/PTOAS/build/python \
python -m pytest -q tests/frontend tests/regression
```

## Related Files

- [`docs/latest_api.md`](/Users/zhoubot/github/pto-org/pto-dsl/docs/latest_api.md)
- [`ptodsl/api/pto.py`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/api/pto.py)
- [`ptodsl/api/micro.py`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/api/micro.py)
- [`ptodsl/lib/a5/README.md`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/README.md)
- [`contribute_guide.md`](/Users/zhoubot/github/pto-org/pto-dsl/contribute_guide.md)
