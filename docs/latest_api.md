# PTODSL Latest API

This document summarizes the public PTODSL surface after the recent A5/micro
upgrade and explains which layer to use for each kind of kernel.

## Public Layers

### `ptodsl.pto`

Use this layer for:

- pointer/type construction
- tensor and partitioned-view authoring
- tile buffer allocation and `tload`/`tstore`
- control flow and synchronization

Key entry points:

- `pto.ptr(dtype, space=None)`
- `pto.TensorType(rank=..., dtype=...)`
- `pto.SubTensorType(shape=..., dtype=...)`
- `pto.TileBufType(shape=..., dtype=..., memory_space=..., valid_shape=..., config=...)`
- `pto.make_tensor(ptr, shape=..., strides=None, dtype=..., type=None, layout=None)`
- `TensorView.slice(offsets, sizes, static_shape=None)`
- `pto.make_tile_buffer(dtype, shape, space=..., valid_shape=None, config=None)`
- `TileBufferSpec.alloc(addr=None, valid_row=None, valid_col=None)`
- `TileBuffer.load_from(view)` / `TileBuffer.store_to(view)`

Type aliases currently exposed through `pto`:

- `bool`
- `float16`
- `float32`
- `bfloat16`
- `int8`
- `int16`
- `int32`
- `uint8`
- `uint16`
- `uint32`

## `ptodsl.micro`

Use this layer when you want raw PTO micro instructions without going through
tile helpers.

Examples:

- `micro.vlds`
- `micro.vadd`
- `micro.vsts`
- `micro.vcadd`
- `micro.vgather2`
- `micro.vmrgsort4`
- `micro.vbitsort`
- `micro.pset_b32`
- `micro.VRegType`
- `micro.MaskType`

This layer is a thin pass-through over the PTO dialect Python bindings, filtered
to the public micro-op surface.

## `ptodsl.lib.a5`

Use the A5 library when you want readable, opcode-focused examples of how an
existing A5 tile helper is expressed with PTO micro instructions.

Examples:

- `a5.tadd`
- `a5.tadds`
- `a5.trow_sum`
- `a5.tcol_expand`
- `a5.tgather`
- `a5.tsort32`

The split modules in [`ptodsl/lib/a5`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5)
are organized by tile helper family:

- [`tbinary.py`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/tbinary.py)
- [`tscalar.py`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/tscalar.py)
- [`tunary.py`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/tunary.py)
- [`texpand.py`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/texpand.py)
- [`treduce.py`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/treduce.py)
- [`tsort.py`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/tsort.py)

## Compile-Time vs Runtime Values

PTODSL now follows the same staging model as the PTO C++ tile headers:

- compile-time constants:
  dtype, memory space, tile capacity, tile layout/config, specialization knobs
- runtime values:
  pointers, offsets, valid row/column bounds, problem sizes

In practice:

- `tile_shape=[ROWS, COLS]` describes the fixed tile envelope
- `valid_row` and `valid_col` describe the runtime active region when the valid
  box is dynamic
- `Constexpr[...]` is used in template-style builders such as
  `build_templated_elementwise_add`

## End-to-End Lowering

The strongest validated path today is:

1. write a pure micro kernel in PTODSL
2. emit `.pto`
3. lower with PTOAS VPTO
4. inspect emitted `llvm.hivm.*` intrinsics

Reference artifacts:

- [`a5_hivm_vadd_demo.pto`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/generated/a5_hivm_vadd_demo.pto)
- [`a5_hivm_vadd_demo.ll`](/Users/zhoubot/github/pto-org/pto-dsl/ptodsl/lib/a5/generated/a5_hivm_vadd_demo.ll)

The higher-level tensor/tile frontend remains fully useful for PTODSL authoring
and regression coverage, but the pure micro path is the clearest proof route
for PTOAS HIVM lowering.
