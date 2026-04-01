"""Implement sort-style tile ops with PTO vector micro instructions."""

from mlir.dialects import pto

from ._common import (
    alloc_tile_buffer,
    check_mrgsort_operands,
    check_sort32_operands,
    const_i64,
    dtype_byte_width,
    ptr,
    raw,
    range_constexpr,
    resolve_tile_spec,
    s,
    store_view,
    load_view,
    uint32_type,
)
from .tindex import tgather


def tmrgsort(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    block_len,
    base_addr=0,
):
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context="TMRGSORT",
    )
    _, cols = check_mrgsort_operands(
        src_view, out_view, dtype=dtype, shape=[rows, cols], block_len=block_len
    )
    if (
        not isinstance(valid_row, int)
        or not isinstance(valid_col, int)
        or valid_row != rows
        or valid_col != cols
        or type_valid_shape != [rows, cols]
    ):
        raise ValueError(
            "TMRGSORT micro lowering currently requires a fully valid single-row tile."
        )
    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + cols * dtype_byte_width(dtype))

    src_tile = alloc_tile_buffer(dtype, [rows, cols], space="VEC", addr=src_addr)
    out_tile = alloc_tile_buffer(dtype, [rows, cols], space="VEC", addr=out_addr)
    load_view(src_view, src_tile)

    ptr_type = ptr(dtype, space="VEC")
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)

    src1_ptr = pto.addptr(src_ptr, raw(s.const(block_len)))
    src2_ptr = pto.addptr(src_ptr, raw(s.const(block_len * 2)))
    src3_ptr = pto.addptr(src_ptr, raw(s.const(block_len * 3)))
    num_structures = (block_len * dtype_byte_width(dtype)) >> 3
    count_value = (
        num_structures
        | (num_structures << 16)
        | (num_structures << 32)
        | (num_structures << 48)
    )
    repeat_times = cols // (block_len * 4)
    config_value = repeat_times | (0b1111 << 8)

    pto.vmrgsort4(
        out_ptr,
        src_ptr,
        src1_ptr,
        src2_ptr,
        src3_ptr,
        const_i64(count_value),
        const_i64(config_value),
    )
    store_view(out_tile, out_view)
    return out_view


def tsort32(
    src_view,
    idx_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    base_addr=0,
):
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context="TSORT32",
    )
    rows, cols, out_cols = check_sort32_operands(
        src_view, idx_view, out_view, dtype=dtype, shape=[rows, cols]
    )
    if (
        not isinstance(valid_row, int)
        or not isinstance(valid_col, int)
        or valid_row != rows
        or valid_col != cols
        or type_valid_shape != [rows, cols]
    ):
        raise ValueError(
            "TSORT32 micro lowering currently requires a fully valid input tile."
        )
    src_bytes = rows * cols * dtype_byte_width(dtype)
    idx_bytes = rows * cols * 4

    src_addr = const_i64(base_addr)
    idx_addr = const_i64(base_addr + src_bytes)
    out_addr = const_i64(base_addr + src_bytes + idx_bytes)

    src_tile = alloc_tile_buffer(dtype, [rows, cols], space="VEC", addr=src_addr)
    idx_tile = alloc_tile_buffer(
        uint32_type(), [rows, cols], space="VEC", addr=idx_addr
    )
    out_tile = alloc_tile_buffer(dtype, [rows, out_cols], space="VEC", addr=out_addr)
    load_view(src_view, src_tile)
    load_view(idx_view, idx_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    idx_ptr = pto.castptr(ptr(uint32_type(), space="VEC"), idx_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)
    repeat_times = s.const(cols // 32)

    for row in range_constexpr(rows):
        src_row = pto.addptr(src_ptr, raw(s.const(row * cols)))
        idx_row = pto.addptr(idx_ptr, raw(s.const(row * cols)))
        out_row = pto.addptr(out_ptr, raw(s.const(row * out_cols)))
        pto.vbitsort(out_row, src_row, idx_row, raw(repeat_times))

    store_view(out_tile, out_view)
    return out_view


__all__ = ["tgather", "tmrgsort", "tsort32"]
