"""Implement gather/sort tile ops with PTO vector micro instructions."""

import builtins

from mlir.dialects import pto

from ._common import (
    alloc_tile_buffer,
    check_gather_operands,
    check_mrgsort_operands,
    check_sort32_operands,
    const_i64,
    dtype_byte_width,
    mask_for_chunk,
    micro_lane_count,
    ptr,
    raw,
    range_constexpr,
    s,
    store_view,
    uint32_type,
    load_view,
    vreg_type,
)


def tgather(
    src_view,
    indices_view,
    out_view,
    *,
    dtype,
    index_dtype=None,
    shape,
    base_addr=0,
):
    index_dtype = uint32_type() if index_dtype is None else index_dtype
    rows, cols = check_gather_operands(
        src_view,
        indices_view,
        out_view,
        dtype=dtype,
        index_dtype=index_dtype,
        shape=shape,
    )
    src_bytes = rows * cols * dtype_byte_width(dtype)
    idx_bytes = rows * cols * dtype_byte_width(index_dtype)

    src_addr = const_i64(base_addr)
    idx_addr = const_i64(base_addr + src_bytes)
    out_addr = const_i64(base_addr + src_bytes + idx_bytes)

    src_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=src_addr)
    idx_tile = alloc_tile_buffer(index_dtype, shape, space="VEC", addr=idx_addr)
    out_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=out_addr)
    load_view(src_view, src_tile)
    load_view(indices_view, idx_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    idx_ptr = pto.castptr(ptr(index_dtype, space="VEC"), idx_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)
    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    index_vector_type = vreg_type(micro_lane_count(index_dtype), index_dtype)

    for row in range_constexpr(rows):
        row_base = row * cols
        for col in range_constexpr(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            offset = s.const(row_base + col)
            mask = mask_for_chunk(dtype, active)
            idx_vec = pto.vlds(index_vector_type, idx_ptr, raw(offset))
            out_vec = pto.vgather2(vector_type, src_ptr, idx_vec, raw(s.const(active)))
            pto.vsts(out_vec, out_ptr, raw(offset), mask)

    store_view(out_tile, out_view)
    return out_view


def tmrgsort(src_view, out_view, *, dtype, shape, block_len, base_addr=0):
    _, cols = check_mrgsort_operands(
        src_view, out_view, dtype=dtype, shape=shape, block_len=block_len
    )
    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + cols * dtype_byte_width(dtype))

    src_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=src_addr)
    out_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=out_addr)
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


def tsort32(src_view, idx_view, out_view, *, dtype, shape, base_addr=0):
    rows, cols, out_cols = check_sort32_operands(
        src_view, idx_view, out_view, dtype=dtype, shape=shape
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
