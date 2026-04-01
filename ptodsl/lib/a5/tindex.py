"""Implement indexed tile ops with PTO vector micro instructions.

This file demonstrates how to rewrite gather/scatter-style tile helpers
directly in terms of PTO micro instructions such as `pto.vgather2`,
`pto.vgatherb`, and `pto.vscatter`.
"""

from mlir.dialects import arith, pto
from mlir.ir import IndexType

from ._common import (
    alloc_tile_buffer,
    check_gather_operands,
    check_gatherb_operands,
    check_scatter_operands,
    const_expr,
    const_i64,
    const_scalar,
    dtype_byte_width,
    full_mask,
    mask_for_chunk,
    matrix_active_lanes,
    micro_lane_count,
    ptr,
    raw,
    range_constexpr,
    resolve_tile_spec,
    s,
    store_view,
    load_view,
    uint16_type,
    uint32_type,
    vreg_type,
)


def _active_lanes_value(active_lanes):
    if isinstance(active_lanes, int):
        return raw(s.const(active_lanes))
    return arith.IndexCastOp(IndexType.get(), raw(active_lanes)).result


def _zero_tile_buffer(out_ptr, *, dtype, rows, cols):
    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    zero_vec = pto.vbr(vector_type, const_scalar(dtype, 0))
    for row in range_constexpr(rows):
        for col in range_constexpr(0, cols, lanes):
            count = min(cols - col, lanes)
            offset = s.const(row * cols + col)
            mask = mask_for_chunk(dtype, count)
            pto.vsts(zero_vec, out_ptr, raw(offset), mask)


def tgather(
    src_view,
    indices_view,
    out_view,
    *,
    dtype,
    index_dtype=None,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    base_addr=0,
):
    index_dtype = uint32_type() if index_dtype is None else index_dtype
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context="TGATHER",
    )
    rows, cols = check_gather_operands(
        src_view,
        indices_view,
        out_view,
        dtype=dtype,
        index_dtype=index_dtype,
        shape=[rows, cols],
    )
    src_bytes = rows * cols * dtype_byte_width(dtype)
    idx_bytes = rows * cols * dtype_byte_width(index_dtype)

    src_addr = const_i64(base_addr)
    idx_addr = const_i64(base_addr + src_bytes)
    out_addr = const_i64(base_addr + src_bytes + idx_bytes)

    src_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=src_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    idx_tile = alloc_tile_buffer(
        index_dtype,
        [rows, cols],
        space="VEC",
        addr=idx_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=out_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
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
            active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
            offset = s.const(row_base + col)
            mask = mask_for_chunk(dtype, active)
            idx_vec = pto.vlds(index_vector_type, idx_ptr, raw(offset))
            out_vec = pto.vgather2(
                vector_type,
                src_ptr,
                idx_vec,
                _active_lanes_value(active),
            )
            pto.vsts(out_vec, out_ptr, raw(offset), mask)

    store_view(out_tile, out_view)
    return out_view


def tgatherb(
    src_view,
    indices_view,
    out_view,
    *,
    dtype,
    index_dtype=None,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    base_addr=0,
):
    index_dtype = uint32_type() if index_dtype is None else index_dtype
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context="TGATHERB",
    )
    rows, cols = check_gatherb_operands(
        src_view,
        indices_view,
        out_view,
        dtype=dtype,
        index_dtype=index_dtype,
        shape=[rows, cols],
    )
    src_bytes = rows * cols * dtype_byte_width(dtype)
    idx_bytes = rows * cols * dtype_byte_width(index_dtype)

    src_addr = const_i64(base_addr)
    idx_addr = const_i64(base_addr + src_bytes)
    out_addr = const_i64(base_addr + src_bytes + idx_bytes)

    src_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=src_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    idx_tile = alloc_tile_buffer(
        index_dtype,
        [rows, cols],
        space="VEC",
        addr=idx_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=out_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    load_view(src_view, src_tile)
    load_view(indices_view, idx_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    idx_ptr = pto.castptr(ptr(index_dtype, space="VEC"), idx_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)
    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    offset_vector_type = vreg_type(micro_lane_count(index_dtype), index_dtype)
    static_repeat_times = (cols + lanes - 1) // lanes

    if const_expr(static_repeat_times > rows):
        for row in range_constexpr(rows):
            row_base = row * cols
            for col in range_constexpr(0, cols, lanes):
                active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
                offset = s.const(row_base + col)
                mask = mask_for_chunk(dtype, active)
                idx_vec = pto.vlds(offset_vector_type, idx_ptr, raw(offset))
                out_vec = pto.vgatherb(
                    vector_type,
                    src_ptr,
                    idx_vec,
                    _active_lanes_value(active),
                )
                pto.vsts(out_vec, out_ptr, raw(offset), mask)
    else:
        for col in range_constexpr(0, cols, lanes):
            for row in range_constexpr(rows):
                active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
                offset = s.const(row * cols + col)
                mask = mask_for_chunk(dtype, active)
                idx_vec = pto.vlds(offset_vector_type, idx_ptr, raw(offset))
                out_vec = pto.vgatherb(
                    vector_type,
                    src_ptr,
                    idx_vec,
                    _active_lanes_value(active),
                )
                pto.vsts(out_vec, out_ptr, raw(offset), mask)

    store_view(out_tile, out_view)
    return out_view


def tscatter(
    src_view,
    indices_view,
    out_view,
    *,
    dtype,
    index_dtype=None,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    base_addr=0,
):
    if index_dtype is None:
        index_dtype = uint32_type() if dtype_byte_width(dtype) == 4 else uint16_type()
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context="TSCATTER",
    )
    rows, cols, _, _ = check_scatter_operands(
        src_view,
        indices_view,
        out_view,
        dtype=dtype,
        index_dtype=index_dtype,
        shape=[rows, cols],
    )
    src_bytes = rows * cols * dtype_byte_width(dtype)
    idx_bytes = rows * cols * dtype_byte_width(index_dtype)

    src_addr = const_i64(base_addr)
    idx_addr = const_i64(base_addr + src_bytes)
    out_addr = const_i64(base_addr + src_bytes + idx_bytes)

    src_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=src_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    idx_tile = alloc_tile_buffer(
        index_dtype,
        [rows, cols],
        space="VEC",
        addr=idx_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=out_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    load_view(src_view, src_tile)
    load_view(indices_view, idx_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    idx_ptr = pto.castptr(ptr(index_dtype, space="VEC"), idx_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)
    _zero_tile_buffer(out_ptr, dtype=dtype, rows=rows, cols=cols)

    batch = micro_lane_count(index_dtype)
    value_vector_type = vreg_type(batch, dtype)
    index_vector_type = vreg_type(batch, index_dtype)
    load_dist = "UNPK_B8" if dtype_byte_width(dtype) == 1 else None

    for row in range_constexpr(rows):
        row_base = row * cols
        for col in range_constexpr(0, cols, batch):
            active = matrix_active_lanes(valid_row, valid_col, row, col, batch)
            offset = s.const(row_base + col)
            idx_vec = pto.vlds(index_vector_type, idx_ptr, raw(offset))
            src_vec = pto.vlds(
                value_vector_type,
                src_ptr,
                raw(offset),
                dist=load_dist,
            )
            pto.vscatter(src_vec, out_ptr, idx_vec, _active_lanes_value(active))

    store_view(out_tile, out_view)
    return out_view


__all__ = ["tgather", "tgatherb", "tscatter"]
