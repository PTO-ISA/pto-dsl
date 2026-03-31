"""Implement tile broadcast/expand ops with PTO vector micro instructions."""

import builtins

from mlir.dialects import pto

from ._common import (
    alloc_tile_buffer,
    check_col_expand_operands,
    check_row_expand_operands,
    const_i64,
    dtype_byte_width,
    mask_for_chunk,
    micro_lane_count,
    ptr,
    raw,
    range_constexpr,
    s,
    store_view,
    load_view,
    vreg_type,
    require_view_dtype,
    require_view_shape,
)


def tcol_expand(src_view, out_view, *, dtype, shape, base_addr=0):
    rows, cols = check_col_expand_operands(
        src_view, out_view, dtype=dtype, shape=shape, context="TCOLEXPAND"
    )
    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)

    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)
    src_tile = alloc_tile_buffer(
        dtype, shape, space="VEC", valid_shape=[1, cols], addr=src_addr
    )
    out_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=out_addr)
    load_view(src_view, src_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)

    for col in range_constexpr(0, cols, lanes):
        active = builtins.min(lanes, cols - col)
        mask = mask_for_chunk(dtype, active)
        col_offset = s.const(col)
        vec = pto.vlds(vector_type, src_ptr, raw(col_offset))
        for row in range_constexpr(rows):
            dst_offset = s.const(row * cols + col)
            pto.vsts(vec, out_ptr, raw(dst_offset), mask)

    store_view(out_tile, out_view)
    return out_view


def trow_expand(src_view, out_view, *, dtype, shape, base_addr=0):
    rows, cols = check_row_expand_operands(
        src_view, out_view, dtype=dtype, shape=shape, context="TROWEXPAND"
    )
    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)

    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)
    src_tile = alloc_tile_buffer(
        dtype, shape, space="VEC", valid_shape=[rows, 1], addr=src_addr
    )
    out_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=out_addr)
    load_view(src_view, src_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)

    for row in range_constexpr(rows):
        scalar_offset = s.const(row * cols)
        align = pto.vldas(pto.AlignType.get(), src_ptr, raw(scalar_offset))
        scalar_vec, _, _ = pto.vldus(
            vector_type,
            pto.AlignType.get(),
            ptr(dtype, space="VEC"),
            src_ptr,
            raw(scalar_offset),
            align,
        )
        broadcast = pto.vdup(vector_type, scalar_vec, position="POS_LOWEST")
        for col in range_constexpr(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = mask_for_chunk(dtype, active)
            dst_offset = s.const(row * cols + col)
            pto.vsts(broadcast, out_ptr, raw(dst_offset), mask)

    store_view(out_tile, out_view)
    return out_view


def trow_expand_sub(base_view, expand_view, out_view, *, dtype, shape, base_addr=0):
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        micro_op_name="vsub",
    )


def trow_expand_mul(base_view, expand_view, out_view, *, dtype, shape, base_addr=0):
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        micro_op_name="vmul",
    )


def trow_expand_div(base_view, expand_view, out_view, *, dtype, shape, base_addr=0):
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        micro_op_name="vdiv",
    )


def _trow_expand_binary(
    base_view,
    expand_view,
    out_view,
    *,
    dtype,
    shape,
    base_addr,
    micro_op_name,
):
    rows, cols = check_row_expand_operands(
        expand_view,
        out_view,
        dtype=dtype,
        shape=shape,
        context=f"TROWEXPAND_{micro_op_name[1:].upper()}",
    )
    require_view_shape(
        base_view,
        [rows, cols],
        message=f"Fix: TROWEXPAND_{micro_op_name[1:].upper()} base input valid shape mismatch with output tile dst shape.",
    )
    require_view_dtype(
        base_view,
        dtype,
        message=f"Fix: TROWEXPAND_{micro_op_name[1:].upper()} input data type must be consistent with the output data type.",
    )

    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)
    base_addr_value = const_i64(base_addr)
    expand_addr_value = const_i64(base_addr + buf_bytes)
    out_addr_value = const_i64(base_addr + buf_bytes * 2)

    base_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=base_addr_value)
    expand_tile = alloc_tile_buffer(
        dtype, shape, space="VEC", valid_shape=[rows, 1], addr=expand_addr_value
    )
    out_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=out_addr_value)
    load_view(base_view, base_tile)
    load_view(expand_view, expand_tile)

    base_ptr = pto.castptr(ptr(dtype, space="VEC"), base_addr_value)
    expand_ptr = pto.castptr(ptr(dtype, space="VEC"), expand_addr_value)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr_value)
    micro_op = getattr(pto, micro_op_name)

    for row in range_constexpr(rows):
        scalar_offset = s.const(row * cols)
        align = pto.vldas(pto.AlignType.get(), expand_ptr, raw(scalar_offset))
        scalar_vec, _, _ = pto.vldus(
            vector_type,
            pto.AlignType.get(),
            ptr(dtype, space="VEC"),
            expand_ptr,
            raw(scalar_offset),
            align,
        )
        broadcast = pto.vdup(vector_type, scalar_vec, position="POS_LOWEST")
        for col in range_constexpr(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = mask_for_chunk(dtype, active)
            row_offset = s.const(row * cols + col)
            base_vec = pto.vlds(vector_type, base_ptr, raw(row_offset))
            out_vec = micro_op(vector_type, base_vec, broadcast, mask)
            pto.vsts(out_vec, out_ptr, raw(row_offset), mask)

    store_view(out_tile, out_view)
    return out_view


__all__ = [
    "tcol_expand",
    "trow_expand",
    "trow_expand_div",
    "trow_expand_mul",
    "trow_expand_sub",
]
