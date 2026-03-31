"""Implement tile unary ops with PTO vector micro instructions.

This file demonstrates how to write tile-style helpers such as `pto.texp`
directly in terms of `pto.vlds`, a unary vector opcode, and `pto.vsts`.
"""

from mlir.dialects import pto

from ._common import (
    alloc_tile_buffer,
    check_tbinop_operands,
    const_i64,
    dtype_byte_width,
    flat_active_lanes,
    mask_for_chunk,
    ptr,
    raw,
    range_constexpr,
    resolve_tile_spec,
    resolve_lanes,
    s,
    store_view,
    load_view,
    vreg_type,
)


def texp(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    lanes=None,
    base_addr=0,
):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TEXP",
        micro_op=pto.vexp,
    )


def tlog(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    lanes=None,
    base_addr=0,
):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TLOG",
        micro_op=pto.vln,
    )


def trelu(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    lanes=None,
    base_addr=0,
):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TRELU",
        micro_op=pto.vrelu,
    )


def tabs(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    lanes=None,
    base_addr=0,
):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TABS",
        micro_op=pto.vabs,
    )


def tsqrt(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    lanes=None,
    base_addr=0,
):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TSQRT",
        micro_op=pto.vsqrt,
    )


def trecip(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    lanes=None,
    base_addr=0,
):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TRECIP",
        micro_op=pto.vrec,
    )


def trsqrt(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape=None,
    shape=None,
    valid_row=None,
    valid_col=None,
    valid_shape=None,
    lanes=None,
    base_addr=0,
):
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context="TRSQRT",
    )
    rows, cols = check_tbinop_operands(
        src_view,
        src_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
        context="TRSQRT",
    )
    lanes = resolve_lanes(dtype, lanes)
    element_count = rows * cols
    buf_bytes = element_count * dtype_byte_width(dtype)
    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)

    src_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=src_addr,
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

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)

    for offset in range_constexpr(0, element_count, lanes):
        active = flat_active_lanes(valid_row, valid_col, offset, lanes)
        mask = mask_for_chunk(dtype, active)
        index = s.const(offset)
        src_vec = pto.vlds(vector_type, src_ptr, raw(index))
        sqrt_vec = pto.vsqrt(vector_type, src_vec, mask)
        out_vec = pto.vrec(vector_type, sqrt_vec, mask)
        pto.vsts(out_vec, out_ptr, raw(index), mask)

    store_view(out_tile, out_view)
    return out_view


def _unary_tile_vop(
    src_view,
    out_view,
    *,
    dtype,
    tile_shape,
    shape,
    valid_row,
    valid_col,
    valid_shape,
    lanes,
    base_addr,
    context,
    micro_op,
):
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context=context,
    )
    rows, cols = check_tbinop_operands(
        src_view,
        src_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
        context=context,
    )
    lanes = resolve_lanes(dtype, lanes)
    element_count = rows * cols
    buf_bytes = element_count * dtype_byte_width(dtype)
    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)

    src_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=src_addr,
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

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)

    for offset in range_constexpr(0, element_count, lanes):
        active = flat_active_lanes(valid_row, valid_col, offset, lanes)
        mask = mask_for_chunk(dtype, active)
        index = s.const(offset)
        src_vec = pto.vlds(vector_type, src_ptr, raw(index))
        out_vec = micro_op(vector_type, src_vec, mask)
        pto.vsts(out_vec, out_ptr, raw(index), mask)

    store_view(out_tile, out_view)
    return out_view


__all__ = ["tabs", "texp", "tlog", "trecip", "trelu", "trsqrt", "tsqrt"]
