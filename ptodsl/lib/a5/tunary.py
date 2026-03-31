"""Implement tile unary ops with PTO vector micro instructions.

This file demonstrates how to write tile-style helpers such as `pto.texp`
directly in terms of `pto.vlds`, a unary vector opcode, and `pto.vsts`.
"""

import builtins

from mlir.dialects import pto

from ._common import (
    alloc_tile_buffer,
    check_tbinop_operands,
    const_i64,
    dtype_byte_width,
    mask_for_chunk,
    ptr,
    raw,
    range_constexpr,
    resolve_lanes,
    s,
    store_view,
    load_view,
    vreg_type,
)


def texp(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vexp",
        context="TEXP",
    )


def tlog(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vln",
        context="TLOG",
    )


def trelu(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vrelu",
        context="TRELU",
    )


def tabs(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vabs",
        context="TABS",
    )


def tsqrt(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vsqrt",
        context="TSQRT",
    )


def trecip(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_tile_vop(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vrec",
        context="TRECIP",
    )


def trsqrt(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    rows, cols = check_tbinop_operands(
        src_view,
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        context="TRSQRT",
    )
    lanes = resolve_lanes(dtype, lanes)
    element_count = rows * cols
    buf_bytes = element_count * dtype_byte_width(dtype)
    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)

    src_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=src_addr)
    out_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=out_addr)
    load_view(src_view, src_tile)

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)

    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
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
    shape,
    lanes,
    base_addr,
    micro_op_name,
    context,
):
    rows, cols = check_tbinop_operands(
        src_view,
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        context=context,
    )
    lanes = resolve_lanes(dtype, lanes)
    element_count = rows * cols
    buf_bytes = element_count * dtype_byte_width(dtype)
    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)

    src_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=src_addr)
    out_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=out_addr)
    load_view(src_view, src_tile)

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)
    micro_op = getattr(pto, micro_op_name)

    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
        mask = mask_for_chunk(dtype, active)
        index = s.const(offset)
        src_vec = pto.vlds(vector_type, src_ptr, raw(index))
        out_vec = micro_op(vector_type, src_vec, mask)
        pto.vsts(out_vec, out_ptr, raw(index), mask)

    store_view(out_tile, out_view)
    return out_view


__all__ = ["tabs", "texp", "tlog", "trecip", "trelu", "trsqrt", "tsqrt"]
