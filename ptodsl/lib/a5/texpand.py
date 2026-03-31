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
    matrix_active_lanes,
    micro_lane_count,
    ptr,
    raw,
    range_constexpr,
    resolve_tile_spec,
    s,
    store_view,
    load_view,
    vreg_type,
    require_view_dtype,
    require_view_shape,
)


def tcol_expand(
    src_view,
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
        context="TCOLEXPAND",
    )
    rows, cols = check_col_expand_operands(
        src_view, out_view, dtype=dtype, shape=[rows, cols], context="TCOLEXPAND"
    )
    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)

    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)
    src_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        valid_shape=[1, type_valid_shape[1]],
        addr=src_addr,
        valid_row=1,
        valid_col=valid_col,
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        valid_shape=type_valid_shape,
        addr=out_addr,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    load_view(src_view, src_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)

    for col in range_constexpr(0, cols, lanes):
        active = matrix_active_lanes(1, valid_col, 0, col, lanes)
        mask = mask_for_chunk(dtype, active)
        col_offset = s.const(col)
        vec = pto.vlds(vector_type, src_ptr, raw(col_offset))
        for row in range_constexpr(rows):
            dst_offset = s.const(row * cols + col)
            pto.vsts(vec, out_ptr, raw(dst_offset), mask)

    store_view(out_tile, out_view)
    return out_view


def trow_expand(
    src_view,
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
        context="TROWEXPAND",
    )
    rows, cols = check_row_expand_operands(
        src_view, out_view, dtype=dtype, shape=[rows, cols], context="TROWEXPAND"
    )
    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)

    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)
    src_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        valid_shape=[type_valid_shape[0], 1],
        addr=src_addr,
        valid_row=valid_row,
        valid_col=1,
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        valid_shape=type_valid_shape,
        addr=out_addr,
        valid_row=valid_row,
        valid_col=valid_col,
    )
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
            active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
            mask = mask_for_chunk(dtype, active)
            dst_offset = s.const(row * cols + col)
            pto.vsts(broadcast, out_ptr, raw(dst_offset), mask)

    store_view(out_tile, out_view)
    return out_view


def trow_expand_add(
    base_view,
    expand_view,
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
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWEXPANDADD",
        micro_op=pto.vadd,
    )


def trow_expand_sub(
    base_view,
    expand_view,
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
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWEXPANDSUB",
        micro_op=pto.vsub,
    )


def trow_expand_mul(
    base_view,
    expand_view,
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
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWEXPANDMUL",
        micro_op=pto.vmul,
    )


def trow_expand_div(
    base_view,
    expand_view,
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
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWEXPANDDIV",
        micro_op=pto.vdiv,
    )


def trow_expand_max(
    base_view,
    expand_view,
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
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWEXPANDMAX",
        micro_op=pto.vmax,
    )


def trow_expand_min(
    base_view,
    expand_view,
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
    return _trow_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWEXPANDMIN",
        micro_op=pto.vmin,
    )


def tcol_expand_add(
    base_view,
    expand_view,
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
    return _tcol_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLEXPANDADD",
        micro_op=pto.vadd,
    )


def tcol_expand_sub(
    base_view,
    expand_view,
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
    return _tcol_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLEXPANDSUB",
        micro_op=pto.vsub,
    )


def tcol_expand_div(
    base_view,
    expand_view,
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
    return _tcol_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLEXPANDDIV",
        micro_op=pto.vdiv,
    )


def tcol_expand_mul(
    base_view,
    expand_view,
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
    return _tcol_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLEXPANDMUL",
        micro_op=pto.vmul,
    )


def tcol_expand_max(
    base_view,
    expand_view,
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
    return _tcol_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLEXPANDMAX",
        micro_op=pto.vmax,
    )


def tcol_expand_min(
    base_view,
    expand_view,
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
    return _tcol_expand_binary(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLEXPANDMIN",
        micro_op=pto.vmin,
    )


def _trow_expand_binary(
    base_view,
    expand_view,
    out_view,
    *,
    dtype,
    tile_shape,
    shape,
    valid_row,
    valid_col,
    valid_shape,
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
    rows, cols = check_row_expand_operands(
        expand_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
        context=context,
    )
    require_view_shape(
        base_view,
        [rows, cols],
        message=f"Fix: {context} base input valid shape mismatch with output tile dst shape.",
    )
    require_view_dtype(
        base_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )

    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)
    base_addr_value = const_i64(base_addr)
    expand_addr_value = const_i64(base_addr + buf_bytes)
    out_addr_value = const_i64(base_addr + buf_bytes * 2)

    base_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=base_addr_value,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    expand_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        valid_shape=[type_valid_shape[0], 1],
        addr=expand_addr_value,
        valid_row=valid_row,
        valid_col=1,
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=out_addr_value,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    load_view(base_view, base_tile)
    load_view(expand_view, expand_tile)

    base_ptr = pto.castptr(ptr(dtype, space="VEC"), base_addr_value)
    expand_ptr = pto.castptr(ptr(dtype, space="VEC"), expand_addr_value)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr_value)

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
            active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
            mask = mask_for_chunk(dtype, active)
            row_offset = s.const(row * cols + col)
            base_vec = pto.vlds(vector_type, base_ptr, raw(row_offset))
            out_vec = micro_op(vector_type, base_vec, broadcast, mask)
            pto.vsts(out_vec, out_ptr, raw(row_offset), mask)

    store_view(out_tile, out_view)
    return out_view


def _tcol_expand_binary(
    base_view,
    expand_view,
    out_view,
    *,
    dtype,
    tile_shape,
    shape,
    valid_row,
    valid_col,
    valid_shape,
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
    rows, cols = check_col_expand_operands(
        expand_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
        context=context,
    )
    require_view_shape(
        base_view,
        [rows, cols],
        message=f"Fix: {context} base input valid shape mismatch with output tile dst shape.",
    )
    require_view_dtype(
        base_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )

    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)
    base_addr_value = const_i64(base_addr)
    expand_addr_value = const_i64(base_addr + buf_bytes)
    out_addr_value = const_i64(base_addr + buf_bytes * 2)

    base_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=base_addr_value,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    expand_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        valid_shape=[1, type_valid_shape[1]],
        addr=expand_addr_value,
        valid_row=1,
        valid_col=valid_col,
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=out_addr_value,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    load_view(base_view, base_tile)
    load_view(expand_view, expand_tile)

    base_ptr = pto.castptr(ptr(dtype, space="VEC"), base_addr_value)
    expand_ptr = pto.castptr(ptr(dtype, space="VEC"), expand_addr_value)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr_value)

    for col in range_constexpr(0, cols, lanes):
        active = matrix_active_lanes(1, valid_col, 0, col, lanes)
        mask = mask_for_chunk(dtype, active)
        col_offset = s.const(col)
        expand_vec = pto.vlds(vector_type, expand_ptr, raw(col_offset))
        for row in range_constexpr(rows):
            offset = s.const(row * cols + col)
            base_vec = pto.vlds(vector_type, base_ptr, raw(offset))
            out_vec = micro_op(vector_type, base_vec, expand_vec, mask)
            pto.vsts(out_vec, out_ptr, raw(offset), mask)

    store_view(out_tile, out_view)
    return out_view


__all__ = [
    "tcol_expand_add",
    "tcol_expand_div",
    "tcol_expand",
    "tcol_expand_max",
    "tcol_expand_min",
    "tcol_expand_mul",
    "tcol_expand_sub",
    "trow_expand_add",
    "trow_expand",
    "trow_expand_div",
    "trow_expand_max",
    "trow_expand_min",
    "trow_expand_mul",
    "trow_expand_sub",
]
