"""Implement tile binary ops with PTO vector micro instructions.

This file demonstrates how to write tile-style helpers such as `pto.tadd`
directly in terms of `pto.vlds`, `pto.vadd`, and `pto.vsts`.
"""

import builtins

from mlir.dialects import pto

from ._common import (
    VF_IMPL_1D_NO_POST_UPDATE,
    VF_IMPL_1D_POST_UPDATE,
    VF_IMPL_2D_NO_POST_UPDATE,
    VF_IMPL_2D_POST_UPDATE,
    VF_IMPL_DEFAULT,
    alloc_tile_buffer,
    check_tbinop_operands,
    const_expr,
    const_i64,
    dtype_byte_width,
    mask_for_chunk,
    normalize_vf_impl_kind,
    ptr,
    raw,
    range_constexpr,
    resolve_lanes,
    s,
    store_view,
    load_view,
    vreg_type,
)


def tadd(
    lhs_view,
    rhs_view,
    out_view,
    *,
    dtype,
    shape,
    lanes=None,
    base_addr=0,
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vadd",
        impl=impl,
    )


def tsub(
    lhs_view,
    rhs_view,
    out_view,
    *,
    dtype,
    shape,
    lanes=None,
    base_addr=0,
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vsub",
        impl=impl,
    )


def tmul(
    lhs_view,
    rhs_view,
    out_view,
    *,
    dtype,
    shape,
    lanes=None,
    base_addr=0,
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vmul",
        impl=impl,
    )


def tdiv(
    lhs_view,
    rhs_view,
    out_view,
    *,
    dtype,
    shape,
    lanes=None,
    base_addr=0,
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vdiv",
        impl=impl,
    )


def tor_(
    lhs_view,
    rhs_view,
    out_view,
    *,
    dtype,
    shape,
    lanes=None,
    base_addr=0,
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        micro_op_name="vor",
        impl=impl,
    )


def tmov(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    rows, cols = check_tbinop_operands(
        src_view,
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        context="TMOV",
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
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)
    vector_type = vreg_type(lanes, dtype)

    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
        mask = mask_for_chunk(dtype, active)
        index = s.const(offset)
        src_vec = pto.vlds(vector_type, src_ptr, raw(index))
        pto.vsts(src_vec, out_ptr, raw(index), mask)

    store_view(out_tile, out_view)
    return out_view


def _binary_tile_vop(
    lhs_view,
    rhs_view,
    out_view,
    *,
    dtype,
    shape,
    lanes,
    base_addr,
    micro_op_name,
    impl,
):
    rows, cols = check_tbinop_operands(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        context=micro_op_name.upper().replace("V", "T", 1),
    )
    lanes = resolve_lanes(dtype, lanes)
    element_count = rows * cols
    buf_bytes = element_count * dtype_byte_width(dtype)
    lhs_addr = const_i64(base_addr)
    rhs_addr = const_i64(base_addr + buf_bytes)
    out_addr = const_i64(base_addr + buf_bytes * 2)

    lhs_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=lhs_addr)
    rhs_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=rhs_addr)
    out_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=out_addr)
    load_view(lhs_view, lhs_tile)
    load_view(rhs_view, rhs_tile)

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    lhs_ptr = pto.castptr(ptr_type, lhs_addr)
    rhs_ptr = pto.castptr(ptr_type, rhs_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)
    micro_op = getattr(pto, micro_op_name)

    impl_kind = normalize_vf_impl_kind(impl)
    is_contiguous = rows == 1 or cols == element_count
    if const_expr(impl_kind == VF_IMPL_DEFAULT):
        impl_kind = (
            VF_IMPL_1D_POST_UPDATE if is_contiguous else VF_IMPL_2D_NO_POST_UPDATE
        )

    if const_expr(impl_kind == VF_IMPL_1D_NO_POST_UPDATE):
        _binary_1d_no_post_update(
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            dtype=dtype,
            lanes=lanes,
            element_count=element_count,
            vector_type=vector_type,
            micro_op=micro_op,
        )
    elif const_expr(impl_kind == VF_IMPL_1D_POST_UPDATE):
        _binary_1d_post_update(
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            ptr_type=ptr_type,
            dtype=dtype,
            lanes=lanes,
            element_count=element_count,
            vector_type=vector_type,
            micro_op=micro_op,
        )
    elif const_expr(impl_kind == VF_IMPL_2D_NO_POST_UPDATE):
        _binary_2d_no_post_update(
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            vector_type=vector_type,
            micro_op=micro_op,
        )
    elif const_expr(impl_kind == VF_IMPL_2D_POST_UPDATE):
        _binary_2d_post_update(
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            ptr_type=ptr_type,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            vector_type=vector_type,
            micro_op=micro_op,
        )
    else:
        raise ValueError(f"Unexpected normalized VF impl kind '{impl_kind}'.")

    store_view(out_tile, out_view)
    return out_view


def _binary_1d_no_post_update(
    lhs_ptr, rhs_ptr, out_ptr, *, dtype, lanes, element_count, vector_type, micro_op
):
    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
        mask = mask_for_chunk(dtype, active)
        index = s.const(offset)
        lhs_vec = pto.vlds(vector_type, lhs_ptr, raw(index))
        rhs_vec = pto.vlds(vector_type, rhs_ptr, raw(index))
        out_vec = micro_op(vector_type, lhs_vec, rhs_vec, mask)
        pto.vsts(out_vec, out_ptr, raw(index), mask)


def _binary_1d_post_update(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    *,
    ptr_type,
    dtype,
    lanes,
    element_count,
    vector_type,
    micro_op,
):
    lhs_cursor = lhs_ptr
    rhs_cursor = rhs_ptr
    out_cursor = out_ptr
    lane_step = s.const(lanes)
    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
        mask = mask_for_chunk(dtype, active)
        lhs_vec, lhs_cursor = pto.vlds_post(
            vector_type, ptr_type, lhs_cursor, raw(lane_step)
        )
        rhs_vec, rhs_cursor = pto.vlds_post(
            vector_type, ptr_type, rhs_cursor, raw(lane_step)
        )
        out_vec = micro_op(vector_type, lhs_vec, rhs_vec, mask)
        out_cursor = pto.vsts_post(ptr_type, out_vec, out_cursor, raw(lane_step), mask)


def _binary_2d_no_post_update(
    lhs_ptr, rhs_ptr, out_ptr, *, dtype, rows, cols, lanes, vector_type, micro_op
):
    for row in range_constexpr(rows):
        for col in range_constexpr(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = mask_for_chunk(dtype, active)
            index = s.const(row * cols + col)
            lhs_vec = pto.vlds(vector_type, lhs_ptr, raw(index))
            rhs_vec = pto.vlds(vector_type, rhs_ptr, raw(index))
            out_vec = micro_op(vector_type, lhs_vec, rhs_vec, mask)
            pto.vsts(out_vec, out_ptr, raw(index), mask)


def _binary_2d_post_update(
    lhs_ptr, rhs_ptr, out_ptr, *, ptr_type, dtype, rows, cols, lanes, vector_type, micro_op
):
    lane_step = s.const(lanes)
    for row in range_constexpr(rows):
        row_base = row * cols
        lhs_row = pto.addptr(lhs_ptr, raw(s.const(row_base)))
        rhs_row = pto.addptr(rhs_ptr, raw(s.const(row_base)))
        out_row = pto.addptr(out_ptr, raw(s.const(row_base)))
        lhs_cursor = lhs_row
        rhs_cursor = rhs_row
        out_cursor = out_row
        for col in range_constexpr(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = mask_for_chunk(dtype, active)
            lhs_vec, lhs_cursor = pto.vlds_post(
                vector_type, ptr_type, lhs_cursor, raw(lane_step)
            )
            rhs_vec, rhs_cursor = pto.vlds_post(
                vector_type, ptr_type, rhs_cursor, raw(lane_step)
            )
            out_vec = micro_op(vector_type, lhs_vec, rhs_vec, mask)
            out_cursor = pto.vsts_post(
                ptr_type, out_vec, out_cursor, raw(lane_step), mask
            )


__all__ = [
    "VF_IMPL_DEFAULT",
    "VF_IMPL_1D_NO_POST_UPDATE",
    "VF_IMPL_1D_POST_UPDATE",
    "VF_IMPL_2D_NO_POST_UPDATE",
    "VF_IMPL_2D_POST_UPDATE",
    "tadd",
    "tdiv",
    "tmov",
    "tmul",
    "tor_",
    "tsub",
]
