"""Implement tile binary ops with PTO vector micro instructions.

This file demonstrates how to write tile-style helpers such as `pto.tadd`
directly in terms of `pto.vlds`, `pto.vadd`, and `pto.vsts`.
"""

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
    flat_active_lanes,
    mask_type,
    mask_for_chunk,
    matrix_active_lanes,
    normalize_vf_impl_kind,
    ptr,
    raw,
    range_constexpr,
    resolve_tile_spec,
    resolve_lanes,
    s,
    store_view,
    load_view,
    const_scalar,
    vreg_type,
)


def tadd(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TADD",
        micro_op=pto.vadd,
        impl=impl,
    )


def tsub(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TSUB",
        micro_op=pto.vsub,
        impl=impl,
    )


def tmul(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TMUL",
        micro_op=pto.vmul,
        impl=impl,
    )


def tdiv(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TDIV",
        micro_op=pto.vdiv,
        impl=impl,
    )


def tor_(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TOR",
        micro_op=pto.vor,
        impl=impl,
    )


def tmax(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TMAX",
        micro_op=pto.vmax,
        impl=impl,
    )


def tmin(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TMIN",
        micro_op=pto.vmin,
        impl=impl,
    )


def tand(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TAND",
        micro_op=pto.vand,
        impl=impl,
    )


def txor(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TXOR",
        micro_op=pto.vxor,
        impl=impl,
    )


def tshl(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TSHL",
        micro_op=pto.vshl,
        impl=impl,
    )


def tshr(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TSHR",
        micro_op=pto.vshr,
        impl=impl,
    )


def tmov(
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
        context="TMOV",
    )
    rows, cols = check_tbinop_operands(
        src_view,
        src_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
        context="TMOV",
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
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)
    vector_type = vreg_type(lanes, dtype)

    for offset in range_constexpr(0, element_count, lanes):
        active = flat_active_lanes(valid_row, valid_col, offset, lanes)
        mask = mask_for_chunk(dtype, active)
        index = s.const(offset)
        src_vec = pto.vlds(vector_type, src_ptr, raw(index))
        pto.vsts(src_vec, out_ptr, raw(index), mask)

    store_view(out_tile, out_view)
    return out_view


def tprelu(
    lhs_view,
    rhs_view,
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
    impl=VF_IMPL_DEFAULT,
):
    zero_scalar = const_scalar(dtype, 0)

    def emit_prelu(vector_type, lhs_vec, rhs_vec, mask):
        neg_vec = pto.vmul(vector_type, lhs_vec, rhs_vec, mask)
        cmp_mask = pto.vcmps(mask_type(), lhs_vec, zero_scalar, mask, "gt")
        return pto.vsel(vector_type, lhs_vec, neg_vec, cmp_mask)

    return _binary_tile_vop(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TPRELU",
        micro_op=emit_prelu,
        impl=impl,
        allowed_dtypes={"f32", "f16"},
    )


def _binary_tile_vop(
    lhs_view,
    rhs_view,
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
    impl,
    allowed_dtypes=None,
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
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
        context=context,
        allowed=allowed_dtypes,
    )
    lanes = resolve_lanes(dtype, lanes)
    element_count = rows * cols
    buf_bytes = element_count * dtype_byte_width(dtype)
    lhs_addr = const_i64(base_addr)
    rhs_addr = const_i64(base_addr + buf_bytes)
    out_addr = const_i64(base_addr + buf_bytes * 2)

    lhs_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=lhs_addr,
        valid_shape=type_valid_shape,
        valid_row=valid_row,
        valid_col=valid_col,
    )
    rhs_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=rhs_addr,
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
    load_view(lhs_view, lhs_tile)
    load_view(rhs_view, rhs_tile)

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    lhs_ptr = pto.castptr(ptr_type, lhs_addr)
    rhs_ptr = pto.castptr(ptr_type, rhs_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)

    impl_kind = normalize_vf_impl_kind(impl)
    is_contiguous = rows == 1 or type_valid_shape[1] == cols
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
            valid_row=valid_row,
            valid_col=valid_col,
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
            valid_row=valid_row,
            valid_col=valid_col,
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
            valid_row=valid_row,
            valid_col=valid_col,
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
            valid_row=valid_row,
            valid_col=valid_col,
            vector_type=vector_type,
            micro_op=micro_op,
        )
    else:
        raise ValueError(f"Unexpected normalized VF impl kind '{impl_kind}'.")

    store_view(out_tile, out_view)
    return out_view


def _binary_1d_no_post_update(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    *,
    dtype,
    lanes,
    element_count,
    valid_row,
    valid_col,
    vector_type,
    micro_op,
):
    for offset in range_constexpr(0, element_count, lanes):
        active = flat_active_lanes(valid_row, valid_col, offset, lanes)
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
    valid_row,
    valid_col,
    vector_type,
    micro_op,
):
    lhs_cursor = lhs_ptr
    rhs_cursor = rhs_ptr
    out_cursor = out_ptr
    lane_step = s.const(lanes)
    for offset in range_constexpr(0, element_count, lanes):
        active = flat_active_lanes(valid_row, valid_col, offset, lanes)
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
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    *,
    dtype,
    rows,
    cols,
    lanes,
    valid_row,
    valid_col,
    vector_type,
    micro_op,
):
    for row in range_constexpr(rows):
        for col in range_constexpr(0, cols, lanes):
            active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
            mask = mask_for_chunk(dtype, active)
            index = s.const(row * cols + col)
            lhs_vec = pto.vlds(vector_type, lhs_ptr, raw(index))
            rhs_vec = pto.vlds(vector_type, rhs_ptr, raw(index))
            out_vec = micro_op(vector_type, lhs_vec, rhs_vec, mask)
            pto.vsts(out_vec, out_ptr, raw(index), mask)


def _binary_2d_post_update(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    *,
    ptr_type,
    dtype,
    rows,
    cols,
    lanes,
    valid_row,
    valid_col,
    vector_type,
    micro_op,
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
            active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
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
    "tand",
    "tdiv",
    "tmax",
    "tmin",
    "tmov",
    "tmul",
    "tor_",
    "tprelu",
    "tshl",
    "tshr",
    "tsub",
    "txor",
]
