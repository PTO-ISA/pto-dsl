"""Implement tile reduce ops with PTO vector micro instructions.

This file demonstrates how to write tile-style helpers such as
`pto.trow_sum` and `pto.tcol_sum` directly in terms of PTO vector
micro instructions while keeping compile-time tile capacity separate
from runtime valid bounds.
"""

from mlir.dialects import pto

from ._common import (
    VF_IMPL_1D_NO_POST_UPDATE,
    VF_IMPL_1D_POST_UPDATE,
    VF_IMPL_2D_NO_POST_UPDATE,
    VF_IMPL_2D_POST_UPDATE,
    VF_IMPL_DEFAULT,
    alloc_tile_buffer,
    check_col_reduce_operands,
    check_row_reduce_operands,
    const_expr,
    const_float,
    const_i64,
    dtype_byte_width,
    full_mask,
    mask_for_chunk,
    matrix_active_lanes,
    micro_lane_count,
    normalize_vf_impl_kind,
    onept_dist,
    ptr,
    raw,
    range_constexpr,
    resolve_tile_spec,
    s,
    load_view,
    store_view,
    vreg_type,
)


def trow_sum(
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
    return _trow_reduce(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWSUM",
        reduce_op=pto.vcadd,
        combine_op=pto.vadd,
        init_value=0.0,
    )


def trow_max(
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
    return _trow_reduce(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWMAX",
        reduce_op=pto.vcmax,
        combine_op=pto.vmax,
        init_value=float("-inf"),
    )


def trow_min(
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
    return _trow_reduce(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TROWMIN",
        reduce_op=pto.vcmin,
        combine_op=pto.vmin,
        init_value=float("inf"),
    )


def tcol_sum(
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
    impl=VF_IMPL_DEFAULT,
):
    return _tcol_reduce(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLSUM",
        reduce_op=pto.vadd,
        init_value=0.0,
        impl=impl,
    )


def tcol_max(
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
    impl=VF_IMPL_DEFAULT,
):
    return _tcol_reduce(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLMAX",
        reduce_op=pto.vmax,
        init_value=float("-inf"),
        impl=impl,
    )


def tcol_min(
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
    impl=VF_IMPL_DEFAULT,
):
    return _tcol_reduce(
        src_view,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        base_addr=base_addr,
        context="TCOLMIN",
        reduce_op=pto.vmin,
        init_value=float("inf"),
        impl=impl,
    )


def _trow_reduce(
    src_view,
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
    reduce_op,
    combine_op,
    init_value,
):
    validation_context = "TROWREDUCE"
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context=validation_context,
    )
    rows, cols = check_row_reduce_operands(
        src_view, out_view, dtype=dtype, shape=[rows, cols], context=validation_context
    )
    width = dtype_byte_width(dtype)
    if width not in {2, 4}:
        raise ValueError(
            f"{validation_context} currently supports only float16/float32."
        )

    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * width
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
        valid_shape=[type_valid_shape[0], 1],
        addr=out_addr,
        valid_row=valid_row,
        valid_col=1,
    )
    load_view(src_view, src_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)
    vector_mask = full_mask(dtype)
    init_scalar = const_float(dtype, init_value)
    neutral_vec = pto.vbr(vector_type, init_scalar)

    for row in range_constexpr(rows):
        accum = neutral_vec
        for col in range_constexpr(0, cols, lanes):
            active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
            mask = mask_for_chunk(dtype, active)
            offset = s.const(row * cols + col)
            vec = pto.vlds(vector_type, src_ptr, raw(offset))
            masked_vec = pto.vsel(vector_type, vec, neutral_vec, mask)
            reduced = reduce_op(vector_type, masked_vec, vector_mask)
            accum = combine_op(vector_type, accum, reduced, vector_mask)
        out_offset = s.const(row * cols)
        store_mask = mask_for_chunk(dtype, matrix_active_lanes(valid_row, 1, row, 0, 1))
        pto.vsts(accum, out_ptr, raw(out_offset), store_mask, dist=onept_dist(dtype))

    store_view(out_tile, out_view)
    return out_view


def _tcol_reduce(
    src_view,
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
    reduce_op,
    init_value,
    impl,
):
    validation_context = "TCOLREDUCE"
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context=validation_context,
    )
    rows, cols = check_col_reduce_operands(
        src_view, out_view, dtype=dtype, shape=[rows, cols], context=validation_context
    )
    lanes = micro_lane_count(dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)
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
        valid_shape=[1, type_valid_shape[1]],
        addr=out_addr,
        valid_row=1,
        valid_col=valid_col,
    )
    load_view(src_view, src_tile)

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)
    impl_kind = normalize_vf_impl_kind(impl)
    init_scalar = const_float(dtype, init_value)
    neutral_vec = pto.vbr(vector_type, init_scalar)
    vector_mask = full_mask(dtype)
    if const_expr(impl_kind == VF_IMPL_DEFAULT):
        impl_kind = VF_IMPL_1D_POST_UPDATE

    if const_expr(impl_kind in {VF_IMPL_1D_NO_POST_UPDATE, VF_IMPL_2D_NO_POST_UPDATE}):
        _tcol_reduce_no_post_update(
            src_ptr,
            out_ptr,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            valid_row=valid_row,
            valid_col=valid_col,
            vector_type=vector_type,
            reduce_op=reduce_op,
            neutral_vec=neutral_vec,
            vector_mask=vector_mask,
        )
    elif const_expr(impl_kind in {VF_IMPL_1D_POST_UPDATE, VF_IMPL_2D_POST_UPDATE}):
        _tcol_reduce_post_update(
            src_ptr,
            out_ptr,
            ptr_type=ptr_type,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            valid_row=valid_row,
            valid_col=valid_col,
            vector_type=vector_type,
            reduce_op=reduce_op,
            neutral_vec=neutral_vec,
            vector_mask=vector_mask,
        )
    else:
        raise ValueError(f"Unexpected normalized VF impl kind '{impl_kind}'.")

    store_view(out_tile, out_view)
    return out_view


def _tcol_reduce_no_post_update(
    src_ptr,
    out_ptr,
    *,
    dtype,
    rows,
    cols,
    lanes,
    valid_row,
    valid_col,
    vector_type,
    reduce_op,
    neutral_vec,
    vector_mask,
):
    for col in range_constexpr(0, cols, lanes):
        col_mask = mask_for_chunk(
            dtype, matrix_active_lanes(1, valid_col, 0, col, lanes)
        )
        accum = neutral_vec
        for row in range_constexpr(rows):
            row_mask = mask_for_chunk(
                dtype, matrix_active_lanes(valid_row, lanes, row, 0, lanes)
            )
            src = pto.vlds(vector_type, src_ptr, raw(s.const(col + row * cols)))
            row_filtered = pto.vsel(vector_type, src, neutral_vec, row_mask)
            contrib = pto.vsel(vector_type, row_filtered, neutral_vec, col_mask)
            accum = reduce_op(vector_type, accum, contrib, vector_mask)
        pto.vsts(accum, out_ptr, raw(s.const(col)), col_mask)


def _tcol_reduce_post_update(
    src_ptr,
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
    reduce_op,
    neutral_vec,
    vector_mask,
):
    lane_step = s.const(lanes)
    for col in range_constexpr(0, cols, lanes):
        col_mask = mask_for_chunk(
            dtype, matrix_active_lanes(1, valid_col, 0, col, lanes)
        )
        row0_ptr = pto.addptr(src_ptr, raw(s.const(col)))
        row_ptr = row0_ptr
        accum = neutral_vec
        for row in range_constexpr(rows):
            src_row, row_ptr = pto.vlds_post(
                vector_type, ptr_type, row_ptr, raw(s.const(cols))
            )
            row_mask = mask_for_chunk(
                dtype, matrix_active_lanes(valid_row, lanes, row, 0, lanes)
            )
            row_filtered = pto.vsel(vector_type, src_row, neutral_vec, row_mask)
            contrib = pto.vsel(vector_type, row_filtered, neutral_vec, col_mask)
            accum = reduce_op(vector_type, accum, contrib, vector_mask)
        out_cursor = pto.addptr(out_ptr, raw(s.const(col)))
        pto.vsts_post(ptr_type, accum, out_cursor, raw(lane_step), col_mask)


__all__ = [
    "VF_IMPL_DEFAULT",
    "VF_IMPL_1D_NO_POST_UPDATE",
    "VF_IMPL_1D_POST_UPDATE",
    "VF_IMPL_2D_NO_POST_UPDATE",
    "VF_IMPL_2D_POST_UPDATE",
    "tcol_max",
    "tcol_min",
    "tcol_sum",
    "trow_max",
    "trow_min",
    "trow_sum",
]
