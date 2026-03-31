"""Implement tile reduce ops with PTO vector micro instructions."""

import builtins

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
    micro_lane_count,
    normalize_vf_impl_kind,
    onept_dist,
    ptr,
    raw,
    range_constexpr,
    s,
    store_view,
    tail_mask,
    load_view,
    vreg_type,
)


def trow_sum(src_view, out_view, *, dtype, shape, base_addr=0):
    return _trow_reduce(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vcadd",
        combine_op_name="vadd",
        init_value=0.0,
    )


def trow_max(src_view, out_view, *, dtype, shape, base_addr=0):
    return _trow_reduce(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vcmax",
        combine_op_name="vmax",
        init_value=float("-inf"),
    )


def trow_min(src_view, out_view, *, dtype, shape, base_addr=0):
    return _trow_reduce(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vcmin",
        combine_op_name="vmin",
        init_value=float("inf"),
    )


def tcol_sum(src_view, out_view, *, dtype, shape, base_addr=0, impl=VF_IMPL_DEFAULT):
    return _tcol_reduce(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        micro_op_name="vadd",
        impl=impl,
    )


def tcol_max(src_view, out_view, *, dtype, shape, base_addr=0, impl=VF_IMPL_DEFAULT):
    return _tcol_reduce(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        micro_op_name="vmax",
        impl=impl,
    )


def tcol_min(src_view, out_view, *, dtype, shape, base_addr=0, impl=VF_IMPL_DEFAULT):
    return _tcol_reduce(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        micro_op_name="vmin",
        impl=impl,
    )


def _trow_reduce(
    src_view,
    out_view,
    *,
    dtype,
    shape,
    base_addr,
    reduce_op_name,
    combine_op_name,
    init_value,
):
    rows, cols = check_row_reduce_operands(
        src_view, out_view, dtype=dtype, shape=shape, context="TROWREDUCE"
    )
    width = dtype_byte_width(dtype)
    if width not in {2, 4}:
        raise ValueError(f"{reduce_op_name} currently supports only float16/float32.")

    lanes = micro_lane_count(dtype)
    vector_type = vreg_type(lanes, dtype)
    buf_bytes = rows * cols * width
    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)

    src_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=src_addr)
    out_tile = alloc_tile_buffer(
        dtype, shape, space="VEC", valid_shape=[rows, 1], addr=out_addr
    )
    load_view(src_view, src_tile)

    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)
    reduce_op = getattr(pto, reduce_op_name)
    combine_op = getattr(pto, combine_op_name)
    row_mask = full_mask(dtype)
    point_mask = tail_mask(dtype, 1)
    init_scalar = const_float(dtype, init_value)

    for row in range_constexpr(rows):
        accum = pto.vbr(vector_type, init_scalar)
        for col in range_constexpr(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = mask_for_chunk(dtype, active)
            offset = s.const(row * cols + col)
            vec = pto.vlds(vector_type, src_ptr, raw(offset))
            reduced = reduce_op(vector_type, vec, mask)
            accum = combine_op(vector_type, accum, reduced, row_mask)
        out_offset = s.const(row * cols)
        pto.vsts(accum, out_ptr, raw(out_offset), point_mask, dist=onept_dist(dtype))

    store_view(out_tile, out_view)
    return out_view


def _tcol_reduce(
    src_view,
    out_view,
    *,
    dtype,
    shape,
    base_addr,
    micro_op_name,
    impl,
):
    rows, cols = check_col_reduce_operands(
        src_view, out_view, dtype=dtype, shape=shape, context="TCOLREDUCE"
    )
    lanes = micro_lane_count(dtype)
    buf_bytes = rows * cols * dtype_byte_width(dtype)
    src_addr = const_i64(base_addr)
    out_addr = const_i64(base_addr + buf_bytes)

    src_tile = alloc_tile_buffer(dtype, shape, space="VEC", addr=src_addr)
    out_tile = alloc_tile_buffer(
        dtype, [1, cols], space="VEC", valid_shape=[1, cols], addr=out_addr
    )
    load_view(src_view, src_tile)

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)
    reduce_op = getattr(pto, micro_op_name)
    impl_kind = normalize_vf_impl_kind(impl)
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
            vector_type=vector_type,
            reduce_op=reduce_op,
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
            vector_type=vector_type,
            reduce_op=reduce_op,
        )
    else:
        raise ValueError(f"Unexpected normalized VF impl kind '{impl_kind}'.")

    store_view(out_tile, out_view)
    return out_view


def _tcol_reduce_no_post_update(
    src_ptr, out_ptr, *, dtype, rows, cols, lanes, vector_type, reduce_op
):
    loop_pairs = (rows - 1) // 2
    remain = (rows - 1) % 2
    for col in range_constexpr(0, cols, lanes):
        active = builtins.min(lanes, cols - col)
        mask = mask_for_chunk(dtype, active)
        accum = pto.vlds(vector_type, src_ptr, raw(s.const(col)))
        for pair in range_constexpr(loop_pairs):
            row0 = 2 * pair + 1
            row1 = 2 * pair + 2
            src0 = pto.vlds(vector_type, src_ptr, raw(s.const(col + row0 * cols)))
            src1 = pto.vlds(vector_type, src_ptr, raw(s.const(col + row1 * cols)))
            pair_sum = reduce_op(vector_type, src0, src1, mask)
            accum = reduce_op(vector_type, accum, pair_sum, mask)
        if const_expr(remain):
            src_tail = pto.vlds(
                vector_type, src_ptr, raw(s.const(col + (rows - 1) * cols))
            )
            accum = reduce_op(vector_type, accum, src_tail, mask)
        pto.vsts(accum, out_ptr, raw(s.const(col)), mask)


def _tcol_reduce_post_update(
    src_ptr, out_ptr, *, ptr_type, dtype, rows, cols, lanes, vector_type, reduce_op
):
    lane_step = s.const(lanes)
    for col in range_constexpr(0, cols, lanes):
        active = builtins.min(lanes, cols - col)
        mask = mask_for_chunk(dtype, active)
        row0_ptr = pto.addptr(src_ptr, raw(s.const(col)))
        accum, _ = pto.vlds_post(vector_type, ptr_type, row0_ptr, raw(lane_step))
        row_ptr = pto.addptr(row0_ptr, raw(s.const(cols)))
        for _ in range_constexpr(rows - 1):
            src_tail, row_ptr = pto.vlds_post(
                vector_type, ptr_type, row_ptr, raw(lane_step)
            )
            accum = reduce_op(vector_type, accum, src_tail, mask)
        out_cursor = pto.addptr(out_ptr, raw(s.const(col)))
        pto.vsts_post(ptr_type, accum, out_cursor, raw(lane_step), mask)


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
