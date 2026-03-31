"""Implement tile scalar ops with PTO vector micro instructions.

This file demonstrates how to write tile-style helpers such as `pto.tadds`
directly in terms of `pto.vlds`, scalar/vector micro opcodes, and `pto.vsts`.
"""

from mlir.dialects import pto

from ._common import (
    VF_IMPL_1D_NO_POST_UPDATE,
    VF_IMPL_1D_POST_UPDATE,
    VF_IMPL_2D_NO_POST_UPDATE,
    VF_IMPL_2D_POST_UPDATE,
    VF_IMPL_DEFAULT,
    alloc_tile_buffer,
    check_tscalar_operands,
    const_expr,
    const_i64,
    const_scalar,
    dtype_byte_width,
    flat_active_lanes,
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
    vreg_type,
)


def tadds(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TADDS",
        impl=impl,
        emit_op=_emit_tadds,
    )


def tsubs(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TSUBS",
        impl=impl,
        emit_op=_emit_tsubs,
    )


def tmuls(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TMULS",
        impl=impl,
        emit_op=_emit_tmuls,
    )


def tdivs(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TDIVS",
        impl=impl,
        emit_op=_emit_tdivs,
    )


def tmaxs(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TMAXS",
        impl=impl,
        emit_op=_emit_tmaxs,
    )


def tmins(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TMINS",
        impl=impl,
        emit_op=_emit_tmins,
    )


def tands(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TANDS",
        impl=impl,
        emit_op=_emit_tands,
    )


def tors(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TORS",
        impl=impl,
        emit_op=_emit_tors,
    )


def txors(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TXORS",
        impl=impl,
        emit_op=_emit_txors,
    )


def tshls(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TSHLS",
        impl=impl,
        emit_op=_emit_tshls,
    )


def tshrs(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TSHRS",
        impl=impl,
        emit_op=_emit_tshrs,
    )


def tlrelu(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TLRELU",
        impl=impl,
        emit_op=_emit_tlrelu,
        allowed_dtypes={"f32", "f16"},
    )


def taxpy(
    src_view,
    scalar,
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
    return _scalar_tile_vop(
        src_view,
        scalar,
        out_view,
        dtype=dtype,
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        lanes=lanes,
        base_addr=base_addr,
        context="TAXPY",
        impl=VF_IMPL_2D_NO_POST_UPDATE,
        emit_op=_emit_taxpy,
        use_out_as_acc=True,
        allowed_dtypes={"f32", "f16", "bf16"},
    )


def _scalar_tile_vop(
    src_view,
    scalar,
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
    impl,
    emit_op,
    use_out_as_acc=False,
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
    rows, cols = check_tscalar_operands(
        src_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
        context=context,
        allowed=allowed_dtypes,
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
    if use_out_as_acc:
        load_view(out_view, out_tile)

    ptr_type = ptr(dtype, space="VEC")
    vector_type = vreg_type(lanes, dtype)
    src_ptr = pto.castptr(ptr_type, src_addr)
    out_ptr = pto.castptr(ptr_type, out_addr)
    scalar_value = raw(scalar)
    if not hasattr(scalar_value, "type"):
        scalar_value = const_scalar(dtype, scalar)
    scalar_vector = pto.vbr(vector_type, scalar_value)

    impl_kind = normalize_vf_impl_kind(impl)
    is_contiguous = rows == 1 or type_valid_shape[1] == cols
    if const_expr(impl_kind == VF_IMPL_DEFAULT):
        impl_kind = VF_IMPL_1D_POST_UPDATE if is_contiguous else VF_IMPL_2D_POST_UPDATE

    if const_expr(impl_kind == VF_IMPL_1D_NO_POST_UPDATE):
        _scalar_1d_no_post_update(
            src_ptr,
            out_ptr,
            dtype=dtype,
            lanes=lanes,
            element_count=element_count,
            valid_row=valid_row,
            valid_col=valid_col,
            vector_type=vector_type,
            scalar_value=scalar_value,
            scalar_vector=scalar_vector,
            emit_op=emit_op,
            use_out_as_acc=use_out_as_acc,
        )
    elif const_expr(impl_kind == VF_IMPL_1D_POST_UPDATE and not use_out_as_acc):
        _scalar_1d_post_update(
            src_ptr,
            out_ptr,
            ptr_type=ptr_type,
            dtype=dtype,
            lanes=lanes,
            element_count=element_count,
            valid_row=valid_row,
            valid_col=valid_col,
            vector_type=vector_type,
            scalar_value=scalar_value,
            scalar_vector=scalar_vector,
            emit_op=emit_op,
        )
    elif const_expr(
        impl_kind
        in {
            VF_IMPL_1D_POST_UPDATE,
            VF_IMPL_2D_NO_POST_UPDATE,
            VF_IMPL_2D_POST_UPDATE,
        }
    ):
        _scalar_2d_no_post_update(
            src_ptr,
            out_ptr,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            valid_row=valid_row,
            valid_col=valid_col,
            vector_type=vector_type,
            scalar_value=scalar_value,
            scalar_vector=scalar_vector,
            emit_op=emit_op,
            use_out_as_acc=use_out_as_acc,
        )
    else:
        raise ValueError(f"Unexpected normalized VF impl kind '{impl_kind}'.")

    store_view(out_tile, out_view)
    return out_view


def _emit_tadds(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, mask, scalar_vector
    return pto.vadds(vector_type, src_vec, scalar_value)


def _emit_tsubs(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, scalar_value
    return pto.vsub(vector_type, src_vec, scalar_vector, mask)


def _emit_tmuls(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, mask, scalar_vector
    return pto.vmuls(vector_type, src_vec, scalar_value)


def _emit_tdivs(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, scalar_value
    return pto.vdiv(vector_type, src_vec, scalar_vector, mask)


def _emit_tmaxs(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, mask, scalar_vector
    return pto.vmaxs(vector_type, src_vec, scalar_value)


def _emit_tmins(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, mask, scalar_vector
    return pto.vmins(vector_type, src_vec, scalar_value)


def _emit_tands(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, scalar_value
    return pto.vand(vector_type, src_vec, scalar_vector, mask)


def _emit_tors(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, scalar_value
    return pto.vor(vector_type, src_vec, scalar_vector, mask)


def _emit_txors(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, scalar_value
    return pto.vxor(vector_type, src_vec, scalar_vector, mask)


def _emit_tshls(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, mask, scalar_vector
    return pto.vshls(vector_type, src_vec, scalar_value)


def _emit_tshrs(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, mask, scalar_vector
    return pto.vshrs(vector_type, src_vec, scalar_value)


def _emit_tlrelu(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del acc_vec, mask, scalar_vector
    return pto.vlrelu(vector_type, src_vec, scalar_value)


def _emit_taxpy(*, vector_type, src_vec, acc_vec, mask, scalar_value, scalar_vector):
    del scalar_value
    return pto.vmula(vector_type, acc_vec, src_vec, scalar_vector, mask)


def _scalar_1d_no_post_update(
    src_ptr,
    out_ptr,
    *,
    dtype,
    lanes,
    element_count,
    valid_row,
    valid_col,
    vector_type,
    scalar_value,
    scalar_vector,
    emit_op,
    use_out_as_acc,
):
    for offset in range_constexpr(0, element_count, lanes):
        active = flat_active_lanes(valid_row, valid_col, offset, lanes)
        mask = mask_for_chunk(dtype, active)
        index = s.const(offset)
        src_vec = pto.vlds(vector_type, src_ptr, raw(index))
        acc_vec = (
            pto.vlds(vector_type, out_ptr, raw(index)) if use_out_as_acc else src_vec
        )
        out_vec = emit_op(
            vector_type=vector_type,
            src_vec=src_vec,
            acc_vec=acc_vec,
            mask=mask,
            scalar_value=scalar_value,
            scalar_vector=scalar_vector,
        )
        pto.vsts(out_vec, out_ptr, raw(index), mask)


def _scalar_1d_post_update(
    src_ptr,
    out_ptr,
    *,
    ptr_type,
    dtype,
    lanes,
    element_count,
    valid_row,
    valid_col,
    vector_type,
    scalar_value,
    scalar_vector,
    emit_op,
):
    cursor_in = src_ptr
    cursor_out = out_ptr
    lane_step = s.const(lanes)

    for offset in range_constexpr(0, element_count, lanes):
        active = flat_active_lanes(valid_row, valid_col, offset, lanes)
        mask = mask_for_chunk(dtype, active)
        src_vec, cursor_in = pto.vlds_post(
            vector_type, ptr_type, cursor_in, raw(lane_step)
        )
        out_vec = emit_op(
            vector_type=vector_type,
            src_vec=src_vec,
            acc_vec=src_vec,
            mask=mask,
            scalar_value=scalar_value,
            scalar_vector=scalar_vector,
        )
        cursor_out = pto.vsts_post(ptr_type, out_vec, cursor_out, raw(lane_step), mask)


def _scalar_2d_no_post_update(
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
    scalar_value,
    scalar_vector,
    emit_op,
    use_out_as_acc,
):
    for row in range_constexpr(rows):
        row_offset = row * cols
        for col in range_constexpr(0, cols, lanes):
            active = matrix_active_lanes(valid_row, valid_col, row, col, lanes)
            mask = mask_for_chunk(dtype, active)
            offset = s.const(row_offset + col)
            src_vec = pto.vlds(vector_type, src_ptr, raw(offset))
            acc_vec = (
                pto.vlds(vector_type, out_ptr, raw(offset))
                if use_out_as_acc
                else src_vec
            )
            out_vec = emit_op(
                vector_type=vector_type,
                src_vec=src_vec,
                acc_vec=acc_vec,
                mask=mask,
                scalar_value=scalar_value,
                scalar_vector=scalar_vector,
            )
            pto.vsts(out_vec, out_ptr, raw(offset), mask)


__all__ = [
    "VF_IMPL_DEFAULT",
    "VF_IMPL_1D_NO_POST_UPDATE",
    "VF_IMPL_1D_POST_UPDATE",
    "VF_IMPL_2D_NO_POST_UPDATE",
    "VF_IMPL_2D_POST_UPDATE",
    "taxpy",
    "tadds",
    "tands",
    "tdivs",
    "tlrelu",
    "tmaxs",
    "tmins",
    "tmuls",
    "tors",
    "tshls",
    "tshrs",
    "tsubs",
    "txors",
]
