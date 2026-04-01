"""Implement tile select ops with PTO predicate and vector micro instructions."""

from mlir.dialects import pto
from ptodsl import language as dsl

from ._common import (
    alloc_tile_buffer,
    check_tsel_operands,
    check_tsels_operands,
    const_i64,
    const_scalar,
    dtype_byte_width,
    full_mask,
    mask_for_chunk,
    mask_type,
    ptr,
    raw,
    range_constexpr,
    resolve_tile_spec,
    extract_tensor_dtype_token,
    s,
    store_view,
    load_view,
    vreg_type,
)


def tsel(
    mask_view,
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
    base_addr=0,
):
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context="TSEL",
    )
    rows, cols = check_tsel_operands(
        mask_view,
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
    )
    if (
        not isinstance(valid_row, int)
        or not isinstance(valid_col, int)
        or type_valid_shape != [valid_row, valid_col]
    ):
        raise ValueError("TSEL lowering currently requires static valid shape.")

    lanes = 64
    elem_bytes = dtype_byte_width(dtype)
    data_bytes = rows * cols * elem_bytes
    mask_bytes = rows * cols

    mask_addr = const_i64(base_addr)
    lhs_addr = const_i64(base_addr + mask_bytes)
    rhs_addr = const_i64(base_addr + mask_bytes + data_bytes)
    out_addr = const_i64(base_addr + mask_bytes + data_bytes * 2)
    mask_token = extract_tensor_dtype_token(mask_view)
    mask_dtype = dsl.uint8 if mask_token == "u8" else dsl.int8

    mask_tile = alloc_tile_buffer(
        mask_dtype,
        [rows, cols],
        space="VEC",
        addr=mask_addr,
        valid_shape=[valid_row, valid_col],
    )
    lhs_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=lhs_addr,
        valid_shape=[valid_row, valid_col],
    )
    rhs_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=rhs_addr,
        valid_shape=[valid_row, valid_col],
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=out_addr,
        valid_shape=[valid_row, valid_col],
    )
    load_view(mask_view, mask_tile)
    load_view(lhs_view, lhs_tile)
    load_view(rhs_view, rhs_tile)

    mask_ptr = pto.castptr(ptr(mask_dtype, space="VEC"), mask_addr)
    lhs_ptr = pto.castptr(ptr(dtype, space="VEC"), lhs_addr)
    rhs_ptr = pto.castptr(ptr(dtype, space="VEC"), rhs_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)
    vector_type = vreg_type(lanes, dtype)
    full_mask16 = pto.pset_b16(mask_type(), "PAT_ALL")

    repeat_times = (valid_col + lanes - 1) // lanes
    paired_repeat_times = repeat_times // 2
    remain_repeat = repeat_times % 2
    repeat_idx_base = paired_repeat_times * 2

    for row in range_constexpr(valid_row):
        row_base = row * cols
        mask_base = row * cols
        for j in range_constexpr(paired_repeat_times):
            repeat_idx = j * 2
            col_offset0 = repeat_idx * lanes
            col_offset1 = col_offset0 + lanes
            mask_offset = s.const(mask_base + repeat_idx * 8)
            count0 = min(lanes, valid_col - col_offset0)
            count1 = min(lanes, valid_col - col_offset1)

            raw_mask = pto.plds(mask_type(), mask_ptr, raw(mask_offset), dist="US")
            low_mask, high_mask = pto.pintlv_b16(
                mask_type(), mask_type(), raw_mask, full_mask16
            )

            data_offset0 = s.const(row_base + col_offset0)
            lhs0 = pto.vlds(vector_type, lhs_ptr, raw(data_offset0))
            rhs0 = pto.vlds(vector_type, rhs_ptr, raw(data_offset0))
            out0 = pto.vsel(vector_type, lhs0, rhs0, low_mask)
            pto.vsts(out0, out_ptr, raw(data_offset0), mask_for_chunk(dtype, count0))

            data_offset1 = s.const(row_base + col_offset1)
            lhs1 = pto.vlds(vector_type, lhs_ptr, raw(data_offset1))
            rhs1 = pto.vlds(vector_type, rhs_ptr, raw(data_offset1))
            out1 = pto.vsel(vector_type, lhs1, rhs1, high_mask)
            pto.vsts(out1, out_ptr, raw(data_offset1), mask_for_chunk(dtype, count1))

        for j in range_constexpr(remain_repeat):
            repeat_idx = repeat_idx_base + j
            col_offset = repeat_idx * lanes
            count = max(0, valid_col - col_offset)
            mask_offset = s.const(mask_base + repeat_idx * 8)
            raw_mask = pto.plds(mask_type(), mask_ptr, raw(mask_offset), dist="US")
            unpacked_mask = pto.punpack(mask_type(), raw_mask, "LOWER")
            data_offset = s.const(row_base + col_offset)
            lhs = pto.vlds(vector_type, lhs_ptr, raw(data_offset))
            rhs = pto.vlds(vector_type, rhs_ptr, raw(data_offset))
            out = pto.vsel(vector_type, lhs, rhs, unpacked_mask)
            pto.vsts(out, out_ptr, raw(data_offset), mask_for_chunk(dtype, count))

    store_view(out_tile, out_view)
    return out_view


def tsels(
    mask_view,
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
    base_addr=0,
):
    rows, cols, valid_row, valid_col, type_valid_shape = resolve_tile_spec(
        tile_shape=tile_shape,
        shape=shape,
        valid_row=valid_row,
        valid_col=valid_col,
        valid_shape=valid_shape,
        context="TSELS",
    )
    rows, cols = check_tsels_operands(
        mask_view,
        src_view,
        out_view,
        dtype=dtype,
        shape=[rows, cols],
    )
    if (
        not isinstance(valid_row, int)
        or not isinstance(valid_col, int)
        or type_valid_shape != [valid_row, valid_col]
    ):
        raise ValueError("TSELS lowering currently requires static valid shape.")

    lanes = 256 // dtype_byte_width(dtype)
    total_elements = valid_row * valid_col
    if total_elements % lanes != 0:
        raise ValueError(
            "TSELS lowering currently requires total valid elements divisible by vector width."
        )

    elem_bytes = dtype_byte_width(dtype)
    buf_bytes = rows * cols * elem_bytes
    mask_addr = const_i64(base_addr)
    src_addr = const_i64(base_addr + buf_bytes)
    out_addr = const_i64(base_addr + buf_bytes * 2)

    mask_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=mask_addr,
        valid_shape=[valid_row, valid_col],
    )
    src_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=src_addr,
        valid_shape=[valid_row, valid_col],
    )
    out_tile = alloc_tile_buffer(
        dtype,
        [rows, cols],
        space="VEC",
        addr=out_addr,
        valid_shape=[valid_row, valid_col],
    )
    load_view(mask_view, mask_tile)
    load_view(src_view, src_tile)

    mask_ptr = pto.castptr(ptr(dtype, space="VEC"), mask_addr)
    src_ptr = pto.castptr(ptr(dtype, space="VEC"), src_addr)
    out_ptr = pto.castptr(ptr(dtype, space="VEC"), out_addr)
    vector_type = vreg_type(lanes, dtype)
    scalar_value = raw(scalar)
    if not hasattr(scalar_value, "type"):
        scalar_value = const_scalar(dtype, scalar)
    scalar_vec = pto.vdup(vector_type, scalar_value, position="POS_LOWEST")
    all_pred = full_mask(dtype)
    zero = const_scalar(dtype, 0)

    for offset in range_constexpr(0, total_elements, lanes):
        index = s.const(offset)
        mask_vec = pto.vlds(vector_type, mask_ptr, raw(index))
        src_vec = pto.vlds(vector_type, src_ptr, raw(index))
        select_mask = pto.vcmps(mask_type(), mask_vec, zero, all_pred, "ne")
        out_vec = pto.vsel(vector_type, src_vec, scalar_vec, select_mask)
        pto.vsts(out_vec, out_ptr, raw(index), all_pred)

    store_view(out_tile, out_view)
    return out_view


__all__ = ["tsel", "tsels"]
