"""Shared A5 helpers for writing tile-style kernels with PTO micro instructions."""

import builtins
import re

from mlir.dialects import arith, pto
from mlir.ir import IntegerAttr, IntegerType

from ... import const_expr, language as dsl, range_constexpr, scalar as s
from ...api.scalar import _unwrap

VF_IMPL_DEFAULT = "default"
VF_IMPL_1D_NO_POST_UPDATE = "1d_no_post_update"
VF_IMPL_1D_POST_UPDATE = "1d_post_update"
VF_IMPL_2D_NO_POST_UPDATE = "2d_no_post_update"
VF_IMPL_2D_POST_UPDATE = "2d_post_update"

_DTYPE_ALIAS_GROUPS = {
    "f32": {"f32", "float32"},
    "f16": {"f16", "float16", "half"},
    "bf16": {"bf16", "bfloat16"},
    "u32": {"u32", "ui32", "uint32"},
    "u16": {"u16", "uint16"},
    "u8": {"u8", "uint8"},
    "i32": {"i32", "int32"},
    "i16": {"i16", "int16"},
    "i8": {"i8", "int8"},
}


def _call(op, *args, **kwargs):
    return op(
        *(_unwrap(arg) for arg in args),
        **{name: _unwrap(value) for name, value in kwargs.items()},
    )


def raw(value):
    return _unwrap(value)


def _space_enum(space):
    return getattr(pto.AddressSpace, str(space).upper())


def ptr(dtype, *, space="GM"):
    return pto.PtrType.get(dtype, _space_enum(space))


def vreg_type(lanes, dtype):
    return pto.VRegType.get(lanes, dtype)


def mask_type():
    return pto.MaskType.get()


def align_type():
    return pto.AlignType.get()


def uint32_type():
    return IntegerType.get_unsigned(32)


def const_i64(value):
    i64 = IntegerType.get_signless(64)
    return arith.ConstantOp(i64, IntegerAttr.get(i64, value)).result


def const_i32(value):
    i32 = IntegerType.get_signless(32)
    return arith.ConstantOp(i32, IntegerAttr.get(i32, value)).result


def const_float(dtype, value):
    return arith.ConstantOp(dtype, value).result


def row_major_strides(shape):
    strides = [None] * len(shape)
    stride = s.const(1)
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        dim = shape[index]
        stride = stride * (s.const(dim) if isinstance(dim, int) else dim)
    return strides


def _index_value(value):
    return s.const(value) if isinstance(value, int) else value


def make_tensor(ptr_value, *, shape, dtype):
    tensor_type = dsl.TensorType(rank=len(shape), dtype=dtype)
    return dsl.as_tensor(
        tensor_type,
        ptr=_unwrap(ptr_value),
        shape=[_index_value(dim) for dim in shape],
        strides=row_major_strides(shape),
    )


def slice_tensor(source, *, offsets, sizes, dtype):
    subtensor_type = dsl.SubTensorType(shape=sizes, dtype=dtype)
    return dsl.slice_view(
        subtensor_type,
        source=_unwrap(source),
        offsets=[_index_value(offset) for offset in offsets],
        sizes=[_index_value(size) for size in sizes],
    )


def alloc_tile_buffer(
    dtype,
    shape,
    *,
    space="VEC",
    valid_shape=None,
    config=None,
    addr=None,
    valid_row=None,
    valid_col=None,
):
    tile_type = dsl.TileBufType(
        shape=shape,
        dtype=dtype,
        memory_space=space,
        valid_shape=valid_shape,
        config=config,
    )
    kwargs = {}
    if addr is not None:
        kwargs["addr"] = _unwrap(addr)
    if valid_row is not None:
        kwargs["valid_row"] = _unwrap(valid_row)
    if valid_col is not None:
        kwargs["valid_col"] = _unwrap(valid_col)
    return pto.AllocTileOp(tile_type, **kwargs).result


def load_view(source, dest):
    pto.TLoadOp(None, source, dest)
    return dest


def store_view(source, dest):
    pto.TStoreOp(None, source, dest)
    return dest


def move_tile(source, dest):
    pto.TMovOp(None, source, dest)
    return dest


def load_tile(
    view,
    tile_buffer=None,
    *,
    dtype=None,
    shape=None,
    space="VEC",
    valid_shape=None,
    config=None,
):
    if tile_buffer is None:
        if dtype is None or shape is None:
            raise ValueError(
                "`load_tile(...)` requires either `tile_buffer=` or both `dtype=` and `shape=`."
            )
        tile_buffer = alloc_tile_buffer(
            dtype,
            shape,
            space=space,
            valid_shape=valid_shape,
            config=config,
        )
    load_view(view, tile_buffer)
    return tile_buffer


def store_tile(tile_buffer, view):
    store_view(tile_buffer, view)
    return view


def dtype_token(dtype):
    text = str(dtype).lower()
    for canonical, aliases in _DTYPE_ALIAS_GROUPS.items():
        if any(alias in text for alias in aliases):
            return canonical
    raise ValueError(f"Unsupported dtype token for '{dtype}'.")


def dtype_byte_width(dtype):
    token = dtype_token(dtype)
    if token in {"f32", "i32", "u32"}:
        return 4
    if token in {"f16", "bf16", "i16", "u16"}:
        return 2
    if token in {"i8", "u8"}:
        return 1
    raise ValueError(f"Unsupported dtype byte width for '{dtype}'.")


def micro_lane_count(dtype):
    return 256 // dtype_byte_width(dtype)


def resolve_lanes(dtype, lanes):
    return micro_lane_count(dtype) if lanes is None else lanes


def extract_static_tensor_shape(value):
    raw = _unwrap(value)
    type_obj = getattr(raw, "type", None)
    if type_obj is None:
        return None
    text = str(type_obj)
    match = re.search(
        r"!pto\.(?:partition_)?tensor_view<(?P<payload>[^>]+)>|!pto\.tile_buf<[^,]+,\s*(?P<tile_payload>[^>]+)>",
        text,
    )
    if not match:
        return None
    payload = match.group("payload") or match.group("tile_payload")
    dims = re.findall(r"(\?|\d+)x", payload)
    if not dims:
        return None
    shape = []
    for dim in dims:
        if dim == "?":
            return None
        shape.append(int(dim))
    return shape


def extract_tensor_dtype_token(value):
    raw = _unwrap(value)
    type_obj = getattr(raw, "type", None)
    if type_obj is None:
        return None
    text = str(type_obj).lower()
    for canonical, aliases in _DTYPE_ALIAS_GROUPS.items():
        if any(alias in text for alias in aliases):
            return canonical
    return None


def require_supported_dtype(dtype, *, allowed, message):
    try:
        token = dtype_token(dtype)
    except ValueError as exc:
        raise ValueError(message) from exc
    if token not in allowed:
        raise ValueError(message)
    return token


def require_view_shape(view, expected_shape, *, message):
    actual_shape = extract_static_tensor_shape(view)
    if actual_shape is None:
        return
    if list(actual_shape) != list(expected_shape):
        raise ValueError(f"{message} Expected {expected_shape}, got {actual_shape}.")


def require_view_dtype(view, dtype, *, message):
    actual_token = extract_tensor_dtype_token(view)
    if actual_token is None:
        return
    if actual_token != dtype_token(dtype):
        raise ValueError(message)


def require_static_matrix_shape(shape, *, context):
    if len(shape) != 2 or any(not isinstance(dim, int) for dim in shape):
        raise ValueError(f"{context} currently requires a static rank-2 integer shape.")
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{context} requires positive row/column sizes.")
    return rows, cols


def full_mask(dtype):
    width = dtype_byte_width(dtype)
    if width == 4:
        return pto.pset_b32(mask_type(), "PAT_ALL")
    if width == 2:
        return pto.pset_b16(mask_type(), "PAT_ALL")
    if width == 1:
        return pto.pset_b8(mask_type(), "PAT_ALL")
    raise ValueError(f"Unsupported dtype mask width for '{dtype}'.")


def tail_mask(dtype, active_lanes):
    i32 = IntegerType.get_signless(32)
    active = const_i32(active_lanes)
    width = dtype_byte_width(dtype)
    if width == 4:
        mask, _ = pto.plt_b32(mask_type(), i32, active)
        return mask
    if width == 2:
        mask, _ = pto.plt_b16(mask_type(), i32, active)
        return mask
    if width == 1:
        mask, _ = pto.plt_b8(mask_type(), i32, active)
        return mask
    raise ValueError(f"Unsupported dtype tail mask width for '{dtype}'.")


def mask_for_chunk(dtype, active_lanes):
    lanes = micro_lane_count(dtype)
    if active_lanes == lanes:
        return full_mask(dtype)
    return tail_mask(dtype, active_lanes)


def onept_dist(dtype):
    width = dtype_byte_width(dtype)
    if width == 4:
        return "ONEPT_B32"
    if width == 2:
        return "ONEPT_B16"
    if width == 1:
        return "ONEPT_B8"
    raise ValueError(f"Unsupported dtype point-store width for '{dtype}'.")


def normalize_vf_impl_kind(impl):
    if impl is None:
        return VF_IMPL_DEFAULT

    normalized = str(impl).strip().lower()
    aliases = {
        "default": VF_IMPL_DEFAULT,
        "vfimpl_default": VF_IMPL_DEFAULT,
        "1d_no_post_update": VF_IMPL_1D_NO_POST_UPDATE,
        "vfimpl_1d_no_post_update": VF_IMPL_1D_NO_POST_UPDATE,
        "1d_post_update": VF_IMPL_1D_POST_UPDATE,
        "vfimpl_1d_post_update": VF_IMPL_1D_POST_UPDATE,
        "2d_no_post_update": VF_IMPL_2D_NO_POST_UPDATE,
        "vfimpl_2d_no_post_update": VF_IMPL_2D_NO_POST_UPDATE,
        "2d_post_update": VF_IMPL_2D_POST_UPDATE,
        "vfimpl_2d_post_update": VF_IMPL_2D_POST_UPDATE,
    }
    if normalized not in aliases:
        supported = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported VF impl kind '{impl}'. Expected one of: {supported}."
        )
    return aliases[normalized]


def check_tbinop_operands(lhs_view, rhs_view, out_view, *, dtype, shape, context):
    rows, cols = require_static_matrix_shape(shape, context=context)
    require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "bf16", "i32", "u32", "i16", "u16", "i8", "u8"},
        message=f"Fix: {context} has invalid data type.",
    )
    for view, label in ((lhs_view, "src0"), (rhs_view, "src1"), (out_view, "dst")):
        require_view_shape(
            view,
            [rows, cols],
            message=f"Fix: {context} input tile {label} valid shape mismatch.",
        )
        require_view_dtype(
            view,
            dtype,
            message=f"Fix: {context} input tile src0, src1 and dst tile data type mismatch.",
        )
    return rows, cols


def check_row_expand_operands(src_view, out_view, *, dtype, shape, context):
    rows, cols = require_static_matrix_shape(shape, context=context)
    require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "bf16", "i32", "u32", "i16", "u16", "i8", "u8"},
        message=f"Fix: {context} input data type is not supported.",
    )
    require_view_shape(
        src_view,
        [rows, 1],
        message=f"Fix: {context} source valid shape must be [rows, 1].",
    )
    require_view_shape(
        out_view,
        [rows, cols],
        message=f"Fix: {context} destination valid shape mismatch.",
    )
    require_view_dtype(
        src_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    require_view_dtype(
        out_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    return rows, cols


def check_col_expand_operands(src_view, out_view, *, dtype, shape, context):
    rows, cols = require_static_matrix_shape(shape, context=context)
    require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "bf16", "i32", "u32", "i16", "u16", "i8", "u8"},
        message=f"Fix: {context} input data type is not supported.",
    )
    require_view_shape(
        src_view,
        [1, cols],
        message=f"Fix: {context} source valid shape must be [1, cols].",
    )
    require_view_shape(
        out_view,
        [rows, cols],
        message=f"Fix: {context} destination valid shape mismatch.",
    )
    require_view_dtype(
        src_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    require_view_dtype(
        out_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    return rows, cols


def check_row_reduce_operands(src_view, out_view, *, dtype, shape, context):
    rows, cols = require_static_matrix_shape(shape, context=context)
    require_supported_dtype(
        dtype,
        allowed={"f32", "f16"},
        message=f"Fix: {context} input data type is not supported.",
    )
    require_view_shape(
        src_view,
        [rows, cols],
        message=f"Fix: {context} source valid shape mismatch.",
    )
    require_view_shape(
        out_view,
        [rows, 1],
        message=f"Fix: {context} use a single-column output tile.",
    )
    require_view_dtype(
        src_view,
        dtype,
        message=f"Fix: {context} input and output data type mismatch.",
    )
    require_view_dtype(
        out_view,
        dtype,
        message=f"Fix: {context} input and output data type mismatch.",
    )
    return rows, cols


def check_col_reduce_operands(src_view, out_view, *, dtype, shape, context):
    rows, cols = require_static_matrix_shape(shape, context=context)
    require_supported_dtype(
        dtype,
        allowed={"f32", "f16"},
        message=f"Fix: {context} input data type is not supported.",
    )
    require_view_shape(
        src_view,
        [rows, cols],
        message=f"Fix: {context} source valid shape mismatch.",
    )
    require_view_shape(
        out_view,
        [1, cols],
        message=f"Fix: {context} use a single-row output tile.",
    )
    require_view_dtype(
        src_view,
        dtype,
        message=f"Fix: {context} input and output data type mismatch.",
    )
    require_view_dtype(
        out_view,
        dtype,
        message=f"Fix: {context} input and output data type mismatch.",
    )
    return rows, cols


def check_gather_operands(
    src_view, indices_view, out_view, *, dtype, index_dtype, shape
):
    rows, cols = require_static_matrix_shape(shape, context="TGATHER")
    require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "bf16", "i32", "u32", "i16", "u16", "i8", "u8"},
        message="Fix: TGATHER source data type is not supported.",
    )
    require_supported_dtype(
        index_dtype,
        allowed={"u32"},
        message="Fix: TGATHER index data type must be uint32.",
    )
    for view, label, view_dtype in (
        (src_view, "src", dtype),
        (indices_view, "indices", index_dtype),
        (out_view, "dst", dtype),
    ):
        require_view_shape(
            view,
            [rows, cols],
            message=f"Fix: TGATHER {label} valid shape mismatch.",
        )
        require_view_dtype(
            view,
            view_dtype,
            message=f"Fix: TGATHER {label} data type mismatch.",
        )
    return rows, cols


def check_mrgsort_operands(src_view, out_view, *, dtype, shape, block_len):
    rows, cols = require_static_matrix_shape(shape, context="TMRGSORT")
    if rows != 1:
        raise ValueError(
            "TMRGSORT micro lowering currently requires a single input row."
        )
    if cols != block_len * 4:
        raise ValueError(
            "TMRGSORT micro lowering currently requires shape[1] == block_len * 4."
        )
    require_view_shape(
        src_view,
        [rows, cols],
        message="Fix: TMRGSORT source valid shape mismatch.",
    )
    require_view_shape(
        out_view,
        [rows, cols],
        message="Fix: TMRGSORT destination valid shape mismatch.",
    )
    require_view_dtype(
        src_view,
        dtype,
        message="Fix: TMRGSORT input and output data type mismatch.",
    )
    require_view_dtype(
        out_view,
        dtype,
        message="Fix: TMRGSORT input and output data type mismatch.",
    )
    return rows, cols


def check_sort32_operands(src_view, idx_view, out_view, *, dtype, shape):
    rows, cols = require_static_matrix_shape(shape, context="TSORT32")
    out_cols = cols * 4 if dtype_token(dtype) == "f16" else cols * 2
    for view, label, expected_shape in (
        (src_view, "src", [rows, cols]),
        (idx_view, "idx", [rows, cols]),
        (out_view, "dst", [rows, out_cols]),
    ):
        require_view_shape(
            view,
            expected_shape,
            message=f"TSORT32 {label} shape mismatch.",
        )
    require_view_dtype(src_view, dtype, message="Dst and src mube be same.")
    require_view_dtype(out_view, dtype, message="Dst and src mube be same.")
    require_view_dtype(idx_view, uint32_type(), message="Idx must be uint32_t.")
    if cols % 32 != 0:
        raise ValueError(
            "TSORT32 micro lowering currently requires column count divisible by 32."
        )
    return rows, cols, out_cols


__all__ = [
    "VF_IMPL_DEFAULT",
    "VF_IMPL_1D_NO_POST_UPDATE",
    "VF_IMPL_1D_POST_UPDATE",
    "VF_IMPL_2D_NO_POST_UPDATE",
    "VF_IMPL_2D_POST_UPDATE",
    "_call",
    "align_type",
    "alloc_tile_buffer",
    "check_col_expand_operands",
    "check_col_reduce_operands",
    "check_gather_operands",
    "check_mrgsort_operands",
    "check_row_expand_operands",
    "check_row_reduce_operands",
    "check_sort32_operands",
    "check_tbinop_operands",
    "const_expr",
    "const_float",
    "const_i32",
    "const_i64",
    "dtype_byte_width",
    "dtype_token",
    "full_mask",
    "load_tile",
    "load_view",
    "make_tensor",
    "mask_for_chunk",
    "mask_type",
    "micro_lane_count",
    "move_tile",
    "normalize_vf_impl_kind",
    "onept_dist",
    "ptr",
    "range_constexpr",
    "resolve_lanes",
    "s",
    "slice_tensor",
    "store_tile",
    "store_view",
    "tail_mask",
    "uint32_type",
    "vreg_type",
]
