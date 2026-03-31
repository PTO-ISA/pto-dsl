import builtins
import re

from mlir.dialects import arith as _arith
from mlir.dialects import pto as _pto
from mlir.ir import IntegerAttr, IntegerType

from ... import pto as _dsl_pto
from ... import scalar as _scalar
from ... import const_expr, range_constexpr
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
    "i32": {"i32", "int32"},
    "u32": {"u32", "uint32"},
    "i16": {"i16", "int16"},
    "u16": {"u16", "uint16"},
    "i8": {"i8", "int8"},
    "u8": {"u8", "uint8"},
}


def _call(op, *args, **kwargs):
    return op(
        *(_unwrap(arg) for arg in args),
        **{name: _unwrap(value) for name, value in kwargs.items()},
    )


def _cmp_mode_attr(mode):
    if mode is None:
        return None
    if isinstance(mode, str):
        return _pto.CmpModeAttr.get(getattr(_pto.CmpMode, mode.upper()))
    return mode


def _const_i64(value):
    i64 = IntegerType.get_signless(64)
    return _arith.ConstantOp(i64, IntegerAttr.get(i64, value)).result


def _const_i32(value):
    i32 = IntegerType.get_signless(32)
    return _arith.ConstantOp(i32, IntegerAttr.get(i32, value)).result


def _const_float(dtype, value):
    return _arith.ConstantOp(_scalar.resolve_type(dtype), value).result


def _dtype_token(dtype):
    text = str(_scalar.resolve_type(dtype)).lower()
    for canonical, aliases in _DTYPE_ALIAS_GROUPS.items():
        if any(alias in text for alias in aliases):
            return canonical
    raise ValueError(f"Unsupported dtype token for '{dtype}'.")


def _dtype_byte_width(dtype):
    text = str(dtype)
    if (
        "float32" in text
        or "f32" in text
        or "int32" in text
        or "i32" in text
        or "uint32" in text
        or "u32" in text
    ):
        return 4
    if (
        "float16" in text
        or "f16" in text
        or "bfloat16" in text
        or "bf16" in text
        or "int16" in text
        or "i16" in text
        or "u16" in text
    ):
        return 2
    if "i8" in text or "u8" in text:
        return 1
    raise ValueError(f"Unsupported dtype byte width for '{dtype}'.")


def _extract_static_tensor_shape(value):
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


def _extract_tensor_dtype_token(value):
    raw = _unwrap(value)
    type_obj = getattr(raw, "type", None)
    if type_obj is None:
        return None
    text = str(type_obj).lower()
    for canonical, aliases in _DTYPE_ALIAS_GROUPS.items():
        if any(alias in text for alias in aliases):
            return canonical
    return None


def _require_supported_dtype(dtype, *, allowed, message):
    try:
        token = _dtype_token(dtype)
    except ValueError as exc:
        raise ValueError(message) from exc
    if token not in allowed:
        raise ValueError(message)
    return token


def _require_view_shape(view, expected_shape, *, context, message):
    actual_shape = _extract_static_tensor_shape(view)
    if actual_shape is None:
        return
    if list(actual_shape) != list(expected_shape):
        raise ValueError(f"{message} Expected {expected_shape}, got {actual_shape}.")


def _require_view_dtype(view, dtype, *, message):
    actual_token = _extract_tensor_dtype_token(view)
    if actual_token is None:
        return
    if actual_token != _dtype_token(dtype):
        raise ValueError(message)


def _micro_lane_count(dtype):
    return 256 // _dtype_byte_width(dtype)


def _resolve_lanes(dtype, lanes):
    if lanes is None:
        return _micro_lane_count(dtype)
    return lanes


def _full_mask(dtype):
    width = _dtype_byte_width(dtype)
    if width == 4:
        return _dsl_pto.pset_b32(_dsl_pto.MaskType(), "PAT_ALL")
    if width == 2:
        return _dsl_pto.pset_b16(_dsl_pto.MaskType(), "PAT_ALL")
    if width == 1:
        return _dsl_pto.pset_b8(_dsl_pto.MaskType(), "PAT_ALL")
    raise ValueError(f"Unsupported dtype mask width for '{dtype}'.")


def _tail_mask(dtype, active_lanes):
    i32 = IntegerType.get_signless(32)
    width = _dtype_byte_width(dtype)
    active = _const_i32(active_lanes)
    if width == 4:
        mask, _ = _dsl_pto.plt_b32(_dsl_pto.MaskType(), i32, active)
        return mask
    if width == 2:
        mask, _ = _dsl_pto.plt_b16(_dsl_pto.MaskType(), i32, active)
        return mask
    if width == 1:
        mask, _ = _dsl_pto.plt_b8(_dsl_pto.MaskType(), i32, active)
        return mask
    raise ValueError(f"Unsupported dtype tail mask width for '{dtype}'.")


def _mask_for_chunk(dtype, active_lanes):
    lanes = _micro_lane_count(dtype)
    if active_lanes == lanes:
        return _full_mask(dtype)
    return _tail_mask(dtype, active_lanes)


def _onept_dist(dtype):
    width = _dtype_byte_width(dtype)
    if width == 4:
        return "ONEPT_B32"
    if width == 2:
        return "ONEPT_B16"
    if width == 1:
        return "ONEPT_B8"
    raise ValueError(f"Unsupported dtype point-store width for '{dtype}'.")


def _normalize_vf_impl_kind(impl):
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


def _alloc_like_view(view, *, dtype, shape, space, valid_shape=None, config=None):
    return _dsl_pto.make_tile_buffer(
        dtype,
        shape,
        space=space,
        valid_shape=valid_shape,
        config=config,
    ).alloc()


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
        tile_buffer = _alloc_like_view(
            view,
            dtype=dtype,
            shape=shape,
            space=space,
            valid_shape=valid_shape,
            config=config,
        )
    _dsl_pto.load(view, tile_buffer)
    return tile_buffer


def store_tile(tile_buffer, view):
    _dsl_pto.store(tile_buffer, view)
    return view


def move_tile(source, dest):
    _call(_pto.TMovOp, None, source, dest)
    return dest


def add(lhs, rhs, out):
    _call(_pto.TAddOp, lhs, rhs, out)
    return out


def add_micro(
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
    return _binary_micro(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vadd",
        impl=impl,
    )


def sub_micro(
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
    return _binary_micro(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vsub",
        impl=impl,
    )


def mul_micro(
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
    return _binary_micro(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vmul",
        impl=impl,
    )


def div_micro(
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
    return _binary_micro(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vdiv",
        impl=impl,
    )


def or_micro(
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
    return _binary_micro(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vor",
        impl=impl,
    )


def mov_micro(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name=None,
    )


def exp_micro(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vexp",
    )


def log_micro(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vln",
    )


def relu_micro(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vrelu",
    )


def abs_micro(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vabs",
    )


def sqrt_micro(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vsqrt",
    )


def rsqrt_micro(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _rsqrt_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
    )


def reciprocal_micro(src_view, out_view, *, dtype, shape, lanes=None, base_addr=0):
    return _unary_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        lanes=lanes,
        base_addr=base_addr,
        op_name="vrec",
    )


def gather_micro(
    src_view,
    indices_view,
    out_view,
    *,
    dtype,
    index_dtype,
    shape,
    base_addr=0,
):
    return _gather_micro(
        src_view,
        indices_view,
        out_view,
        dtype=dtype,
        index_dtype=index_dtype,
        shape=shape,
        base_addr=base_addr,
    )


def col_expand_micro(src_view, out_view, *, dtype, shape, base_addr=0):
    rows, cols = _check_col_expand_operands(
        src_view, out_view, dtype=dtype, shape=shape, context="TCOLEXPAND"
    )
    lanes = _micro_lane_count(dtype)
    vreg_type = _dsl_pto.VRegType(lanes, dtype)
    buf_bytes = rows * cols * _dtype_byte_width(dtype)

    src_addr = _const_i64(base_addr)
    out_addr = _const_i64(base_addr + buf_bytes)

    src_tile = _dsl_pto.make_tile_buffer(
        dtype, shape, space="VEC", valid_shape=[1, cols]
    ).alloc(addr=src_addr)
    out_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=out_addr)

    _dsl_pto.load(src_view, src_tile)

    src_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), src_addr)
    out_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), out_addr)

    for col in range(0, cols, lanes):
        active = builtins.min(lanes, cols - col)
        mask = _mask_for_chunk(dtype, active)
        col_offset = _scalar.const(col)
        vec = _dsl_pto.vlds(vreg_type, src_ptr, col_offset)
        for row in range(rows):
            dst_offset = _scalar.const(row * cols + col)
            _dsl_pto.vsts(vec, out_ptr, dst_offset, mask)

    _dsl_pto.store(out_tile, out_view)
    return out_view


def row_expand_micro(src_view, out_view, *, dtype, shape, base_addr=0):
    rows, cols = _check_row_expand_operands(
        src_view, out_view, dtype=dtype, shape=shape, context="TROWEXPAND"
    )
    lanes = _micro_lane_count(dtype)
    vreg_type = _dsl_pto.VRegType(lanes, dtype)
    buf_bytes = rows * cols * _dtype_byte_width(dtype)

    src_addr = _const_i64(base_addr)
    out_addr = _const_i64(base_addr + buf_bytes)

    src_tile = _dsl_pto.make_tile_buffer(
        dtype, shape, space="VEC", valid_shape=[rows, 1]
    ).alloc(addr=src_addr)
    out_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=out_addr)

    _dsl_pto.load(src_view, src_tile)

    src_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), src_addr)
    out_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), out_addr)

    for row in range(rows):
        scalar_offset = _scalar.const(row * cols)
        align = _dsl_pto.vldas(_dsl_pto.AlignType(), src_ptr, scalar_offset)
        scalar_vec, _, _ = _dsl_pto.vldus(
            vreg_type,
            _dsl_pto.AlignType(),
            _dsl_pto.ptr(dtype, space="VEC"),
            src_ptr,
            scalar_offset,
            align,
        )
        broadcast = _dsl_pto.vdup(vreg_type, scalar_vec, position="POS_LOWEST")
        for col in range(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = _mask_for_chunk(dtype, active)
            dst_offset = _scalar.const(row * cols + col)
            _dsl_pto.vsts(broadcast, out_ptr, dst_offset, mask)

    _dsl_pto.store(out_tile, out_view)
    return out_view


def row_expand_sub_micro(
    base_view, expand_view, out_view, *, dtype, shape, base_addr=0
):
    return _row_expand_binary_micro(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        op_name="vsub",
    )


def row_expand_mul_micro(
    base_view, expand_view, out_view, *, dtype, shape, base_addr=0
):
    return _row_expand_binary_micro(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        op_name="vmul",
    )


def row_expand_div_micro(
    base_view, expand_view, out_view, *, dtype, shape, base_addr=0
):
    return _row_expand_binary_micro(
        base_view,
        expand_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        op_name="vdiv",
    )


def row_sum_micro(src_view, out_view, *, dtype, shape, base_addr=0):
    return _row_reduce_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vcadd",
        combine_op_name="vadd",
        init_value=0.0,
    )


def row_max_micro(src_view, out_view, *, dtype, shape, base_addr=0):
    return _row_reduce_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vcmax",
        combine_op_name="vmax",
        init_value=float("-inf"),
    )


def row_min_micro(src_view, out_view, *, dtype, shape, base_addr=0):
    return _row_reduce_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vcmin",
        combine_op_name="vmin",
        init_value=float("inf"),
    )


def col_sum_micro(
    src_view, out_view, *, dtype, shape, base_addr=0, impl=VF_IMPL_DEFAULT
):
    return _col_reduce_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vadd",
        impl=impl,
    )


def col_max_micro(
    src_view, out_view, *, dtype, shape, base_addr=0, impl=VF_IMPL_DEFAULT
):
    return _col_reduce_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vmax",
        impl=impl,
    )


def col_min_micro(
    src_view, out_view, *, dtype, shape, base_addr=0, impl=VF_IMPL_DEFAULT
):
    return _col_reduce_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
        reduce_op_name="vmin",
        impl=impl,
    )


def mrgsort_micro(src_view, out_view, *, dtype, shape, block_len, base_addr=0):
    return _mrgsort_micro(
        src_view,
        out_view,
        dtype=dtype,
        shape=shape,
        block_len=block_len,
        base_addr=base_addr,
    )


def sort32_micro(src_view, idx_view, out_view, *, dtype, shape, base_addr=0):
    return _sort32_micro(
        src_view,
        idx_view,
        out_view,
        dtype=dtype,
        shape=shape,
        base_addr=base_addr,
    )


def _require_static_matrix_shape(shape, *, context):
    if len(shape) != 2 or any(not isinstance(dim, int) for dim in shape):
        raise ValueError(f"{context} currently requires a static rank-2 integer shape.")
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{context} requires positive row/column sizes.")
    return rows, cols


def _check_tbinop_operands(lhs_view, rhs_view, out_view, *, dtype, shape, context):
    rows, cols = _require_static_matrix_shape(shape, context=context)
    _require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "bf16", "i32", "u32", "i16", "u16", "i8", "u8"},
        message=f"Fix: {context} has invalid data type.",
    )
    for view, label in ((lhs_view, "src0"), (rhs_view, "src1"), (out_view, "dst")):
        _require_view_shape(
            view,
            [rows, cols],
            context=context,
            message=f"Fix: {context} input tile {label} valid shape mismatch with output tile dst shape.",
        )
        _require_view_dtype(
            view,
            dtype,
            message=f"Fix: {context} input tile src0, src1 and dst tile data type mismatch.",
        )
    return rows, cols


def _check_row_expand_operands(src_view, out_view, *, dtype, shape, context):
    rows, cols = _require_static_matrix_shape(shape, context=context)
    _require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "bf16", "i32", "u32", "i16", "u16", "i8", "u8"},
        message=f"Fix: {context} data type must be b8/b16/b32",
    )
    _require_view_shape(
        src_view,
        [rows, 1],
        context=context,
        message=f"Fix: {context} source valid shape must be [rows, 1].",
    )
    _require_view_shape(
        out_view,
        [rows, cols],
        context=context,
        message=f"Fix: {context} output valid shape mismatch.",
    )
    _require_view_dtype(
        src_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    _require_view_dtype(
        out_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    return rows, cols


def _check_col_expand_operands(src_view, out_view, *, dtype, shape, context):
    rows, cols = _require_static_matrix_shape(shape, context=context)
    _require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "bf16", "i32", "u32", "i16", "u16", "i8", "u8"},
        message=f"Fix: {context} data type must be b8/b16/b32",
    )
    _require_view_shape(
        src_view,
        [1, cols],
        context=context,
        message=f"Fix: {context} input valid col must be consistent with output valid col.",
    )
    _require_view_shape(
        out_view,
        [rows, cols],
        context=context,
        message=f"Fix: {context} output valid shape mismatch.",
    )
    _require_view_dtype(
        src_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    _require_view_dtype(
        out_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    return rows, cols


def _check_row_reduce_operands(src_view, out_view, *, dtype, shape, context):
    rows, cols = _require_static_matrix_shape(shape, context=context)
    _require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "i32", "i16"},
        message=(
            "Row reduction only supports 'half', 'float', 'int32', or 'int16' data types. "
            "Fix: Define TileDataIn with DType = half, float, int32, or int16."
        ),
    )
    _require_view_shape(
        src_view,
        [rows, cols],
        context=context,
        message="Fix: Ensure src valid shape matches [rows, cols].",
    )
    _require_view_shape(
        out_view,
        [rows, 1],
        context=context,
        message="Fix: Pass dstValidRow = srcValidRows and use a single-column output tile.",
    )
    _require_view_dtype(
        src_view,
        dtype,
        message="Fix: Ensure TileDataOut uses the same DType as TileDataIn.",
    )
    _require_view_dtype(
        out_view,
        dtype,
        message="Fix: Ensure TileDataOut uses the same DType as TileDataIn.",
    )
    return rows, cols


def _check_col_reduce_operands(src_view, out_view, *, dtype, shape, context):
    rows, cols = _require_static_matrix_shape(shape, context=context)
    _require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "bf16", "i32", "u32", "i16", "u16", "i8", "u8"},
        message=f"Fix: {context} input data type is not supported by this instruction.",
    )
    _require_view_shape(
        src_view,
        [rows, cols],
        context=context,
        message=f"Fix: {context} input shape mismatch.",
    )
    _require_view_shape(
        out_view,
        [1, cols],
        context=context,
        message=f"Fix: {context} input valid row must be consistent with the output valid row.",
    )
    _require_view_dtype(
        src_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    _require_view_dtype(
        out_view,
        dtype,
        message=f"Fix: {context} input data type must be consistent with the output data type.",
    )
    return rows, cols


def _check_gather_operands(
    src_view, indices_view, out_view, *, dtype, index_dtype, shape
):
    rows, cols = _require_static_matrix_shape(shape, context="TGATHER")
    dtype_token = _require_supported_dtype(
        dtype,
        allowed={"f32", "f16", "i32", "u32", "i16", "u16"},
        message="Fix: TGATHER Src data type must be int16_t/uint16_t/int32_t/uint32_t/half/float.",
    )
    index_token = _require_supported_dtype(
        index_dtype,
        allowed={"i32", "u32", "i16", "u16"},
        message="Fix: TGATHER expect b16/b32",
    )
    if _dtype_byte_width(dtype) != _dtype_byte_width(index_dtype):
        raise ValueError(
            "Fix: TGATHER micro lowering currently supports same-width source/index pairs only."
        )
    for view, expected_shape, label in (
        (src_view, [rows, cols], "src"),
        (indices_view, [rows, cols], "indices"),
        (out_view, [rows, cols], "dst"),
    ):
        _require_view_shape(
            view,
            expected_shape,
            context="TGATHER",
            message=f"Fix: TGATHER {label} shape mismatch.",
        )
    _require_view_dtype(
        src_view,
        dtype,
        message="Fix: TGATHER expect same type size for dst and src",
    )
    _require_view_dtype(
        out_view,
        dtype,
        message="Fix: TGATHER expect same type size for dst and src",
    )
    _require_view_dtype(
        indices_view,
        index_dtype,
        message="Fix: TGATHER expect b16/b32",
    )
    return rows, cols, dtype_token, index_token


def _check_mrgsort_operands(src_view, out_view, *, dtype, shape, block_len):
    rows, cols = _require_static_matrix_shape(shape, context="TMRGSORT")
    _require_supported_dtype(
        dtype,
        allowed={"f32", "f16"},
        message="TMrgsort: Unsupported data type! Supported types is half/float",
    )
    if rows != 1:
        raise ValueError("TMrgsort: the row of Destination and Source tile must be 1.")
    if block_len <= 0 or cols % (block_len * 4) != 0:
        raise ValueError("TMrgsort: src columns must be divisible by blockLen * 4.")
    _require_view_shape(
        src_view,
        [rows, cols],
        context="TMRGSORT",
        message="TMrgsort: source tile shape mismatch.",
    )
    _require_view_shape(
        out_view,
        [rows, cols],
        context="TMRGSORT",
        message="TMrgsort: destination tile shape mismatch.",
    )
    _require_view_dtype(
        src_view,
        dtype,
        message="TMrgsort: Destination and Source tile data types must be the same.",
    )
    _require_view_dtype(
        out_view,
        dtype,
        message="TMrgsort: Destination and Source tile data types must be the same.",
    )
    return rows, cols


def _check_sort32_operands(src_view, idx_view, out_view, *, dtype, shape):
    rows, cols = _require_static_matrix_shape(shape, context="TSORT32")
    _require_supported_dtype(
        dtype,
        allowed={"f32", "f16"},
        message="Dst and src must be float or half.",
    )
    out_cols = cols * (2 if _dtype_token(dtype) == "f32" else 4)
    for view, expected_shape, label in (
        (src_view, [rows, cols], "src"),
        (idx_view, [rows, cols], "idx"),
        (out_view, [rows, out_cols], "dst"),
    ):
        _require_view_shape(
            view,
            expected_shape,
            context="TSORT32",
            message=f"TSORT32 {label} shape mismatch.",
        )
    _require_view_dtype(
        src_view,
        dtype,
        message="Dst and src mube be same.",
    )
    _require_view_dtype(
        out_view,
        dtype,
        message="Dst and src mube be same.",
    )
    _require_view_dtype(
        idx_view,
        _dsl_pto.uint32,
        message="Idx must be uint32_t.",
    )
    if cols % 32 != 0:
        raise ValueError(
            "TSORT32 micro lowering currently requires column count divisible by 32."
        )
    return rows, cols, out_cols


def _row_expand_binary_micro(
    base_view, expand_view, out_view, *, dtype, shape, base_addr, op_name
):
    rows, cols = _check_row_expand_operands(
        expand_view,
        out_view,
        dtype=dtype,
        shape=shape,
        context=f"TROWEXPAND_{op_name[1:].upper()}",
    )
    _require_view_shape(
        base_view,
        [rows, cols],
        context=op_name,
        message=f"Fix: TROWEXPAND_{op_name[1:].upper()} base input valid shape mismatch with output tile dst shape.",
    )
    _require_view_dtype(
        base_view,
        dtype,
        message=f"Fix: TROWEXPAND_{op_name[1:].upper()} input data type must be consistent with the output data type.",
    )
    lanes = _micro_lane_count(dtype)
    vreg_type = _dsl_pto.VRegType(lanes, dtype)
    buf_bytes = rows * cols * _dtype_byte_width(dtype)

    base_addr_value = _const_i64(base_addr)
    expand_addr_value = _const_i64(base_addr + buf_bytes)
    out_addr_value = _const_i64(base_addr + buf_bytes * 2)

    base_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(
        addr=base_addr_value
    )
    expand_tile = _dsl_pto.make_tile_buffer(
        dtype, shape, space="VEC", valid_shape=[rows, 1]
    ).alloc(addr=expand_addr_value)
    out_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(
        addr=out_addr_value
    )

    _dsl_pto.load(base_view, base_tile)
    _dsl_pto.load(expand_view, expand_tile)

    base_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), base_addr_value)
    expand_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), expand_addr_value)
    out_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), out_addr_value)
    micro_op = getattr(_dsl_pto, op_name)

    for row in range(rows):
        scalar_offset = _scalar.const(row * cols)
        align = _dsl_pto.vldas(_dsl_pto.AlignType(), expand_ptr, scalar_offset)
        scalar_vec, _, _ = _dsl_pto.vldus(
            vreg_type,
            _dsl_pto.AlignType(),
            _dsl_pto.ptr(dtype, space="VEC"),
            expand_ptr,
            scalar_offset,
            align,
        )
        broadcast = _dsl_pto.vdup(vreg_type, scalar_vec, position="POS_LOWEST")
        for col in range(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = _mask_for_chunk(dtype, active)
            row_offset = _scalar.const(row * cols + col)
            base_vec = _dsl_pto.vlds(vreg_type, base_ptr, row_offset)
            out_vec = micro_op(vreg_type, base_vec, broadcast, mask)
            _dsl_pto.vsts(out_vec, out_ptr, row_offset, mask)

    _dsl_pto.store(out_tile, out_view)
    return out_view


def _row_reduce_micro(
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
    rows, cols = _check_row_reduce_operands(
        src_view, out_view, dtype=dtype, shape=shape, context="TROWREDUCE"
    )
    width = _dtype_byte_width(dtype)
    if width not in {2, 4}:
        raise ValueError(f"{reduce_op_name} currently supports only float16/float32.")

    lanes = _micro_lane_count(dtype)
    vreg_type = _dsl_pto.VRegType(lanes, dtype)
    buf_bytes = rows * cols * width

    src_addr = _const_i64(base_addr)
    out_addr = _const_i64(base_addr + buf_bytes)

    src_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=src_addr)
    out_tile = _dsl_pto.make_tile_buffer(
        dtype, shape, space="VEC", valid_shape=[rows, 1]
    ).alloc(addr=out_addr)

    _dsl_pto.load(src_view, src_tile)

    src_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), src_addr)
    out_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), out_addr)
    reduce_op = getattr(_dsl_pto, reduce_op_name)
    combine_op = getattr(_dsl_pto, combine_op_name)
    full_mask = _full_mask(dtype)
    point_mask = _tail_mask(dtype, 1)
    init_scalar = _const_float(dtype, init_value)

    for row in range(rows):
        accum = _dsl_pto.vbr(vreg_type, init_scalar)
        for col in range(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = _mask_for_chunk(dtype, active)
            offset = _scalar.const(row * cols + col)
            vec = _dsl_pto.vlds(vreg_type, src_ptr, offset)
            reduced = reduce_op(vreg_type, vec, mask)
            accum = combine_op(vreg_type, accum, reduced, full_mask)
        out_offset = _scalar.const(row * cols)
        _dsl_pto.vsts(accum, out_ptr, out_offset, point_mask, dist=_onept_dist(dtype))

    _dsl_pto.store(out_tile, out_view)
    return out_view


def _col_reduce_micro(
    src_view,
    out_view,
    *,
    dtype,
    shape,
    base_addr,
    reduce_op_name,
    impl,
):
    rows, cols = _check_col_reduce_operands(
        src_view, out_view, dtype=dtype, shape=shape, context="TCOLREDUCE"
    )
    lanes = _micro_lane_count(dtype)
    buf_bytes = rows * cols * _dtype_byte_width(dtype)

    src_addr = _const_i64(base_addr)
    out_addr = _const_i64(base_addr + buf_bytes)

    src_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=src_addr)
    out_tile = _dsl_pto.make_tile_buffer(
        dtype, [1, cols], space="VEC", valid_shape=[1, cols]
    ).alloc(addr=out_addr)

    _dsl_pto.load(src_view, src_tile)

    ptr_type = _dsl_pto.ptr(dtype, space="VEC")
    vreg_type = _dsl_pto.VRegType(lanes, dtype)
    src_ptr = _dsl_pto.castptr(ptr_type, src_addr)
    out_ptr = _dsl_pto.castptr(ptr_type, out_addr)
    reduce_op = getattr(_dsl_pto, reduce_op_name)
    impl_kind = _normalize_vf_impl_kind(impl)
    if const_expr(impl_kind == VF_IMPL_DEFAULT):
        impl_kind = VF_IMPL_1D_POST_UPDATE

    if const_expr(impl_kind in {VF_IMPL_1D_NO_POST_UPDATE, VF_IMPL_2D_NO_POST_UPDATE}):
        _col_reduce_micro_no_post_update(
            src_ptr,
            out_ptr,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            vreg_type=vreg_type,
            reduce_op=reduce_op,
        )
    elif const_expr(impl_kind in {VF_IMPL_1D_POST_UPDATE, VF_IMPL_2D_POST_UPDATE}):
        _col_reduce_micro_post_update(
            src_ptr,
            out_ptr,
            ptr_type=ptr_type,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            vreg_type=vreg_type,
            reduce_op=reduce_op,
        )
    else:
        raise ValueError(f"Unexpected normalized VF impl kind '{impl_kind}'.")

    _dsl_pto.store(out_tile, out_view)
    return out_view


def _col_reduce_micro_no_post_update(
    src_ptr, out_ptr, *, dtype, rows, cols, lanes, vreg_type, reduce_op
):
    loop_pairs = (rows - 1) // 2
    remain = (rows - 1) % 2
    for col in range_constexpr(0, cols, lanes):
        active = builtins.min(lanes, cols - col)
        mask = _mask_for_chunk(dtype, active)
        accum = _dsl_pto.vlds(vreg_type, src_ptr, _scalar.const(col))
        for pair in range_constexpr(loop_pairs):
            row0 = 2 * pair + 1
            row1 = 2 * pair + 2
            src0 = _dsl_pto.vlds(vreg_type, src_ptr, _scalar.const(col + row0 * cols))
            src1 = _dsl_pto.vlds(vreg_type, src_ptr, _scalar.const(col + row1 * cols))
            tmp = reduce_op(vreg_type, src0, src1, mask)
            accum = reduce_op(vreg_type, accum, tmp, mask)
        if const_expr(remain):
            tail_row = 2 * loop_pairs + 1
            src_tail = _dsl_pto.vlds(
                vreg_type, src_ptr, _scalar.const(col + tail_row * cols)
            )
            accum = reduce_op(vreg_type, accum, src_tail, mask)
        _dsl_pto.vsts(accum, out_ptr, _scalar.const(col), mask)


def _col_reduce_micro_post_update(
    src_ptr, out_ptr, *, ptr_type, dtype, rows, cols, lanes, vreg_type, reduce_op
):
    src_cursor = src_ptr
    out_cursor = out_ptr
    loop_pairs = (rows - 1) // 2
    remain = (rows - 1) % 2
    lane_step = _scalar.const(lanes)
    pair_stride = _scalar.const(cols * 2)
    for col in range_constexpr(0, cols, lanes):
        active = builtins.min(lanes, cols - col)
        mask = _mask_for_chunk(dtype, active)
        chunk_base = src_cursor
        accum, src_cursor = _dsl_pto.vlds_post(
            vreg_type, ptr_type, src_cursor, lane_step
        )
        row0_ptr = _dsl_pto.addptr(chunk_base, _scalar.const(cols))
        row1_ptr = _dsl_pto.addptr(chunk_base, _scalar.const(cols * 2))
        for _ in range_constexpr(loop_pairs):
            src0, row0_ptr = _dsl_pto.vlds_post(
                vreg_type, ptr_type, row0_ptr, pair_stride
            )
            src1, row1_ptr = _dsl_pto.vlds_post(
                vreg_type, ptr_type, row1_ptr, pair_stride
            )
            tmp = reduce_op(vreg_type, src0, src1, mask)
            accum = reduce_op(vreg_type, accum, tmp, mask)
        if const_expr(remain):
            src_tail = _dsl_pto.vlds(vreg_type, row0_ptr, _scalar.const(0))
            accum = reduce_op(vreg_type, accum, src_tail, mask)
        out_cursor = _dsl_pto.vsts_post(ptr_type, accum, out_cursor, lane_step, mask)


def _gather_micro(
    src_view,
    indices_view,
    out_view,
    *,
    dtype,
    index_dtype,
    shape,
    base_addr,
):
    rows, cols, _, _ = _check_gather_operands(
        src_view,
        indices_view,
        out_view,
        dtype=dtype,
        index_dtype=index_dtype,
        shape=shape,
    )
    src_bytes = rows * cols * _dtype_byte_width(dtype)
    idx_bytes = rows * cols * _dtype_byte_width(index_dtype)

    src_addr = _const_i64(base_addr)
    idx_addr = _const_i64(base_addr + src_bytes)
    out_addr = _const_i64(base_addr + src_bytes + idx_bytes)

    src_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=src_addr)
    idx_tile = _dsl_pto.make_tile_buffer(index_dtype, shape, space="VEC").alloc(
        addr=idx_addr
    )
    out_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=out_addr)

    _dsl_pto.load(src_view, src_tile)
    _dsl_pto.load(indices_view, idx_tile)

    src_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), src_addr)
    idx_ptr = _dsl_pto.castptr(_dsl_pto.ptr(index_dtype, space="VEC"), idx_addr)
    out_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), out_addr)
    lanes = _micro_lane_count(dtype)
    vreg_type = _dsl_pto.VRegType(lanes, dtype)
    index_vreg_type = _dsl_pto.VRegType(_micro_lane_count(index_dtype), index_dtype)

    for row in range_constexpr(rows):
        row_base = row * cols
        for col in range_constexpr(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            offset = _scalar.const(row_base + col)
            mask = _mask_for_chunk(dtype, active)
            idx_vec = _dsl_pto.vlds(index_vreg_type, idx_ptr, offset)
            out_vec = _dsl_pto.vgather2(
                vreg_type, src_ptr, idx_vec, _scalar.const(active)
            )
            _dsl_pto.vsts(out_vec, out_ptr, offset, mask)

    _dsl_pto.store(out_tile, out_view)
    return out_view


def _mrgsort_micro(src_view, out_view, *, dtype, shape, block_len, base_addr):
    _, cols = _check_mrgsort_operands(
        src_view, out_view, dtype=dtype, shape=shape, block_len=block_len
    )
    src_addr = _const_i64(base_addr)
    out_addr = _const_i64(base_addr + cols * _dtype_byte_width(dtype))

    src_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=src_addr)
    out_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=out_addr)
    _dsl_pto.load(src_view, src_tile)

    ptr_type = _dsl_pto.ptr(dtype, space="VEC")
    src_ptr = _dsl_pto.castptr(ptr_type, src_addr)
    out_ptr = _dsl_pto.castptr(ptr_type, out_addr)

    src1_ptr = _dsl_pto.addptr(src_ptr, _scalar.const(block_len))
    src2_ptr = _dsl_pto.addptr(src_ptr, _scalar.const(block_len * 2))
    src3_ptr = _dsl_pto.addptr(src_ptr, _scalar.const(block_len * 3))

    num_structures = (block_len * _dtype_byte_width(dtype)) >> 3
    count_value = (
        num_structures
        | (num_structures << 16)
        | (num_structures << 32)
        | (num_structures << 48)
    )
    repeat_times = cols // (block_len * 4)
    config_value = repeat_times | (0b1111 << 8)

    _dsl_pto.vmrgsort4(
        out_ptr,
        src_ptr,
        src1_ptr,
        src2_ptr,
        src3_ptr,
        _const_i64(count_value),
        _const_i64(config_value),
    )
    _dsl_pto.store(out_tile, out_view)
    return out_view


def _sort32_micro(src_view, idx_view, out_view, *, dtype, shape, base_addr):
    rows, cols, out_cols = _check_sort32_operands(
        src_view, idx_view, out_view, dtype=dtype, shape=shape
    )
    src_bytes = rows * cols * _dtype_byte_width(dtype)
    idx_bytes = rows * cols * 4

    src_addr = _const_i64(base_addr)
    idx_addr = _const_i64(base_addr + src_bytes)
    out_addr = _const_i64(base_addr + src_bytes + idx_bytes)

    src_tile = _dsl_pto.make_tile_buffer(dtype, [rows, cols], space="VEC").alloc(
        addr=src_addr
    )
    idx_tile = _dsl_pto.make_tile_buffer(
        _dsl_pto.uint32, [rows, cols], space="VEC"
    ).alloc(addr=idx_addr)
    out_tile = _dsl_pto.make_tile_buffer(dtype, [rows, out_cols], space="VEC").alloc(
        addr=out_addr
    )

    _dsl_pto.load(src_view, src_tile)
    _dsl_pto.load(idx_view, idx_tile)

    src_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), src_addr)
    idx_ptr = _dsl_pto.castptr(_dsl_pto.ptr(_dsl_pto.uint32, space="VEC"), idx_addr)
    out_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), out_addr)
    repeat_times = _scalar.const(cols // 32)

    for row in range_constexpr(rows):
        src_row = _dsl_pto.addptr(src_ptr, _scalar.const(row * cols))
        idx_row = _dsl_pto.addptr(idx_ptr, _scalar.const(row * cols))
        out_row = _dsl_pto.addptr(out_ptr, _scalar.const(row * out_cols))
        _dsl_pto.vbitsort(out_row, src_row, idx_row, repeat_times)

    _dsl_pto.store(out_tile, out_view)
    return out_view


def _binary_micro(
    lhs_view, rhs_view, out_view, *, dtype, shape, lanes, base_addr, op_name, impl
):
    rows, cols = _check_tbinop_operands(
        lhs_view,
        rhs_view,
        out_view,
        dtype=dtype,
        shape=shape,
        context=op_name.upper().replace("V", "T", 1),
    )
    lanes = _resolve_lanes(dtype, lanes)
    element_count = rows * cols
    buf_bytes = element_count * _dtype_byte_width(dtype)
    lhs_addr = _const_i64(base_addr)
    rhs_addr = _const_i64(base_addr + buf_bytes)
    out_addr = _const_i64(base_addr + buf_bytes * 2)

    lhs_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=lhs_addr)
    rhs_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=rhs_addr)
    out_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=out_addr)

    _dsl_pto.load(lhs_view, lhs_tile)
    _dsl_pto.load(rhs_view, rhs_tile)

    ptr_type = _dsl_pto.ptr(dtype, space="VEC")
    vreg_type = _dsl_pto.VRegType(lanes, dtype)
    lhs_ptr = _dsl_pto.castptr(ptr_type, lhs_addr)
    rhs_ptr = _dsl_pto.castptr(ptr_type, rhs_addr)
    out_ptr = _dsl_pto.castptr(ptr_type, out_addr)
    micro_op = getattr(_dsl_pto, op_name)
    impl_kind = _normalize_vf_impl_kind(impl)
    is_contiguous = rows == 1 or cols == element_count
    if const_expr(impl_kind == VF_IMPL_DEFAULT):
        impl_kind = (
            VF_IMPL_1D_POST_UPDATE if is_contiguous else VF_IMPL_2D_NO_POST_UPDATE
        )

    if const_expr(impl_kind == VF_IMPL_1D_NO_POST_UPDATE):
        _binary_micro_1d_no_post_update(
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            dtype=dtype,
            lanes=lanes,
            element_count=element_count,
            vreg_type=vreg_type,
            micro_op=micro_op,
        )
    elif const_expr(impl_kind == VF_IMPL_1D_POST_UPDATE):
        _binary_micro_1d_post_update(
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            ptr_type=ptr_type,
            dtype=dtype,
            lanes=lanes,
            element_count=element_count,
            vreg_type=vreg_type,
            micro_op=micro_op,
        )
    elif const_expr(impl_kind == VF_IMPL_2D_NO_POST_UPDATE):
        _binary_micro_2d_no_post_update(
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            vreg_type=vreg_type,
            micro_op=micro_op,
        )
    elif const_expr(impl_kind == VF_IMPL_2D_POST_UPDATE):
        _binary_micro_2d_post_update(
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            dtype=dtype,
            rows=rows,
            cols=cols,
            lanes=lanes,
            vreg_type=vreg_type,
            micro_op=micro_op,
        )
    else:
        raise ValueError(f"Unexpected normalized VF impl kind '{impl_kind}'.")

    _dsl_pto.store(out_tile, out_view)
    return out_view


def _binary_micro_1d_no_post_update(
    lhs_ptr, rhs_ptr, out_ptr, *, dtype, lanes, element_count, vreg_type, micro_op
):
    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
        mask = _mask_for_chunk(dtype, active)
        index = _scalar.const(offset)
        lhs_vec = _dsl_pto.vlds(vreg_type, lhs_ptr, index)
        rhs_vec = _dsl_pto.vlds(vreg_type, rhs_ptr, index)
        out_vec = micro_op(vreg_type, lhs_vec, rhs_vec, mask)
        _dsl_pto.vsts(out_vec, out_ptr, index, mask)


def _binary_micro_1d_post_update(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    *,
    ptr_type,
    dtype,
    lanes,
    element_count,
    vreg_type,
    micro_op,
):
    lhs_cursor = lhs_ptr
    rhs_cursor = rhs_ptr
    out_cursor = out_ptr
    lane_step = _scalar.const(lanes)
    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
        mask = _mask_for_chunk(dtype, active)
        lhs_vec, lhs_cursor = _dsl_pto.vlds_post(
            vreg_type, ptr_type, lhs_cursor, lane_step
        )
        rhs_vec, rhs_cursor = _dsl_pto.vlds_post(
            vreg_type, ptr_type, rhs_cursor, lane_step
        )
        out_vec = micro_op(vreg_type, lhs_vec, rhs_vec, mask)
        out_cursor = _dsl_pto.vsts_post(ptr_type, out_vec, out_cursor, lane_step, mask)


def _binary_micro_2d_no_post_update(
    lhs_ptr, rhs_ptr, out_ptr, *, dtype, rows, cols, lanes, vreg_type, micro_op
):
    for row in range_constexpr(rows):
        row_base = row * cols
        for col in range_constexpr(0, cols, lanes):
            active = builtins.min(lanes, cols - col)
            mask = _mask_for_chunk(dtype, active)
            index = _scalar.const(row_base + col)
            lhs_vec = _dsl_pto.vlds(vreg_type, lhs_ptr, index)
            rhs_vec = _dsl_pto.vlds(vreg_type, rhs_ptr, index)
            out_vec = micro_op(vreg_type, lhs_vec, rhs_vec, mask)
            _dsl_pto.vsts(out_vec, out_ptr, index, mask)


def _binary_micro_2d_post_update(
    lhs_ptr, rhs_ptr, out_ptr, *, dtype, rows, cols, lanes, vreg_type, micro_op
):
    _binary_micro_2d_no_post_update(
        lhs_ptr,
        rhs_ptr,
        out_ptr,
        dtype=dtype,
        rows=rows,
        cols=cols,
        lanes=lanes,
        vreg_type=vreg_type,
        micro_op=micro_op,
    )


def _rsqrt_micro(src_view, out_view, *, dtype, shape, lanes, base_addr):
    if any(not isinstance(dim, int) for dim in shape):
        raise ValueError(
            "micro tile lowering currently requires a static integer shape."
        )

    lanes = _resolve_lanes(dtype, lanes)
    element_count = 1
    for dim in shape:
        element_count *= dim

    buf_bytes = element_count * _dtype_byte_width(dtype)
    src_addr = _const_i64(base_addr)
    out_addr = _const_i64(base_addr + buf_bytes)

    src_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=src_addr)
    out_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=out_addr)

    _dsl_pto.load(src_view, src_tile)

    vreg_type = _dsl_pto.VRegType(lanes, dtype)
    src_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), src_addr)
    out_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), out_addr)

    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
        mask = _mask_for_chunk(dtype, active)
        index = _scalar.const(offset)
        src_vec = _dsl_pto.vlds(vreg_type, src_ptr, index)
        sqrt_vec = _dsl_pto.vsqrt(vreg_type, src_vec, mask)
        out_vec = _dsl_pto.vrec(vreg_type, sqrt_vec, mask)
        _dsl_pto.vsts(out_vec, out_ptr, index, mask)

    _dsl_pto.store(out_tile, out_view)
    return out_view


def _unary_micro(src_view, out_view, *, dtype, shape, lanes, base_addr, op_name):
    if any(not isinstance(dim, int) for dim in shape):
        raise ValueError(
            "micro tile lowering currently requires a static integer shape."
        )

    lanes = _resolve_lanes(dtype, lanes)
    element_count = 1
    for dim in shape:
        element_count *= dim

    buf_bytes = element_count * _dtype_byte_width(dtype)
    src_addr = _const_i64(base_addr)
    out_addr = _const_i64(base_addr + buf_bytes)

    src_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=src_addr)
    out_tile = _dsl_pto.make_tile_buffer(dtype, shape, space="VEC").alloc(addr=out_addr)

    _dsl_pto.load(src_view, src_tile)

    src_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), src_addr)
    out_ptr = _dsl_pto.castptr(_dsl_pto.ptr(dtype, space="VEC"), out_addr)
    micro_op = getattr(_dsl_pto, op_name) if op_name is not None else None

    for offset in range_constexpr(0, element_count, lanes):
        active = builtins.min(lanes, element_count - offset)
        mask = _mask_for_chunk(dtype, active)
        index = _scalar.const(offset)
        src_vec = _dsl_pto.vlds(_dsl_pto.VRegType(lanes, dtype), src_ptr, index)
        out_vec = (
            src_vec
            if micro_op is None
            else micro_op(_dsl_pto.VRegType(lanes, dtype), src_vec, mask)
        )
        _dsl_pto.vsts(out_vec, out_ptr, index, mask)

    _dsl_pto.store(out_tile, out_view)
    return out_view


def adds(src, scalar, out):
    _call(_pto.TAddSOp, src, scalar, out)
    return out


def sub(lhs, rhs, out):
    _call(_pto.TSubOp, lhs, rhs, out)
    return out


def subs(src, scalar, out):
    _call(_pto.TSubSOp, src, scalar, out)
    return out


def mul(lhs, rhs, out):
    _call(_pto.TMulOp, lhs, rhs, out)
    return out


def muls(src, scalar, out):
    _call(_pto.TMulSOp, src, scalar, out)
    return out


def div(lhs, rhs, out):
    _call(_pto.TDivOp, lhs, rhs, out)
    return out


def divs(src, scalar, out):
    _call(_pto.TDivSOp, src, scalar, out)
    return out


def max(lhs, rhs, out):
    _call(_pto.TMaxOp, lhs, rhs, out)
    return out


def maxs(src, scalar, out):
    _call(_pto.TMaxSOp, src, scalar, out)
    return out


def min(lhs, rhs, out):
    _call(_pto.TMinOp, lhs, rhs, out)
    return out


def mins(src, scalar, out):
    _call(_pto.TMinSOp, src, scalar, out)
    return out


def and_(lhs, rhs, out):
    _call(_pto.TAndOp, lhs, rhs, out)
    return out


def or_(lhs, rhs, out):
    _call(_pto.TOrOp, lhs, rhs, out)
    return out


def xor(lhs, rhs, out):
    _call(_pto.TXorOp, lhs, rhs, out)
    return out


def shl(lhs, rhs, out):
    _call(_pto.TShlOp, lhs, rhs, out)
    return out


def shls(src, scalar, out):
    _call(_pto.TShlSOp, src, scalar, out)
    return out


def shr(lhs, rhs, out):
    _call(_pto.TShrOp, lhs, rhs, out)
    return out


def shrs(src, scalar, out):
    _call(_pto.TShrSOp, src, scalar, out)
    return out


def compare(src0, src1, out, *, mode):
    _call(_pto.TCmpOp, src0, src1, out, cmpMode=_cmp_mode_attr(mode))
    return out


def exp(src, out):
    _call(_pto.TExpOp, src, out)
    return out


def log(src, out):
    _call(_pto.TLogOp, src, out)
    return out


def relu(src, out):
    _call(_pto.TReluOp, src, out)
    return out


def abs(src, out):
    _call(_pto.TAbsOp, src, out)
    return out


def sqrt(src, out):
    _call(_pto.TSqrtOp, src, out)
    return out


def rsqrt(src, out):
    _call(_pto.TRsqrtOp, src, out)
    return out


def reciprocal(src, out):
    _call(_pto.TRecipOp, src, out)
    return out


def lrelu(src, slope, out):
    _call(_pto.TLReluOp, src, slope, out)
    return out


def gather(src, out, *, indices=None, mask_pattern=None):
    kwargs = {}
    if indices is not None:
        kwargs["indices"] = indices
    if mask_pattern is not None:
        kwargs["maskPattern"] = _pto.MaskPatternAttr.get(
            getattr(_pto.MaskPattern, mask_pattern)
        )
    _call(_pto.TGatherOp, src, out, **kwargs)
    return out


def scatter(src, indices, out):
    _call(_pto.TScatterOp, src, indices, out)
    return out


def select(mask, src0, src1, tmp, out):
    _call(_pto.TSelOp, mask, src0, src1, tmp, out)
    return out


def concat(src0, src1, out):
    _call(_pto.TConcatOp, src0, src1, out)
    return out


def extract(source, index_row, index_col, out):
    _call(_pto.TExtractOp, source, index_row, index_col, out)
    return out


def insert(source, index_row, index_col, out):
    _call(_pto.TInsertOp, source, index_row, index_col, out)
    return out


def row_sum(src, tmp, dst):
    _call(_pto.TRowSumOp, src=src, tmp=tmp, dst=dst)
    return dst


def row_min(src, tmp, dst):
    _call(_pto.TRowMinOp, src=src, tmp=tmp, dst=dst)
    return dst


def row_max(src, tmp, dst):
    _call(_pto.TRowMaxOp, src=src, tmp=tmp, dst=dst)
    return dst


def col_sum(src, tmp, dst, *, is_binary=True):
    _call(_pto.TColSumOp, src=src, tmp=tmp, dst=dst, isBinary=is_binary)
    return dst


def col_min(src, dst):
    _call(_pto.TColMinOp, src=src, dst=dst)
    return dst


def col_max(src, dst):
    _call(_pto.TColMaxOp, src=src, dst=dst)
    return dst


def row_expand(src, dst):
    _call(_pto.TRowExpandOp, src=src, dst=dst)
    return dst


def row_expand_sub(src0, src1, dst):
    _call(_pto.TRowExpandSubOp, src0=src0, src1=src1, dst=dst)
    return dst


def row_expand_mul(src0, src1, dst):
    _call(_pto.TRowExpandMulOp, src0=src0, src1=src1, dst=dst)
    return dst


def row_expand_div(src0, src1, dst):
    _call(_pto.TRowExpandDivOp, src0=src0, src1=src1, dst=dst)
    return dst


def col_expand(src, dst):
    _call(_pto.TColExpandOp, src=src, dst=dst)
    return dst


def col_expand_mul(src0, src1, dst):
    _call(_pto.TColExpandMulOp, src0=src0, src1=src1, dst=dst)
    return dst


def col_expand_max(src0, src1, dst):
    _call(_pto.TColExpandMaxOp, src0=src0, src1=src1, dst=dst)
    return dst


def col_expand_min(src0, src1, dst):
    _call(_pto.TColExpandMinOp, src0=src0, src1=src1, dst=dst)
    return dst


def trans(src, dst):
    _call(_pto.TTransOp, src, dst)
    return dst


def mrgsort(src, dst, block_len):
    _call(_pto.TMrgSortOp, srcs=[src], dsts=[dst], blockLen=block_len)
    return dst


def sort32(src, dst, idx):
    _call(_pto.TSort32Op, src, dst, idx)
    return dst


def matmul(lhs, rhs, out):
    _call(_pto.TMatmulOp, None, lhs, rhs, out)
    return out


def matmul_acc(acc, lhs, rhs, out):
    _call(_pto.TMatmulAccOp, None, acc, lhs, rhs, out)
    return out


def matmul_bias(lhs, rhs, bias, out):
    _call(_pto.TMatmulBiasOp, None, lhs, rhs, bias, out)
    return out


def matmul_mx(lhs, lhs_scale, rhs, rhs_scale, out):
    _call(_pto.TMatmulMxOp, None, lhs, lhs_scale, rhs, rhs_scale, out)
    return out


def matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, out):
    _call(_pto.TMatmulMxAccOp, None, acc, lhs, lhs_scale, rhs, rhs_scale, out)
    return out


def matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, out):
    _call(_pto.TMatmulMxBiasOp, None, lhs, lhs_scale, rhs, rhs_scale, bias, out)
    return out


def full_mask_b32():
    return _dsl_pto.pset_b32(_dsl_pto.MaskType(), "PAT_ALL")


def vload(ptr, offset, *, lanes=64, dtype=None):
    dtype = _dsl_pto.float32 if dtype is None else dtype
    return _dsl_pto.vlds(_dsl_pto.VRegType(lanes, dtype), ptr, offset)


def vstore(vector, ptr, offset, *, mask=None):
    if mask is None:
        mask = full_mask_b32()
    _dsl_pto.vsts(vector, ptr, offset, mask)
    return ptr


def vector_copy(src_ptr, dst_ptr, offset, *, lanes=64, dtype=None):
    vec = vload(src_ptr, offset, lanes=lanes, dtype=dtype)
    vstore(vec, dst_ptr, offset)
    return vec


TLoad = load_tile
TStore = store_tile
TMov = move_tile
TAdd = add
TAddS = adds
TSub = sub
TSubS = subs
TMul = mul
TMulS = muls
TDiv = div
TDivS = divs
TMax = max
TMaxS = maxs
TMin = min
TMinS = mins
TAnd = and_
TOr = or_
TXor = xor
TShl = shl
TShlS = shls
TShr = shr
TShrS = shrs
TCmp = compare
TExp = exp
TLog = log
TRelu = relu
TAbs = abs
TSqrt = sqrt
TRsqrt = rsqrt
TRecip = reciprocal
TLRelu = lrelu
TGather = gather
TScatter = scatter
TSel = select
TConcat = concat
TExtract = extract
TInsert = insert
TRowSum = row_sum
TRowMin = row_min
TRowMax = row_max
TColSum = col_sum
TColMin = col_min
TColMax = col_max
TRowExpand = row_expand
TRowExpandSub = row_expand_sub
TRowExpandMul = row_expand_mul
TRowExpandDiv = row_expand_div
TColExpand = col_expand
TColExpandMul = col_expand_mul
TColExpandMax = col_expand_max
TColExpandMin = col_expand_min
TTrans = trans
TMrgSort = mrgsort
TSort32 = sort32
TMatmul = matmul
TMatmulAcc = matmul_acc
TMatmulBias = matmul_bias
TMatmulMx = matmul_mx
TMatmulMxAcc = matmul_mx_acc
TMatmulMxBias = matmul_mx_bias


__all__ = [
    "VF_IMPL_DEFAULT",
    "VF_IMPL_1D_NO_POST_UPDATE",
    "VF_IMPL_1D_POST_UPDATE",
    "VF_IMPL_2D_NO_POST_UPDATE",
    "VF_IMPL_2D_POST_UPDATE",
    "TAbs",
    "TAdd",
    "TAddS",
    "TAnd",
    "TColExpand",
    "TColExpandMax",
    "TColExpandMin",
    "TColExpandMul",
    "TColMax",
    "TColMin",
    "TColSum",
    "TConcat",
    "TCmp",
    "TDiv",
    "TDivS",
    "TExp",
    "TExtract",
    "TGather",
    "TInsert",
    "TLRelu",
    "TLoad",
    "TLog",
    "TMatmul",
    "TMatmulAcc",
    "TMatmulBias",
    "TMatmulMx",
    "TMatmulMxAcc",
    "TMatmulMxBias",
    "TMax",
    "TMaxS",
    "TMin",
    "TMinS",
    "TMov",
    "TMrgSort",
    "TMul",
    "TMulS",
    "TOr",
    "TRecip",
    "TRelu",
    "TRowExpand",
    "TRowExpandDiv",
    "TRowExpandMul",
    "TRowExpandSub",
    "TRowMax",
    "TRowMin",
    "TRowSum",
    "TRsqrt",
    "TScatter",
    "TSel",
    "TShl",
    "TShlS",
    "TShr",
    "TShrS",
    "TSort32",
    "TSqrt",
    "TStore",
    "TSub",
    "TSubS",
    "TTrans",
    "TXor",
    "add",
    "add_micro",
    "abs_micro",
    "adds",
    "and_",
    "col_expand",
    "col_expand_micro",
    "col_expand_max",
    "col_expand_min",
    "col_expand_mul",
    "col_max",
    "col_max_micro",
    "col_min",
    "col_min_micro",
    "col_sum",
    "col_sum_micro",
    "compare",
    "concat",
    "div",
    "divs",
    "exp",
    "exp_micro",
    "extract",
    "full_mask_b32",
    "gather",
    "gather_micro",
    "insert",
    "load_tile",
    "log",
    "log_micro",
    "lrelu",
    "matmul",
    "matmul_acc",
    "matmul_bias",
    "matmul_mx",
    "matmul_mx_acc",
    "matmul_mx_bias",
    "max",
    "maxs",
    "min",
    "mins",
    "move_tile",
    "mov_micro",
    "mrgsort",
    "mrgsort_micro",
    "mul",
    "muls",
    "or_",
    "reciprocal",
    "reciprocal_micro",
    "relu",
    "relu_micro",
    "row_expand",
    "row_expand_div_micro",
    "row_expand_micro",
    "row_expand_mul_micro",
    "row_expand_div",
    "row_expand_sub_micro",
    "row_expand_mul",
    "row_expand_sub",
    "row_max",
    "row_max_micro",
    "row_min",
    "row_min_micro",
    "row_sum",
    "row_sum_micro",
    "rsqrt",
    "rsqrt_micro",
    "scatter",
    "select",
    "shl",
    "shls",
    "shr",
    "shrs",
    "sort32",
    "sort32_micro",
    "sqrt",
    "sqrt_micro",
    "store_tile",
    "sub",
    "sub_micro",
    "subs",
    "div_micro",
    "mul_micro",
    "or_micro",
    "trans",
    "vector_copy",
    "vload",
    "vstore",
    "xor",
]
