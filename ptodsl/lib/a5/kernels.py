from mlir.dialects import pto as raw_pto
from mlir.ir import IndexType

from ... import Constexpr, language as dsl, scalar as s, to_ir_module
from ...api.scalar import _unwrap
from ...language import make_mxfp8
from ._common import (
    VF_IMPL_DEFAULT,
    alloc_tile_buffer,
    load_tile,
    mask_type,
    make_tensor,
    ptr,
    slice_tensor,
    store_tile,
    vreg_type,
)
from . import ops

_DSL_DTYPE_BY_NAME = {
    "float32": lambda: dsl.float32,
    "float16": lambda: dsl.float16,
    "bfloat16": lambda: dsl.bfloat16,
    "int32": lambda: dsl.int32,
    "int16": lambda: dsl.int16,
    "int8": lambda: dsl.int8,
    "uint32": lambda: dsl.uint32,
    "uint16": lambda: dsl.uint16,
    "uint8": lambda: dsl.uint8,
    "bool": lambda: dsl.bool,
}


def _resolve_dtype(dtype, default_name):
    if dtype is None:
        return _DSL_DTYPE_BY_NAME[default_name]()
    if isinstance(dtype, str):
        if dtype not in _DSL_DTYPE_BY_NAME:
            raise ValueError(f"Unsupported PTODSL dtype '{dtype}'.")
        return _DSL_DTYPE_BY_NAME[dtype]()
    return dtype


def build_elementwise_add(*, rows=32, cols=32, tile_rows=32, tile_cols=32, dtype=None):
    def meta_data():
        element_dtype = _resolve_dtype(dtype, "float32")
        return {
            "ptr_t": ptr(element_dtype),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_elementwise_add(
        src0: "ptr_t",
        src1: "ptr_t",
        dst: "ptr_t",
        n_rows: "index_t",
        n_cols: "index_t",
    ) -> None:
        element_dtype = _resolve_dtype(dtype, "float32")
        lhs = make_tensor(src0, shape=[n_rows, n_cols], dtype=element_dtype)
        rhs = make_tensor(src1, shape=[n_rows, n_cols], dtype=element_dtype)
        out = make_tensor(dst, shape=[n_rows, n_cols], dtype=element_dtype)

        lhs_tile = slice_tensor(
            lhs,
            offsets=[0, 0],
            sizes=[tile_rows, tile_cols],
            dtype=element_dtype,
        )
        rhs_tile = slice_tensor(
            rhs,
            offsets=[0, 0],
            sizes=[tile_rows, tile_cols],
            dtype=element_dtype,
        )
        out_tile = slice_tensor(
            out,
            offsets=[0, 0],
            sizes=[tile_rows, tile_cols],
            dtype=element_dtype,
        )

        with dsl.vector_section():
            ops.tadd(
                lhs_tile,
                rhs_tile,
                out_tile,
                dtype=element_dtype,
                tile_shape=[tile_rows, tile_cols],
            )

    return a5_elementwise_add


def build_templated_elementwise_add(*, dtype=None):
    def specialize(
        *,
        ROWS: Constexpr[int] = 32,
        COLS: Constexpr[int] = 32,
        VF_IMPL: Constexpr[str] = VF_IMPL_DEFAULT,
    ):
        def meta_data():
            element_dtype = _resolve_dtype(dtype, "float32")
            return {
                "ptr_t": ptr(element_dtype),
                "shape": [ROWS, COLS],
            }

        @to_ir_module(meta_data=meta_data)
        def a5_templated_elementwise_add(
            src0: "ptr_t",
            src1: "ptr_t",
            dst: "ptr_t",
        ) -> None:
            element_dtype = _resolve_dtype(dtype, "float32")
            lhs = make_tensor(src0, shape=shape, dtype=element_dtype)
            rhs = make_tensor(src1, shape=shape, dtype=element_dtype)
            out = make_tensor(dst, shape=shape, dtype=element_dtype)

            with dsl.vector_section():
                ops.tadd(
                    slice_tensor(lhs, offsets=[0, 0], sizes=shape, dtype=element_dtype),
                    slice_tensor(rhs, offsets=[0, 0], sizes=shape, dtype=element_dtype),
                    slice_tensor(out, offsets=[0, 0], sizes=shape, dtype=element_dtype),
                    dtype=element_dtype,
                    tile_shape=shape,
                    impl=VF_IMPL,
                )

        return a5_templated_elementwise_add

    return specialize


def build_vector_copy(*, lanes=64, dtype=None):
    def meta_data():
        element_dtype = _resolve_dtype(dtype, "float32")
        return {
            "ptr_t": ptr(element_dtype, space="VEC"),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_vector_copy(src: "ptr_t", dst: "ptr_t", offset: "index_t") -> None:
        element_dtype = _resolve_dtype(dtype, "float32")
        ops.vector_copy(src, dst, offset, lanes=lanes, dtype=element_dtype)

    return a5_vector_copy


def build_hivm_vadd_demo(*, lanes=64, dtype=None):
    def meta_data():
        element_dtype = _resolve_dtype(dtype, "float32")
        return {
            "ptr_t": ptr(element_dtype, space="VEC"),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_hivm_vadd_demo(
        src0: "ptr_t", src1: "ptr_t", dst: "ptr_t", offset: "index_t"
    ) -> None:
        element_dtype = _resolve_dtype(dtype, "float32")
        vector_type = vreg_type(lanes, element_dtype)
        mask = raw_pto.pset_b32(mask_type(), "PAT_ALL")
        lhs = raw_pto.vlds(vector_type, _unwrap(src0), _unwrap(offset))
        rhs = raw_pto.vlds(vector_type, _unwrap(src1), _unwrap(offset))
        out = raw_pto.vadd(vector_type, lhs, rhs, mask)
        raw_pto.vsts(out, _unwrap(dst), _unwrap(offset), mask)

    return a5_hivm_vadd_demo


def build_mxfp8_matmul(*, m=16, k=64, n=32, lhs_variant="e5m2", rhs_variant="e5m2"):
    def meta_data():
        mx = make_mxfp8(lhs=lhs_variant, rhs=rhs_variant)
        return {
            "ptr_lhs": ptr(mx.lhs),
            "ptr_rhs": ptr(mx.rhs),
            "ptr_scale": ptr(mx.scale),
            "ptr_bias": ptr(mx.acc),
            "ptr_out": ptr(mx.acc),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_mxfp8_matmul(
        lhs_ptr: "ptr_lhs",
        lhs_scale_ptr: "ptr_scale",
        rhs_ptr: "ptr_rhs",
        rhs_scale_ptr: "ptr_scale",
        bias_ptr: "ptr_bias",
        out_ptr: "ptr_out",
    ) -> None:
        mx = make_mxfp8(lhs=lhs_variant, rhs=rhs_variant)
        scale_k = mx.scale_k(k)
        lhs = make_tensor(lhs_ptr, shape=[m, k], dtype=mx.lhs)
        rhs = make_tensor(rhs_ptr, shape=[k, n], dtype=mx.rhs)
        lhs_scale = make_tensor(lhs_scale_ptr, shape=[m, scale_k], dtype=mx.scale)
        rhs_scale = make_tensor(rhs_scale_ptr, shape=[scale_k, n], dtype=mx.scale)
        bias = make_tensor(bias_ptr, shape=[1, n], dtype=mx.acc)
        out = make_tensor(out_ptr, shape=[m, n], dtype=mx.acc)

        with dsl.cube_section():
            lhs_tile = load_tile(
                slice_tensor(lhs, offsets=[0, 0], sizes=[m, k], dtype=mx.lhs),
                dtype=mx.lhs,
                shape=[m, k],
                space="LEFT",
            )
            rhs_tile = load_tile(
                slice_tensor(rhs, offsets=[0, 0], sizes=[k, n], dtype=mx.rhs),
                dtype=mx.rhs,
                shape=[k, n],
                space="RIGHT",
            )
            lhs_scale_tile = load_tile(
                slice_tensor(
                    lhs_scale,
                    offsets=[0, 0],
                    sizes=[m, scale_k],
                    dtype=mx.scale,
                ),
                dtype=mx.scale,
                shape=[m, scale_k],
                space="SCALING",
                config=dsl.TileBufConfig(
                    blayout="RowMajor",
                    slayout="RowMajor",
                    s_fractal_size=raw_pto.TileConfig.fractalMxSize,
                ),
            )
            rhs_scale_tile = load_tile(
                slice_tensor(
                    rhs_scale,
                    offsets=[0, 0],
                    sizes=[scale_k, n],
                    dtype=mx.scale,
                ),
                dtype=mx.scale,
                shape=[scale_k, n],
                space="SCALING",
                config=dsl.TileBufConfig(
                    blayout="ColMajor",
                    slayout="ColMajor",
                    s_fractal_size=raw_pto.TileConfig.fractalMxSize,
                ),
            )
            bias_tile = load_tile(
                slice_tensor(bias, offsets=[0, 0], sizes=[1, n], dtype=mx.acc),
                dtype=mx.acc,
                shape=[1, n],
                space="BIAS",
            )
            acc_tile = alloc_tile_buffer(mx.acc, [m, n], space="ACC")
            ops.matmul_mx_bias(
                lhs_tile,
                lhs_scale_tile,
                rhs_tile,
                rhs_scale_tile,
                bias_tile,
                acc_tile,
            )
            store_tile(
                acc_tile, slice_tensor(out, offsets=[0, 0], sizes=[m, n], dtype=mx.acc)
            )

    return a5_mxfp8_matmul


def build_cube_matmul(
    *, m=16, k=32, n=16, lhs_dtype=None, rhs_dtype=None, acc_dtype=None
):
    def meta_data():
        lhs_element_dtype = _resolve_dtype(lhs_dtype, "float16")
        rhs_element_dtype = _resolve_dtype(rhs_dtype, "float16")
        acc_element_dtype = _resolve_dtype(acc_dtype, "float32")
        return {
            "ptr_lhs": ptr(lhs_element_dtype),
            "ptr_rhs": ptr(rhs_element_dtype),
            "ptr_out": ptr(acc_element_dtype),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_cube_matmul(
        lhs_ptr: "ptr_lhs", rhs_ptr: "ptr_rhs", out_ptr: "ptr_out"
    ) -> None:
        lhs_element_dtype = _resolve_dtype(lhs_dtype, "float16")
        rhs_element_dtype = _resolve_dtype(rhs_dtype, "float16")
        acc_element_dtype = _resolve_dtype(acc_dtype, "float32")
        c0 = s.const(0)
        lhs = make_tensor(lhs_ptr, shape=[m, k], dtype=lhs_element_dtype)
        rhs = make_tensor(rhs_ptr, shape=[k, n], dtype=rhs_element_dtype)
        out = make_tensor(out_ptr, shape=[m, n], dtype=acc_element_dtype)

        with dsl.cube_section():
            lhs_mat = load_tile(
                slice_tensor(
                    lhs, offsets=[0, 0], sizes=[m, k], dtype=lhs_element_dtype
                ),
                dtype=lhs_element_dtype,
                tile_shape=[m, k],
                space="MAT",
            )
            rhs_mat = load_tile(
                slice_tensor(
                    rhs, offsets=[0, 0], sizes=[k, n], dtype=rhs_element_dtype
                ),
                dtype=rhs_element_dtype,
                tile_shape=[k, n],
                space="MAT",
            )
            lhs_tile = alloc_tile_buffer(lhs_element_dtype, [m, k], space="LEFT")
            rhs_tile = alloc_tile_buffer(rhs_element_dtype, [k, n], space="RIGHT")
            acc_tile = alloc_tile_buffer(acc_element_dtype, [m, n], space="ACC")
            ops.extract(lhs_mat, c0, c0, lhs_tile)
            ops.move_tile(rhs_mat, rhs_tile)
            ops.matmul(lhs_tile, rhs_tile, acc_tile)
            store_tile(
                acc_tile,
                slice_tensor(
                    out, offsets=[0, 0], sizes=[m, n], dtype=acc_element_dtype
                ),
            )

    return a5_cube_matmul


KERNEL_BUILDERS = {
    "a5_elementwise_add": build_elementwise_add,
    "a5_hivm_vadd_demo": build_hivm_vadd_demo,
    "a5_vector_copy": build_vector_copy,
    "a5_cube_matmul": build_cube_matmul,
}

HIVM_LLVM_KERNELS = {
    "a5_hivm_vadd_demo",
    "a5_vector_copy",
}


__all__ = [
    "KERNEL_BUILDERS",
    "HIVM_LLVM_KERNELS",
    "build_cube_matmul",
    "build_elementwise_add",
    "build_hivm_vadd_demo",
    "build_vector_copy",
    "build_mxfp8_matmul",
    "build_templated_elementwise_add",
]
