from mlir.dialects import pto as _raw_pto
from mlir.ir import IndexType

from ... import Constexpr, pto, scalar as s, to_ir_module
from ...language import make_mxfp8
from . import ops


def build_elementwise_add(*, rows=32, cols=32, tile_rows=32, tile_cols=32, dtype=None):
    dtype = pto.float32 if dtype is None else dtype

    def meta_data():
        return {
            "ptr_t": pto.ptr(dtype),
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
        lhs = pto.make_tensor(src0, shape=[n_rows, n_cols], dtype=dtype)
        rhs = pto.make_tensor(src1, shape=[n_rows, n_cols], dtype=dtype)
        out = pto.make_tensor(dst, shape=[n_rows, n_cols], dtype=dtype)

        lhs_tile = lhs.slice([0, 0], [tile_rows, tile_cols])
        rhs_tile = rhs.slice([0, 0], [tile_rows, tile_cols])
        out_tile = out.slice([0, 0], [tile_rows, tile_cols])

        with pto.vector_section():
            ops.add_micro(
                lhs_tile,
                rhs_tile,
                out_tile,
                dtype=dtype,
                shape=[tile_rows, tile_cols],
            )

    return a5_elementwise_add


def build_templated_elementwise_add(*, dtype=None):
    dtype = pto.float32 if dtype is None else dtype

    def meta_data(ROWS=32, COLS=32):
        return {
            "ptr_t": pto.ptr(dtype),
            "shape": [ROWS, COLS],
        }

    @to_ir_module(meta_data=meta_data)
    def a5_templated_elementwise_add(
        src0: "ptr_t",
        src1: "ptr_t",
        dst: "ptr_t",
        ROWS: Constexpr[int] = 32,
        COLS: Constexpr[int] = 32,
        VF_IMPL: Constexpr[str] = ops.VF_IMPL_DEFAULT,
    ) -> None:
        lhs = pto.make_tensor(src0, shape=shape, dtype=dtype)
        rhs = pto.make_tensor(src1, shape=shape, dtype=dtype)
        out = pto.make_tensor(dst, shape=shape, dtype=dtype)

        with pto.vector_section():
            ops.add_micro(
                lhs.slice([0, 0], shape),
                rhs.slice([0, 0], shape),
                out.slice([0, 0], shape),
                dtype=dtype,
                shape=shape,
                impl=VF_IMPL,
            )

    return a5_templated_elementwise_add


def build_micro_vector_copy(*, lanes=64, dtype=None):
    dtype = pto.float32 if dtype is None else dtype

    def meta_data():
        return {
            "ptr_t": pto.ptr(dtype, space="VEC"),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_micro_vector_copy(src: "ptr_t", dst: "ptr_t", offset: "index_t") -> None:
        with pto.vector_section():
            ops.vector_copy(src, dst, offset, lanes=lanes, dtype=dtype)

    return a5_micro_vector_copy


def build_mxfp8_matmul(*, m=16, k=64, n=32, lhs_variant="e5m2", rhs_variant="e5m2"):
    mx = make_mxfp8(lhs=lhs_variant, rhs=rhs_variant)
    scale_k = mx.scale_k(k)

    def meta_data():
        return {
            "ptr_lhs": pto.ptr(mx.lhs),
            "ptr_rhs": pto.ptr(mx.rhs),
            "ptr_scale": pto.ptr(mx.scale),
            "ptr_bias": pto.ptr(mx.acc),
            "ptr_out": pto.ptr(mx.acc),
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
        lhs = pto.make_tensor(lhs_ptr, shape=[m, k], dtype=mx.lhs)
        rhs = pto.make_tensor(rhs_ptr, shape=[k, n], dtype=mx.rhs)
        lhs_scale = pto.make_tensor(lhs_scale_ptr, shape=[m, scale_k], dtype=mx.scale)
        rhs_scale = pto.make_tensor(rhs_scale_ptr, shape=[scale_k, n], dtype=mx.scale)
        bias = pto.make_tensor(bias_ptr, shape=[1, n], dtype=mx.acc)
        out = pto.make_tensor(out_ptr, shape=[m, n], dtype=mx.acc)

        with pto.cube_section():
            lhs_tile = ops.load_tile(
                lhs.slice([0, 0], [m, k]), dtype=mx.lhs, shape=[m, k], space="LEFT"
            )
            rhs_tile = ops.load_tile(
                rhs.slice([0, 0], [k, n]), dtype=mx.rhs, shape=[k, n], space="RIGHT"
            )
            lhs_scale_tile = ops.load_tile(
                lhs_scale.slice([0, 0], [m, scale_k]),
                dtype=mx.scale,
                shape=[m, scale_k],
                space="SCALING",
                config=pto.TileBufConfig(
                    blayout="RowMajor",
                    slayout="RowMajor",
                    s_fractal_size=_raw_pto.TileConfig.fractalMxSize,
                ),
            )
            rhs_scale_tile = ops.load_tile(
                rhs_scale.slice([0, 0], [scale_k, n]),
                dtype=mx.scale,
                shape=[scale_k, n],
                space="SCALING",
                config=pto.TileBufConfig(
                    blayout="ColMajor",
                    slayout="ColMajor",
                    s_fractal_size=_raw_pto.TileConfig.fractalMxSize,
                ),
            )
            bias_tile = ops.load_tile(
                bias.slice([0, 0], [1, n]), dtype=mx.acc, shape=[1, n], space="BIAS"
            )
            acc_tile = pto.make_tile_buffer(mx.acc, [m, n], space="ACC").alloc()
            ops.matmul_mx_bias(
                lhs_tile,
                lhs_scale_tile,
                rhs_tile,
                rhs_scale_tile,
                bias_tile,
                acc_tile,
            )
            ops.store_tile(acc_tile, out.slice([0, 0], [m, n]))

    return a5_mxfp8_matmul


def build_cube_matmul(
    *, m=16, k=32, n=16, lhs_dtype=None, rhs_dtype=None, acc_dtype=None
):
    lhs_dtype = pto.float16 if lhs_dtype is None else lhs_dtype
    rhs_dtype = pto.float16 if rhs_dtype is None else rhs_dtype
    acc_dtype = pto.float32 if acc_dtype is None else acc_dtype

    def meta_data():
        return {
            "ptr_lhs": pto.ptr(lhs_dtype),
            "ptr_rhs": pto.ptr(rhs_dtype),
            "ptr_out": pto.ptr(acc_dtype),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_cube_matmul(
        lhs_ptr: "ptr_lhs", rhs_ptr: "ptr_rhs", out_ptr: "ptr_out"
    ) -> None:
        c0 = s.const(0)
        lhs = pto.make_tensor(lhs_ptr, shape=[m, k], dtype=lhs_dtype)
        rhs = pto.make_tensor(rhs_ptr, shape=[k, n], dtype=rhs_dtype)
        out = pto.make_tensor(out_ptr, shape=[m, n], dtype=acc_dtype)

        with pto.cube_section():
            lhs_mat = ops.load_tile(
                lhs.slice([0, 0], [m, k]), dtype=lhs_dtype, shape=[m, k], space="MAT"
            )
            rhs_mat = ops.load_tile(
                rhs.slice([0, 0], [k, n]), dtype=rhs_dtype, shape=[k, n], space="MAT"
            )
            lhs_tile = pto.make_tile_buffer(lhs_dtype, [m, k], space="LEFT").alloc()
            rhs_tile = pto.make_tile_buffer(rhs_dtype, [k, n], space="RIGHT").alloc()
            acc_tile = pto.make_tile_buffer(acc_dtype, [m, n], space="ACC").alloc()
            ops.extract(lhs_mat, c0, c0, lhs_tile)
            ops.move_tile(rhs_mat, rhs_tile)
            ops.matmul(lhs_tile, rhs_tile, acc_tile)
            ops.store_tile(acc_tile, out.slice([0, 0], [m, n]))

    return a5_cube_matmul


KERNEL_BUILDERS = {
    "a5_elementwise_add": build_elementwise_add,
    "a5_micro_vector_copy": build_micro_vector_copy,
    "a5_cube_matmul": build_cube_matmul,
}


__all__ = [
    "KERNEL_BUILDERS",
    "build_cube_matmul",
    "build_elementwise_add",
    "build_micro_vector_copy",
    "build_mxfp8_matmul",
    "build_templated_elementwise_add",
]
