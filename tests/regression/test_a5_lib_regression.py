from pathlib import Path

import pytest
from mlir.dialects import arith
from mlir.ir import IndexType, IntegerType

import ptodsl.language as pto
import ptodsl.pto as public_pto
from ptodsl import to_ir_module
from ptodsl.lib import a5
from scripts.generate_a5_pto import emit_kernels

_PTOAS_BIN = (
    Path(__file__).resolve().parents[3]
    / "PTOAS"
    / "build-src312"
    / "tools"
    / "ptoas"
    / "ptoas"
)


def _index(value):
    return pto.const(value) if isinstance(value, int) else value


def _row_major_strides(shape):
    strides = [None] * len(shape)
    stride = pto.const(1)
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        dim = shape[index]
        stride = stride * _index(dim)
    return strides


def _make_tensor(ptr_value, *, shape, dtype):
    return pto.as_tensor(
        pto.TensorType(rank=len(shape), dtype=dtype),
        ptr=ptr_value,
        shape=[_index(dim) for dim in shape],
        strides=_row_major_strides(shape),
    )


def _slice_tensor(source, *, offsets, sizes, dtype):
    return pto.slice_view(
        pto.SubTensorType(shape=sizes, dtype=dtype),
        source=source,
        offsets=[_index(offset) for offset in offsets],
        sizes=[_index(size) for size in sizes],
    )


def test_a5_split_modules_are_publicly_exposed():
    assert a5.tbinary.tadd is a5.tadd
    assert a5.tbinary.tprelu is a5.tprelu
    assert a5.tscalar.tadds is a5.tadds
    assert a5.tscalar.texpands is a5.texpands
    assert a5.tunary.trsqrt is a5.trsqrt
    assert a5.texpand.trow_expand is a5.trow_expand
    assert a5.treduce.trow_sum is a5.trow_sum
    assert a5.tindex.tgather is a5.tgather
    assert a5.tindex.tgatherb is a5.tgatherb
    assert a5.tindex.tscatter is a5.tscatter
    assert a5.tselect.tsel is a5.tsel
    assert a5.tselect.tsels is a5.tsels


def test_public_pto_ptr_supports_explicit_memory_spaces():
    def meta_data():
        return {"vec_ptr_t": public_pto.ptr(pto.float32, space="VEC")}

    @to_ir_module(meta_data=meta_data)
    def ptr_surface_demo(arg: "vec_ptr_t") -> None:
        return None

    assert "!pto.ptr<f32, ub>" in str(ptr_surface_demo)


def test_a5_elementwise_add_kernel_emits_tile_flow():
    text = str(a5.build_elementwise_add())

    assert "func.func @a5_elementwise_add" in text
    assert "pto.make_tensor_view" in text
    assert "pto.tload" in text
    assert "pto.vlds" in text
    assert "pto.vadd" in text
    assert "pto.vsts" in text
    assert "pto.tadd" not in text
    assert "pto.tstore" in text


def test_a5_templated_elementwise_add_specializes_constexpr_impl():
    specializer = a5.build_templated_elementwise_add()
    text = str(
        specializer(
            ROWS=8,
            COLS=64,
            VF_IMPL=a5.VF_IMPL_1D_POST_UPDATE,
        )
    )

    assert "func.func @a5_templated_elementwise_add(%arg0" in text
    assert "ROWS" not in text
    assert "COLS" not in text
    assert "VF_IMPL" not in text
    assert "scf.if" not in text
    assert "pto.vlds_post" in text
    assert "pto.vsts_post" in text
    assert "pto.tadd" not in text


def test_a5_vector_copy_emits_vector_opcodes():
    text = str(a5.build_vector_copy())

    assert "func.func @a5_vector_copy" in text
    assert "pto.pset_b32" in text
    assert "pto.vlds" in text
    assert "pto.vsts" in text


def test_a5_tadd_separates_tile_shape_from_dynamic_valid_bounds():
    def meta_data():
        return {
            "ptr_t": pto.PtrType(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_tadd_dynamic_valid(
        src0: "ptr_t",
        src1: "ptr_t",
        dst: "ptr_t",
        valid_row: "index_t",
        valid_col: "index_t",
    ) -> None:
        lhs = _make_tensor(src0, shape=[8, 64], dtype=pto.float32)
        rhs = _make_tensor(src1, shape=[8, 64], dtype=pto.float32)
        out = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.tadd(
                _slice_tensor(lhs, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32),
                _slice_tensor(rhs, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32),
                _slice_tensor(out, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32),
                dtype=pto.float32,
                tile_shape=[8, 64],
                valid_row=valid_row,
                valid_col=valid_col,
            )

    text = str(a5_tadd_dynamic_valid)

    assert "!pto.tile_buf<vec, 8x64xf32, valid=?x?>" in text
    assert "valid_row = %arg3 valid_col = %arg4" in text
    assert "pto.vadd" in text
    assert "pto.tadd" not in text


def test_a5_trow_sum_uses_dynamic_valid_bounds_and_masked_reduction():
    def meta_data():
        return {
            "ptr_src": pto.PtrType(pto.float32),
            "ptr_dst": pto.PtrType(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_trow_sum_dynamic_valid(
        src: "ptr_src", dst: "ptr_dst", valid_row: "index_t", valid_col: "index_t"
    ) -> None:
        src_view = _make_tensor(src, shape=[8, 64], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[8, 1], dtype=pto.float32)
        with pto.vector_section():
            a5.trow_sum(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[8, 1], dtype=pto.float32
                ),
                dtype=pto.float32,
                tile_shape=[8, 64],
                valid_row=valid_row,
                valid_col=valid_col,
            )

    text = str(a5_trow_sum_dynamic_valid)

    assert "!pto.tile_buf<vec, 8x64xf32, valid=?x1>" in text
    assert "valid_row = %arg2" in text
    assert "pto.vsel" in text
    assert "pto.vcadd" in text
    assert 'dist = "ONEPT_B32"' in text


def test_a5_tcol_expand_emits_broadcast_micro_ops():
    def meta_data():
        return {
            "ptr_t": pto.PtrType(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_tcol_expand(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = _make_tensor(src, shape=[1, 32], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[32, 32], dtype=pto.float32)
        with pto.vector_section():
            a5.tcol_expand(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.float32
                ),
                _slice_tensor(
                    dst_view,
                    offsets=[0, 0],
                    sizes=[32, 32],
                    dtype=pto.float32,
                ),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_tcol_expand)

    assert "func.func @a5_tcol_expand" in text
    assert "pto.vlds" in text
    assert "pto.vsts" in text
    assert "pto.tcolexpand" not in text


def test_a5_tgather_emits_indexed_gather_opcodes():
    def uint32():
        return IntegerType.get_unsigned(32)

    def meta_data():
        return {
            "ptr_src": pto.PtrType(pto.float32),
            "ptr_idx": pto.PtrType(uint32()),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_tgather(src: "ptr_src", idx: "ptr_idx", dst: "ptr_src") -> None:
        src_view = _make_tensor(src, shape=[1, 64], dtype=pto.float32)
        idx_view = _make_tensor(idx, shape=[1, 64], dtype=uint32())
        dst_view = _make_tensor(dst, shape=[1, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.tgather(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                ),
                _slice_tensor(idx_view, offsets=[0, 0], sizes=[1, 64], dtype=uint32()),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                ),
                dtype=pto.float32,
                index_dtype=uint32(),
                shape=[1, 64],
            )

    text = str(a5_tgather)

    assert "func.func @a5_tgather" in text
    assert "pto.vgather2" in text
    assert "pto.vsts" in text
    assert "pto.tgather" not in text


def test_a5_tgather_supports_dynamic_valid_bounds():
    def uint32():
        return IntegerType.get_unsigned(32)

    def meta_data():
        return {
            "ptr_src": pto.PtrType(pto.float32),
            "ptr_idx": pto.PtrType(uint32()),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_tgather_dynamic_valid(
        src: "ptr_src",
        idx: "ptr_idx",
        dst: "ptr_src",
        valid_row: "index_t",
        valid_col: "index_t",
    ) -> None:
        src_view = _make_tensor(src, shape=[8, 64], dtype=pto.float32)
        idx_view = _make_tensor(idx, shape=[8, 64], dtype=uint32())
        dst_view = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.tgather(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                _slice_tensor(idx_view, offsets=[0, 0], sizes=[8, 64], dtype=uint32()),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                dtype=pto.float32,
                index_dtype=uint32(),
                tile_shape=[8, 64],
                valid_row=valid_row,
                valid_col=valid_col,
            )

    text = str(a5_tgather_dynamic_valid)

    assert "!pto.tile_buf<vec, 8x64xf32, valid=?x?>" in text
    assert "valid_row = %arg3 valid_col = %arg4" in text
    assert "pto.vgather2" in text
    assert "arith.index_cast" in text


def test_a5_tgatherb_emits_byte_gather_micro_opcodes():
    def uint32():
        return IntegerType.get_unsigned(32)

    def meta_data():
        return {
            "ptr_src": pto.PtrType(pto.float32),
            "ptr_idx": pto.PtrType(uint32()),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_tgatherb(src: "ptr_src", idx: "ptr_idx", dst: "ptr_src") -> None:
        src_view = _make_tensor(src, shape=[8, 64], dtype=pto.float32)
        idx_view = _make_tensor(idx, shape=[8, 64], dtype=uint32())
        dst_view = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.tgatherb(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                _slice_tensor(idx_view, offsets=[0, 0], sizes=[8, 64], dtype=uint32()),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                dtype=pto.float32,
                index_dtype=uint32(),
                shape=[8, 64],
            )

    text = str(a5_tgatherb)

    assert "func.func @a5_tgatherb" in text
    assert "pto.vgatherb" in text
    assert "pto.tgatherb" not in text


def test_a5_tscatter_emits_zero_fill_then_vscatter():
    def uint32():
        return IntegerType.get_unsigned(32)

    def meta_data():
        return {
            "ptr_src": pto.PtrType(pto.float32),
            "ptr_idx": pto.PtrType(uint32()),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_tscatter(src: "ptr_src", idx: "ptr_idx", dst: "ptr_src") -> None:
        src_view = _make_tensor(src, shape=[8, 64], dtype=pto.float32)
        idx_view = _make_tensor(idx, shape=[8, 64], dtype=uint32())
        dst_view = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.tscatter(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                _slice_tensor(idx_view, offsets=[0, 0], sizes=[8, 64], dtype=uint32()),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                dtype=pto.float32,
                index_dtype=uint32(),
                shape=[8, 64],
            )

    text = str(a5_tscatter)

    assert "func.func @a5_tscatter" in text
    assert "pto.vbr" in text
    assert "pto.vscatter" in text
    assert "pto.tscatter" not in text


def test_a5_trow_expand_emits_broadcast_micro_ops():
    def meta_data():
        return {
            "ptr_t": pto.PtrType(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_trow_expand(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = _make_tensor(src, shape=[32, 1], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[32, 32], dtype=pto.float32)
        with pto.vector_section():
            a5.trow_expand(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[32, 1], dtype=pto.float32
                ),
                _slice_tensor(
                    dst_view,
                    offsets=[0, 0],
                    sizes=[32, 32],
                    dtype=pto.float32,
                ),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_trow_expand)

    assert "func.func @a5_trow_expand" in text
    assert "pto.vldas" in text
    assert "pto.vldus" in text
    assert "pto.vdup" in text
    assert "pto.vsts" in text
    assert "pto.trowexpand" not in text


def test_a5_trow_expand_mul_emits_broadcast_compute_micro_ops():
    def meta_data():
        return {
            "ptr_t": pto.PtrType(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_trow_expand_mul(base: "ptr_t", scale: "ptr_t", dst: "ptr_t") -> None:
        base_view = _make_tensor(base, shape=[32, 32], dtype=pto.float32)
        scale_view = _make_tensor(scale, shape=[32, 1], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[32, 32], dtype=pto.float32)
        with pto.vector_section():
            a5.trow_expand_mul(
                _slice_tensor(
                    base_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32
                ),
                _slice_tensor(
                    scale_view, offsets=[0, 0], sizes=[32, 1], dtype=pto.float32
                ),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_trow_expand_mul)

    assert "func.func @a5_trow_expand_mul" in text
    assert "pto.vldas" in text
    assert "pto.vldus" in text
    assert "pto.vdup" in text
    assert "pto.vmul" in text
    assert "pto.vsts" in text
    assert "pto.trowexpandmul" not in text


def test_a5_trsqrt_emits_vsqrt_then_vrec():
    def meta_data():
        return {
            "ptr_t": pto.PtrType(pto.float32),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_trsqrt(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = _make_tensor(src, shape=[1, 64], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[1, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.trsqrt(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                ),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[1, 64],
            )

    text = str(a5_trsqrt)

    assert "func.func @a5_trsqrt" in text
    assert "pto.vsqrt" in text
    assert "pto.vrec" in text
    assert "pto.trsqrt" not in text


@pytest.mark.parametrize(
    ("helper_name", "dtype", "micro_op", "tile_op"),
    [
        ("tmax", "float32", "pto.vmax", "pto.tmax"),
        ("tmin", "float32", "pto.vmin", "pto.tmin"),
        ("tand", "int32", "pto.vand", "pto.tand"),
        ("txor", "int32", "pto.vxor", "pto.txor"),
        ("tshl", "int32", "pto.vshl", "pto.tshl"),
        ("tshr", "int32", "pto.vshr", "pto.tshr"),
    ],
)
def test_a5_binary_header_helpers_emit_micro_opcodes(
    helper_name, dtype, micro_op, tile_op
):
    def meta_data():
        mlir_dtype = getattr(pto, dtype)
        return {"ptr_t": pto.PtrType(mlir_dtype)}

    helper = getattr(a5, helper_name)

    @to_ir_module(meta_data=meta_data)
    def a5_binary_helper(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t") -> None:
        mlir_dtype = getattr(pto, dtype)
        lhs = _make_tensor(src0, shape=[1, 64], dtype=mlir_dtype)
        rhs = _make_tensor(src1, shape=[1, 64], dtype=mlir_dtype)
        out = _make_tensor(dst, shape=[1, 64], dtype=mlir_dtype)
        with pto.vector_section():
            helper(
                _slice_tensor(lhs, offsets=[0, 0], sizes=[1, 64], dtype=mlir_dtype),
                _slice_tensor(rhs, offsets=[0, 0], sizes=[1, 64], dtype=mlir_dtype),
                _slice_tensor(out, offsets=[0, 0], sizes=[1, 64], dtype=mlir_dtype),
                dtype=mlir_dtype,
                shape=[1, 64],
            )

    text = str(a5_binary_helper)

    assert micro_op in text
    assert tile_op not in text


@pytest.mark.parametrize(
    ("helper_name", "dtype", "scalar_value", "micro_op", "tile_op"),
    [
        ("tadds", "float32", 2.0, "pto.vadds", "pto.tadds"),
        ("tsubs", "float32", 2.0, "pto.vsub", "pto.tsubs"),
        ("tmaxs", "float32", 2.0, "pto.vmaxs", "pto.tmaxs"),
        ("tands", "int32", 7, "pto.vand", "pto.tands"),
        ("tshls", "int32", 1, "pto.vshls", "pto.tshls"),
        ("tlrelu", "float32", 0.1, "pto.vlrelu", "pto.tlrelu"),
    ],
)
def test_a5_scalar_header_helpers_emit_micro_opcodes(
    helper_name, dtype, scalar_value, micro_op, tile_op
):
    def meta_data():
        mlir_dtype = getattr(pto, dtype)
        return {"ptr_t": pto.PtrType(mlir_dtype)}

    helper = getattr(a5, helper_name)

    @to_ir_module(meta_data=meta_data)
    def a5_scalar_helper(src: "ptr_t", dst: "ptr_t") -> None:
        mlir_dtype = getattr(pto, dtype)
        src_view = _make_tensor(src, shape=[1, 64], dtype=mlir_dtype)
        dst_view = _make_tensor(dst, shape=[1, 64], dtype=mlir_dtype)
        scalar = arith.ConstantOp(mlir_dtype, scalar_value).result
        with pto.vector_section():
            helper(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[1, 64], dtype=mlir_dtype
                ),
                scalar,
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[1, 64], dtype=mlir_dtype
                ),
                dtype=mlir_dtype,
                shape=[1, 64],
            )

    text = str(a5_scalar_helper)

    assert micro_op in text
    assert tile_op not in text


def test_a5_taxpy_emits_vmula():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    @to_ir_module(meta_data=meta_data)
    def a5_taxpy(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = _make_tensor(src, shape=[8, 64], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
        scalar = arith.ConstantOp(pto.float32, 0.5).result
        with pto.vector_section():
            a5.taxpy(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                scalar,
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[8, 64],
            )

    text = str(a5_taxpy)

    assert "pto.vmula" in text
    assert "pto.taxpy" not in text


def test_a5_texpands_emits_vbr_and_vsts():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    @to_ir_module(meta_data=meta_data)
    def a5_texpands(dst: "ptr_t") -> None:
        dst_view = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
        scalar = arith.ConstantOp(pto.float32, 1.5).result
        with pto.vector_section():
            a5.texpands(
                scalar,
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[8, 64],
            )

    text = str(a5_texpands)

    assert "func.func @a5_texpands" in text
    assert "pto.vbr" in text
    assert "pto.vsts" in text
    assert "pto.texpands" not in text


def test_a5_tprelu_emits_vcmps_vmul_and_vsel():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    @to_ir_module(meta_data=meta_data)
    def a5_tprelu(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t") -> None:
        lhs = _make_tensor(src0, shape=[8, 64], dtype=pto.float32)
        rhs = _make_tensor(src1, shape=[8, 64], dtype=pto.float32)
        out = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.tprelu(
                _slice_tensor(lhs, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32),
                _slice_tensor(rhs, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32),
                _slice_tensor(out, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32),
                dtype=pto.float32,
                shape=[8, 64],
            )

    text = str(a5_tprelu)

    assert "func.func @a5_tprelu" in text
    assert "pto.vcmps" in text
    assert "pto.vmul" in text
    assert "pto.vsel" in text
    assert "pto.tprelu" not in text


def test_a5_tsel_emits_plds_pintlv_and_vsel():
    def meta_data():
        return {
            "ptr_mask": pto.PtrType(pto.int8),
            "ptr_data": pto.PtrType(pto.float32),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_tsel(
        mask: "ptr_mask", src0: "ptr_data", src1: "ptr_data", dst: "ptr_data"
    ) -> None:
        mask_view = _make_tensor(mask, shape=[1, 128], dtype=pto.int8)
        lhs = _make_tensor(src0, shape=[1, 128], dtype=pto.float32)
        rhs = _make_tensor(src1, shape=[1, 128], dtype=pto.float32)
        out = _make_tensor(dst, shape=[1, 128], dtype=pto.float32)
        with pto.vector_section():
            a5.tsel(
                _slice_tensor(
                    mask_view, offsets=[0, 0], sizes=[1, 128], dtype=pto.int8
                ),
                _slice_tensor(lhs, offsets=[0, 0], sizes=[1, 128], dtype=pto.float32),
                _slice_tensor(rhs, offsets=[0, 0], sizes=[1, 128], dtype=pto.float32),
                _slice_tensor(out, offsets=[0, 0], sizes=[1, 128], dtype=pto.float32),
                dtype=pto.float32,
                shape=[1, 128],
            )

    text = str(a5_tsel)

    assert "func.func @a5_tsel" in text
    assert "pto.pset_b16" in text
    assert "pto.plds" in text
    assert "pto.pintlv_b16" in text
    assert "pto.vsel" in text
    assert "pto.tsel" not in text


def test_a5_tsels_emits_vcmps_vdup_and_vsel():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    @to_ir_module(meta_data=meta_data)
    def a5_tsels(mask: "ptr_t", src: "ptr_t", dst: "ptr_t") -> None:
        mask_view = _make_tensor(mask, shape=[1, 64], dtype=pto.float32)
        src_view = _make_tensor(src, shape=[1, 64], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[1, 64], dtype=pto.float32)
        scalar = arith.ConstantOp(pto.float32, 3.0).result
        with pto.vector_section():
            a5.tsels(
                _slice_tensor(
                    mask_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                ),
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                ),
                scalar,
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[1, 64],
            )

    text = str(a5_tsels)

    assert "func.func @a5_tsels" in text
    assert "pto.vcmps" in text
    assert "pto.vdup" in text
    assert "pto.vsel" in text
    assert "pto.tsels" not in text


@pytest.mark.parametrize(
    ("helper_name", "micro_op", "tile_op"),
    [
        ("trow_expand_add", "pto.vadd", "pto.trowexpandadd"),
        ("trow_expand_max", "pto.vmax", "pto.trowexpandmax"),
        ("tcol_expand_add", "pto.vadd", "pto.tcolexpandadd"),
        ("tcol_expand_max", "pto.vmax", "pto.tcolexpandmax"),
    ],
)
def test_a5_expand_header_helpers_emit_micro_opcodes(helper_name, micro_op, tile_op):
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    helper = getattr(a5, helper_name)

    @to_ir_module(meta_data=meta_data)
    def a5_expand_helper(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t") -> None:
        with pto.vector_section():
            if helper_name.startswith("trow_"):
                base_view = _make_tensor(src0, shape=[8, 64], dtype=pto.float32)
                expand_view = _make_tensor(src1, shape=[8, 1], dtype=pto.float32)
                out_view = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
                helper(
                    _slice_tensor(
                        base_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                    ),
                    _slice_tensor(
                        expand_view, offsets=[0, 0], sizes=[8, 1], dtype=pto.float32
                    ),
                    _slice_tensor(
                        out_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                    ),
                    dtype=pto.float32,
                    shape=[8, 64],
                )
            else:
                base_view = _make_tensor(src0, shape=[8, 64], dtype=pto.float32)
                expand_view = _make_tensor(src1, shape=[1, 64], dtype=pto.float32)
                out_view = _make_tensor(dst, shape=[8, 64], dtype=pto.float32)
                helper(
                    _slice_tensor(
                        base_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                    ),
                    _slice_tensor(
                        expand_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                    ),
                    _slice_tensor(
                        out_view, offsets=[0, 0], sizes=[8, 64], dtype=pto.float32
                    ),
                    dtype=pto.float32,
                    shape=[8, 64],
                )

    text = str(a5_expand_helper)

    assert micro_op in text
    assert tile_op not in text


@pytest.mark.parametrize(
    ("helper_name", "reduce_op", "combine_op", "extra_token", "tile_op"),
    [
        ("trow_sum", "pto.vcadd", "pto.vadd", None, "pto.trowsum"),
        ("trow_max", "pto.vcmax", "pto.vmax", None, "pto.trowmax"),
        ("trow_min", "pto.vcmin", "pto.vmin", None, "pto.trowmin"),
        ("trow_prod", "pto.vmul", "pto.vmul", "pto.vintlv", "pto.trowprod"),
    ],
)
def test_a5_trow_reduce_emits_reduction_micro_ops(
    helper_name, reduce_op, combine_op, extra_token, tile_op
):
    def meta_data():
        return {
            "ptr_t": pto.PtrType(pto.float32),
            "index_t": IndexType.get(),
        }

    helper = getattr(a5, helper_name)

    @to_ir_module(meta_data=meta_data)
    def a5_trow_reduce(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = _make_tensor(src, shape=[32, 32], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[32, 1], dtype=pto.float32)
        with pto.vector_section():
            helper(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32
                ),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[32, 1], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_trow_reduce)

    assert reduce_op in text
    assert combine_op in text
    if extra_token is not None:
        assert extra_token in text
    assert 'dist = "ONEPT_B32"' in text
    assert tile_op not in text


@pytest.mark.parametrize(
    ("helper_name", "reduce_op", "tile_op", "impl"),
    [
        ("tcol_sum", "pto.vadd", "pto.tcolsum", a5.VF_IMPL_1D_POST_UPDATE),
        ("tcol_max", "pto.vmax", "pto.tcolmax", a5.VF_IMPL_1D_NO_POST_UPDATE),
        ("tcol_min", "pto.vmin", "pto.tcolmin", a5.VF_IMPL_1D_POST_UPDATE),
        ("tcol_prod", "pto.vmul", "pto.tcolprod", a5.VF_IMPL_1D_POST_UPDATE),
    ],
)
def test_a5_tcol_reduce_emits_template_lowering(helper_name, reduce_op, tile_op, impl):
    def meta_data():
        return {
            "ptr_t": pto.PtrType(pto.float32),
        }

    helper = getattr(a5, helper_name)

    @to_ir_module(meta_data=meta_data)
    def a5_tcol_reduce(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = _make_tensor(src, shape=[32, 32], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[1, 32], dtype=pto.float32)
        with pto.vector_section():
            helper(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32
                ),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[32, 32],
                impl=impl,
            )

    text = str(a5_tcol_reduce)

    assert reduce_op in text
    assert tile_op not in text
    if impl == a5.VF_IMPL_1D_POST_UPDATE:
        assert "pto.vlds_post" in text
        assert "pto.vsts_post" in text


def test_a5_tsort32_emits_vbitsort():
    def uint32():
        return IntegerType.get_unsigned(32)

    def meta_data():
        return {
            "ptr_src": pto.PtrType(pto.float32),
            "ptr_idx": pto.PtrType(uint32()),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_tsort32(src: "ptr_src", idx: "ptr_idx", dst: "ptr_src") -> None:
        src_view = _make_tensor(src, shape=[1, 64], dtype=pto.float32)
        idx_view = _make_tensor(idx, shape=[1, 64], dtype=uint32())
        dst_view = _make_tensor(dst, shape=[1, 128], dtype=pto.float32)
        with pto.vector_section():
            a5.tsort32(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                ),
                _slice_tensor(idx_view, offsets=[0, 0], sizes=[1, 64], dtype=uint32()),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[1, 128], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[1, 64],
            )

    text = str(a5_tsort32)

    assert "func.func @a5_tsort32" in text
    assert "pto.vbitsort" in text
    assert "pto.tsort32" not in text


def test_a5_tsort32_rejects_dynamic_valid_input_tiles():
    def uint32():
        return IntegerType.get_unsigned(32)

    def meta_data():
        return {
            "ptr_src": pto.PtrType(pto.float32),
            "ptr_idx": pto.PtrType(uint32()),
            "index_t": IndexType.get(),
        }

    with pytest.raises(
        ValueError,
        match="TSORT32 micro lowering currently requires a fully valid input tile",
    ):

        @to_ir_module(meta_data=meta_data)
        def invalid_tsort32(
            src: "ptr_src",
            idx: "ptr_idx",
            dst: "ptr_src",
            valid_row: "index_t",
            valid_col: "index_t",
        ) -> None:
            src_view = _make_tensor(src, shape=[1, 64], dtype=pto.float32)
            idx_view = _make_tensor(idx, shape=[1, 64], dtype=uint32())
            dst_view = _make_tensor(dst, shape=[1, 128], dtype=pto.float32)
            with pto.vector_section():
                a5.tsort32(
                    _slice_tensor(
                        src_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32
                    ),
                    _slice_tensor(
                        idx_view, offsets=[0, 0], sizes=[1, 64], dtype=uint32()
                    ),
                    _slice_tensor(
                        dst_view, offsets=[0, 0], sizes=[1, 128], dtype=pto.float32
                    ),
                    dtype=pto.float32,
                    tile_shape=[1, 64],
                    valid_row=valid_row,
                    valid_col=valid_col,
                )


def test_a5_tmrgsort_emits_vmrgsort4():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    @to_ir_module(meta_data=meta_data)
    def a5_tmrgsort(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = _make_tensor(src, shape=[1, 256], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[1, 256], dtype=pto.float32)
        with pto.vector_section():
            a5.tmrgsort(
                _slice_tensor(
                    src_view, offsets=[0, 0], sizes=[1, 256], dtype=pto.float32
                ),
                _slice_tensor(
                    dst_view, offsets=[0, 0], sizes=[1, 256], dtype=pto.float32
                ),
                dtype=pto.float32,
                shape=[1, 256],
                block_len=64,
            )

    text = str(a5_tmrgsort)

    assert "func.func @a5_tmrgsort" in text
    assert "pto.vmrgsort4" in text
    assert "pto.tmrgsort" not in text


def test_a5_generation_script_emits_pto_files(tmp_path):
    generated = emit_kernels(output_dir=tmp_path)

    generated_names = sorted(path.name for path in generated)
    assert generated_names == [
        "a5_cube_matmul.pto",
        "a5_elementwise_add.pto",
        "a5_hivm_vadd_demo.pto",
        "a5_vector_copy.pto",
    ]

    for path in generated:
        text = path.read_text(encoding="utf-8")
        assert "func.func @" in text


def test_a5_generation_script_can_emit_hivm_llvm_for_micro_kernel(tmp_path):
    generated = emit_kernels(
        output_dir=tmp_path,
        ptoas_bin=_PTOAS_BIN,
        emit_hivm_llvm=True,
        kernel_names={"a5_hivm_vadd_demo"},
    )

    assert [path.name for path in generated] == ["a5_hivm_vadd_demo.pto"]

    llvm_path = tmp_path / "a5_hivm_vadd_demo.ll"
    assert llvm_path.exists()

    text = llvm_path.read_text(encoding="utf-8")
    assert "llvm.hivm.vldsx1.v64f32" in text
    assert "llvm.hivm.vadd.v64f32.x" in text
    assert "llvm.hivm.vstsx1.v64f32" in text


def test_a5_generation_script_only_emits_hivm_sidecars_for_supported_kernels(tmp_path):
    emit_kernels(output_dir=tmp_path, ptoas_bin=_PTOAS_BIN, emit_hivm_llvm=True)

    assert (tmp_path / "a5_hivm_vadd_demo.ll").exists()
    assert (tmp_path / "a5_vector_copy.ll").exists()
    assert not (tmp_path / "a5_elementwise_add.ll").exists()
    assert not (tmp_path / "a5_cube_matmul.ll").exists()


def test_a5_tadd_rejects_view_dtype_mismatch():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float16)}

    with pytest.raises(
        ValueError, match="TADD input tile src0, src1 and dst tile data type mismatch"
    ):

        @to_ir_module(meta_data=meta_data)
        def invalid_add(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t") -> None:
            lhs = _make_tensor(src0, shape=[32, 32], dtype=pto.float16)
            rhs = _make_tensor(src1, shape=[32, 32], dtype=pto.float16)
            out = _make_tensor(dst, shape=[32, 32], dtype=pto.float16)
            with pto.vector_section():
                a5.tadd(
                    _slice_tensor(
                        lhs, offsets=[0, 0], sizes=[32, 32], dtype=pto.float16
                    ),
                    _slice_tensor(
                        rhs, offsets=[0, 0], sizes=[32, 32], dtype=pto.float16
                    ),
                    _slice_tensor(
                        out, offsets=[0, 0], sizes=[32, 32], dtype=pto.float16
                    ),
                    dtype=pto.float32,
                    shape=[32, 32],
                )


def test_a5_trow_expand_rejects_non_column_source():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    with pytest.raises(
        ValueError, match="TROWEXPAND source valid shape must be \\[rows, 1\\]"
    ):

        @to_ir_module(meta_data=meta_data)
        def invalid_row_expand(src: "ptr_t", dst: "ptr_t") -> None:
            src_view = _make_tensor(src, shape=[1, 32], dtype=pto.float32)
            dst_view = _make_tensor(dst, shape=[32, 32], dtype=pto.float32)
            with pto.vector_section():
                a5.trow_expand(
                    _slice_tensor(
                        src_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.float32
                    ),
                    _slice_tensor(
                        dst_view,
                        offsets=[0, 0],
                        sizes=[32, 32],
                        dtype=pto.float32,
                    ),
                    dtype=pto.float32,
                    shape=[32, 32],
                )


def test_a5_trow_sum_rejects_non_single_column_output():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    with pytest.raises(ValueError, match="use a single-column output tile"):

        @to_ir_module(meta_data=meta_data)
        def invalid_row_reduce(src: "ptr_t", dst: "ptr_t") -> None:
            src_view = _make_tensor(src, shape=[32, 32], dtype=pto.float32)
            dst_view = _make_tensor(dst, shape=[1, 32], dtype=pto.float32)
            with pto.vector_section():
                a5.trow_sum(
                    _slice_tensor(
                        src_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32
                    ),
                    _slice_tensor(
                        dst_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.float32
                    ),
                    dtype=pto.float32,
                    shape=[32, 32],
                )


def test_a5_tcol_sum_rejects_unsupported_dtype():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.bool)}

    with pytest.raises(ValueError, match="TCOLREDUCE input data type is not supported"):

        @to_ir_module(meta_data=meta_data)
        def invalid_col_reduce(src: "ptr_t", dst: "ptr_t") -> None:
            src_view = _make_tensor(src, shape=[32, 32], dtype=pto.bool)
            dst_view = _make_tensor(dst, shape=[1, 32], dtype=pto.bool)
            with pto.vector_section():
                a5.tcol_sum(
                    _slice_tensor(
                        src_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.bool
                    ),
                    _slice_tensor(
                        dst_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.bool
                    ),
                    dtype=pto.bool,
                    shape=[32, 32],
                )
