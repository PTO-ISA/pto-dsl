import pytest
from mlir.ir import IndexType, IntegerType

import ptodsl.language as pto
from ptodsl import to_ir_module
from ptodsl.lib import a5
from scripts.generate_a5_pto import emit_kernels


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
    assert a5.tunary.trsqrt is a5.trsqrt
    assert a5.texpand.trow_expand is a5.trow_expand
    assert a5.treduce.trow_sum is a5.trow_sum
    assert a5.tsort.tgather is a5.tgather


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
                _slice_tensor(src_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.float32),
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
                _slice_tensor(src_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32),
                _slice_tensor(idx_view, offsets=[0, 0], sizes=[1, 64], dtype=uint32()),
                _slice_tensor(dst_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32),
                dtype=pto.float32,
                index_dtype=uint32(),
                shape=[1, 64],
            )

    text = str(a5_tgather)

    assert "func.func @a5_tgather" in text
    assert "pto.vgather2" in text
    assert "pto.vsts" in text
    assert "pto.tgather" not in text


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
                _slice_tensor(src_view, offsets=[0, 0], sizes=[32, 1], dtype=pto.float32),
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
                _slice_tensor(base_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32),
                _slice_tensor(scale_view, offsets=[0, 0], sizes=[32, 1], dtype=pto.float32),
                _slice_tensor(dst_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32),
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
                _slice_tensor(src_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32),
                _slice_tensor(dst_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32),
                dtype=pto.float32,
                shape=[1, 64],
            )

    text = str(a5_trsqrt)

    assert "func.func @a5_trsqrt" in text
    assert "pto.vsqrt" in text
    assert "pto.vrec" in text
    assert "pto.trsqrt" not in text


@pytest.mark.parametrize(
    ("helper_name", "reduce_op", "combine_op", "tile_op"),
    [
        ("trow_sum", "pto.vcadd", "pto.vadd", "pto.trowsum"),
        ("trow_max", "pto.vcmax", "pto.vmax", "pto.trowmax"),
        ("trow_min", "pto.vcmin", "pto.vmin", "pto.trowmin"),
    ],
)
def test_a5_trow_reduce_emits_reduction_micro_ops(
    helper_name, reduce_op, combine_op, tile_op
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
                _slice_tensor(src_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32),
                _slice_tensor(dst_view, offsets=[0, 0], sizes=[32, 1], dtype=pto.float32),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_trow_reduce)

    assert reduce_op in text
    assert combine_op in text
    assert 'dist = "ONEPT_B32"' in text
    assert tile_op not in text


@pytest.mark.parametrize(
    ("helper_name", "reduce_op", "tile_op", "impl"),
    [
        ("tcol_sum", "pto.vadd", "pto.tcolsum", a5.VF_IMPL_1D_POST_UPDATE),
        ("tcol_max", "pto.vmax", "pto.tcolmax", a5.VF_IMPL_1D_NO_POST_UPDATE),
        ("tcol_min", "pto.vmin", "pto.tcolmin", a5.VF_IMPL_1D_POST_UPDATE),
    ],
)
def test_a5_tcol_reduce_emits_template_lowering(
    helper_name, reduce_op, tile_op, impl
):
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
                _slice_tensor(src_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32),
                _slice_tensor(dst_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.float32),
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
                _slice_tensor(src_view, offsets=[0, 0], sizes=[1, 64], dtype=pto.float32),
                _slice_tensor(idx_view, offsets=[0, 0], sizes=[1, 64], dtype=uint32()),
                _slice_tensor(dst_view, offsets=[0, 0], sizes=[1, 128], dtype=pto.float32),
                dtype=pto.float32,
                shape=[1, 64],
            )

    text = str(a5_tsort32)

    assert "func.func @a5_tsort32" in text
    assert "pto.vbitsort" in text
    assert "pto.tsort32" not in text


def test_a5_tmrgsort_emits_vmrgsort4():
    def meta_data():
        return {"ptr_t": pto.PtrType(pto.float32)}

    @to_ir_module(meta_data=meta_data)
    def a5_tmrgsort(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = _make_tensor(src, shape=[1, 256], dtype=pto.float32)
        dst_view = _make_tensor(dst, shape=[1, 256], dtype=pto.float32)
        with pto.vector_section():
            a5.tmrgsort(
                _slice_tensor(src_view, offsets=[0, 0], sizes=[1, 256], dtype=pto.float32),
                _slice_tensor(dst_view, offsets=[0, 0], sizes=[1, 256], dtype=pto.float32),
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
        "a5_vector_copy.pto",
    ]

    for path in generated:
        text = path.read_text(encoding="utf-8")
        assert "func.func @" in text


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
                    _slice_tensor(lhs, offsets=[0, 0], sizes=[32, 32], dtype=pto.float16),
                    _slice_tensor(rhs, offsets=[0, 0], sizes=[32, 32], dtype=pto.float16),
                    _slice_tensor(out, offsets=[0, 0], sizes=[32, 32], dtype=pto.float16),
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
                    _slice_tensor(src_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.float32),
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
                    _slice_tensor(src_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.float32),
                    _slice_tensor(dst_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.float32),
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
                    _slice_tensor(src_view, offsets=[0, 0], sizes=[32, 32], dtype=pto.bool),
                    _slice_tensor(dst_view, offsets=[0, 0], sizes=[1, 32], dtype=pto.bool),
                    dtype=pto.bool,
                    shape=[32, 32],
                )
