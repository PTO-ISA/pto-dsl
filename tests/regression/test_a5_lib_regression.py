import pytest
from mlir.ir import IndexType

from ptodsl import pto, to_ir_module
from ptodsl.lib import a5
from scripts.generate_a5_pto import emit_kernels


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


def test_a5_micro_vector_copy_emits_micro_ops():
    text = str(a5.build_micro_vector_copy())

    assert "func.func @a5_micro_vector_copy" in text
    assert "pto.pset_b32" in text
    assert "pto.vlds" in text
    assert "pto.vsts" in text


def test_a5_col_expand_micro_emits_broadcast_micro_ops():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_col_expand_micro(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = pto.make_tensor(src, shape=[1, 32], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[32, 32], dtype=pto.float32)
        with pto.vector_section():
            a5.col_expand_micro(
                src_view.slice([0, 0], [1, 32]),
                dst_view.slice([0, 0], [32, 32]),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_col_expand_micro)

    assert "func.func @a5_col_expand_micro" in text
    assert "pto.vlds" in text
    assert "pto.vsts" in text
    assert "pto.tcolexpand" not in text


def test_a5_gather_micro_emits_indexed_gather_micro_ops():
    def meta_data():
        return {
            "ptr_src": pto.ptr(pto.float32),
            "ptr_idx": pto.ptr(pto.uint32),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_gather_micro(src: "ptr_src", idx: "ptr_idx", dst: "ptr_src") -> None:
        src_view = pto.make_tensor(src, shape=[1, 64], dtype=pto.float32)
        idx_view = pto.make_tensor(idx, shape=[1, 64], dtype=pto.uint32)
        dst_view = pto.make_tensor(dst, shape=[1, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.gather_micro(
                src_view.slice([0, 0], [1, 64]),
                idx_view.slice([0, 0], [1, 64]),
                dst_view.slice([0, 0], [1, 64]),
                dtype=pto.float32,
                index_dtype=pto.uint32,
                shape=[1, 64],
            )

    text = str(a5_gather_micro)

    assert "func.func @a5_gather_micro" in text
    assert "pto.vgather2" in text
    assert "pto.vsts" in text
    assert "pto.tgather" not in text


def test_a5_row_expand_micro_emits_broadcast_micro_ops():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_row_expand_micro(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = pto.make_tensor(src, shape=[32, 1], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[32, 32], dtype=pto.float32)
        with pto.vector_section():
            a5.row_expand_micro(
                src_view.slice([0, 0], [32, 1]),
                dst_view.slice([0, 0], [32, 32]),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_row_expand_micro)

    assert "func.func @a5_row_expand_micro" in text
    assert "pto.vldas" in text
    assert "pto.vldus" in text
    assert "pto.vdup" in text
    assert "pto.vsts" in text
    assert "pto.trowexpand" not in text


def test_a5_row_expand_mul_micro_emits_broadcast_compute_micro_ops():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_row_expand_mul_micro(base: "ptr_t", scale: "ptr_t", dst: "ptr_t") -> None:
        base_view = pto.make_tensor(base, shape=[32, 32], dtype=pto.float32)
        scale_view = pto.make_tensor(scale, shape=[32, 1], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[32, 32], dtype=pto.float32)
        with pto.vector_section():
            a5.row_expand_mul_micro(
                base_view.slice([0, 0], [32, 32]),
                scale_view.slice([0, 0], [32, 1]),
                dst_view.slice([0, 0], [32, 32]),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_row_expand_mul_micro)

    assert "func.func @a5_row_expand_mul_micro" in text
    assert "pto.vldas" in text
    assert "pto.vldus" in text
    assert "pto.vdup" in text
    assert "pto.vmul" in text
    assert "pto.vsts" in text
    assert "pto.trowexpandmul" not in text


def test_a5_rsqrt_micro_emits_vsqrt_then_vrec():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_rsqrt_micro(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = pto.make_tensor(src, shape=[1, 64], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[1, 64], dtype=pto.float32)
        with pto.vector_section():
            a5.rsqrt_micro(
                src_view.slice([0, 0], [1, 64]),
                dst_view.slice([0, 0], [1, 64]),
                dtype=pto.float32,
                shape=[1, 64],
            )

    text = str(a5_rsqrt_micro)

    assert "func.func @a5_rsqrt_micro" in text
    assert "pto.vsqrt" in text
    assert "pto.vrec" in text
    assert "pto.trsqrt" not in text


@pytest.mark.parametrize(
    ("helper_name", "reduce_op", "combine_op", "tile_op"),
    [
        ("row_sum_micro", "pto.vcadd", "pto.vadd", "pto.trowsum"),
        ("row_max_micro", "pto.vcmax", "pto.vmax", "pto.trowmax"),
        ("row_min_micro", "pto.vcmin", "pto.vmin", "pto.trowmin"),
    ],
)
def test_a5_row_reduce_micro_emits_reduction_micro_ops(
    helper_name, reduce_op, combine_op, tile_op
):
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    helper = getattr(a5, helper_name)

    @to_ir_module(meta_data=meta_data)
    def a5_row_reduce_micro(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = pto.make_tensor(src, shape=[32, 32], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[32, 1], dtype=pto.float32)
        with pto.vector_section():
            helper(
                src_view.slice([0, 0], [32, 32]),
                dst_view.slice([0, 0], [32, 1]),
                dtype=pto.float32,
                shape=[32, 32],
            )

    text = str(a5_row_reduce_micro)

    assert reduce_op in text
    assert combine_op in text
    assert 'dist = "ONEPT_B32"' in text
    assert tile_op not in text


@pytest.mark.parametrize(
    ("helper_name", "reduce_op", "tile_op", "impl"),
    [
        ("col_sum_micro", "pto.vadd", "pto.tcolsum", a5.VF_IMPL_1D_POST_UPDATE),
        ("col_max_micro", "pto.vmax", "pto.tcolmax", a5.VF_IMPL_1D_NO_POST_UPDATE),
        ("col_min_micro", "pto.vmin", "pto.tcolmin", a5.VF_IMPL_1D_POST_UPDATE),
    ],
)
def test_a5_col_reduce_micro_emits_template_lowering(helper_name, reduce_op, tile_op, impl):
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
        }

    helper = getattr(a5, helper_name)

    @to_ir_module(meta_data=meta_data)
    def a5_col_reduce_micro(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = pto.make_tensor(src, shape=[32, 32], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[1, 32], dtype=pto.float32)
        with pto.vector_section():
            helper(
                src_view.slice([0, 0], [32, 32]),
                dst_view.slice([0, 0], [1, 32]),
                dtype=pto.float32,
                shape=[32, 32],
                impl=impl,
            )

    text = str(a5_col_reduce_micro)

    assert reduce_op in text
    assert tile_op not in text
    if impl == a5.VF_IMPL_1D_POST_UPDATE:
        assert "pto.vlds_post" in text
        assert "pto.vsts_post" in text


def test_a5_sort32_micro_emits_vbitsort():
    def meta_data():
        return {
            "ptr_src": pto.ptr(pto.float32),
            "ptr_idx": pto.ptr(pto.uint32),
        }

    @to_ir_module(meta_data=meta_data)
    def a5_sort32_micro(src: "ptr_src", idx: "ptr_idx", dst: "ptr_src") -> None:
        src_view = pto.make_tensor(src, shape=[1, 64], dtype=pto.float32)
        idx_view = pto.make_tensor(idx, shape=[1, 64], dtype=pto.uint32)
        dst_view = pto.make_tensor(dst, shape=[1, 128], dtype=pto.float32)
        with pto.vector_section():
            a5.sort32_micro(
                src_view.slice([0, 0], [1, 64]),
                idx_view.slice([0, 0], [1, 64]),
                dst_view.slice([0, 0], [1, 128]),
                dtype=pto.float32,
                shape=[1, 64],
            )

    text = str(a5_sort32_micro)

    assert "func.func @a5_sort32_micro" in text
    assert "pto.vbitsort" in text
    assert "pto.tsort32" not in text


def test_a5_mrgsort_micro_emits_vmrgsort4():
    def meta_data():
        return {"ptr_t": pto.ptr(pto.float32)}

    @to_ir_module(meta_data=meta_data)
    def a5_mrgsort_micro(src: "ptr_t", dst: "ptr_t") -> None:
        src_view = pto.make_tensor(src, shape=[1, 256], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[1, 256], dtype=pto.float32)
        with pto.vector_section():
            a5.mrgsort_micro(
                src_view.slice([0, 0], [1, 256]),
                dst_view.slice([0, 0], [1, 256]),
                dtype=pto.float32,
                shape=[1, 256],
                block_len=64,
            )

    text = str(a5_mrgsort_micro)

    assert "func.func @a5_mrgsort_micro" in text
    assert "pto.vmrgsort4" in text
    assert "pto.tmrgsort" not in text


def test_a5_generation_script_emits_pto_files(tmp_path):
    generated = emit_kernels(output_dir=tmp_path)

    generated_names = sorted(path.name for path in generated)
    assert generated_names == [
        "a5_cube_matmul.pto",
        "a5_elementwise_add.pto",
        "a5_micro_vector_copy.pto",
    ]

    for path in generated:
        text = path.read_text(encoding="utf-8")
        assert "func.func @" in text


def test_a5_add_micro_rejects_view_dtype_mismatch():
    def meta_data():
        return {"ptr_t": pto.ptr(pto.float16)}

    with pytest.raises(ValueError, match="TADD input tile src0, src1 and dst tile data type mismatch"):
        @to_ir_module(meta_data=meta_data)
        def invalid_add(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t") -> None:
            lhs = pto.make_tensor(src0, shape=[32, 32], dtype=pto.float16)
            rhs = pto.make_tensor(src1, shape=[32, 32], dtype=pto.float16)
            out = pto.make_tensor(dst, shape=[32, 32], dtype=pto.float16)
            with pto.vector_section():
                a5.add_micro(
                    lhs.slice([0, 0], [32, 32]),
                    rhs.slice([0, 0], [32, 32]),
                    out.slice([0, 0], [32, 32]),
                    dtype=pto.float32,
                    shape=[32, 32],
                )


def test_a5_row_expand_micro_rejects_non_column_source():
    def meta_data():
        return {"ptr_t": pto.ptr(pto.float32)}

    with pytest.raises(ValueError, match="TROWEXPAND source valid shape must be \\[rows, 1\\]"):
        @to_ir_module(meta_data=meta_data)
        def invalid_row_expand(src: "ptr_t", dst: "ptr_t") -> None:
            src_view = pto.make_tensor(src, shape=[1, 32], dtype=pto.float32)
            dst_view = pto.make_tensor(dst, shape=[32, 32], dtype=pto.float32)
            with pto.vector_section():
                a5.row_expand_micro(
                    src_view.slice([0, 0], [1, 32]),
                    dst_view.slice([0, 0], [32, 32]),
                    dtype=pto.float32,
                    shape=[32, 32],
                )


def test_a5_row_reduce_micro_rejects_non_single_column_output():
    def meta_data():
        return {"ptr_t": pto.ptr(pto.float32)}

    with pytest.raises(ValueError, match="use a single-column output tile"):
        @to_ir_module(meta_data=meta_data)
        def invalid_row_reduce(src: "ptr_t", dst: "ptr_t") -> None:
            src_view = pto.make_tensor(src, shape=[32, 32], dtype=pto.float32)
            dst_view = pto.make_tensor(dst, shape=[1, 32], dtype=pto.float32)
            with pto.vector_section():
                a5.row_sum_micro(
                    src_view.slice([0, 0], [32, 32]),
                    dst_view.slice([0, 0], [1, 32]),
                    dtype=pto.float32,
                    shape=[32, 32],
                )


def test_a5_col_reduce_micro_rejects_unsupported_dtype():
    def meta_data():
        return {"ptr_t": pto.ptr(pto.bool)}

    with pytest.raises(ValueError, match="TCOLREDUCE input data type is not supported"):
        @to_ir_module(meta_data=meta_data)
        def invalid_col_reduce(src: "ptr_t", dst: "ptr_t") -> None:
            src_view = pto.make_tensor(src, shape=[32, 32], dtype=pto.bool)
            dst_view = pto.make_tensor(dst, shape=[1, 32], dtype=pto.bool)
            with pto.vector_section():
                a5.col_sum_micro(
                    src_view.slice([0, 0], [32, 32]),
                    dst_view.slice([0, 0], [1, 32]),
                    dtype=pto.bool,
                    shape=[32, 32],
                )
