from mlir.ir import IndexType

from ptodsl import micro, pto, tile, to_ir_module
from ptodsl.api.scalar import _unwrap


def test_pythonic_pto_tensor_and_tile_buffer_surface_emits_expected_ir():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def wrapper_demo(
        src: "ptr_t", dst: "ptr_t", valid_row: "index_t", valid_col: "index_t"
    ) -> None:
        src_view = pto.make_tensor(src, shape=[8, 64], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[8, 64], dtype=pto.float32)

        src_tile = src_view.slice([0, 0], [8, 64])
        dst_tile = dst_view.slice([0, 0], [8, 64])

        with pto.vector_section():
            buf = pto.make_tile_buffer(
                pto.float32,
                [8, 64],
                space="VEC",
                valid_shape=[-1, -1],
            ).alloc(valid_row=valid_row, valid_col=valid_col)
            buf.load_from(src_tile)
            buf.store_to(dst_tile)

    text = str(wrapper_demo)

    assert "pto.make_tensor_view" in text
    assert "pto.partition_view" in text
    assert "!pto.tile_buf<vec, 8x64xf32, valid=?x?>" in text
    assert "valid_row = %arg2 valid_col = %arg3" in text
    assert "pto.tload" in text
    assert "pto.tstore" in text


def test_tile_ops_accept_pythonic_tile_buffer_wrappers():
    def meta_data():
        return {"ptr_t": pto.ptr(pto.float32)}

    @to_ir_module(meta_data=meta_data)
    def wrapper_add(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t") -> None:
        lhs = pto.make_tensor(src0, shape=[8, 64], dtype=pto.float32)
        rhs = pto.make_tensor(src1, shape=[8, 64], dtype=pto.float32)
        out = pto.make_tensor(dst, shape=[8, 64], dtype=pto.float32)

        with pto.vector_section():
            lhs_buf = pto.make_tile_buffer(pto.float32, [8, 64], space="VEC").alloc()
            rhs_buf = pto.make_tile_buffer(pto.float32, [8, 64], space="VEC").alloc()
            out_buf = pto.make_tile_buffer(pto.float32, [8, 64], space="VEC").alloc()

            lhs_buf.load_from(lhs.slice([0, 0], [8, 64]))
            rhs_buf.load_from(rhs.slice([0, 0], [8, 64]))
            tile.add(lhs_buf, rhs_buf, out_buf)
            out_buf.store_to(out.slice([0, 0], [8, 64]))

    text = str(wrapper_add)

    assert "pto.tload" in text
    assert "pto.tadd" in text
    assert "pto.tstore" in text


def test_public_micro_module_exposes_raw_micro_surface():
    assert hasattr(micro, "vadd")
    assert hasattr(micro, "pset_b32")

    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32, space="VEC"),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def vadd_demo(
        src0: "ptr_t", src1: "ptr_t", dst: "ptr_t", offset: "index_t"
    ) -> None:
        v64f32 = micro.VRegType.get(64, pto.float32)
        mask = micro.pset_b32(micro.MaskType.get(), "PAT_ALL")
        lhs = micro.vlds(v64f32, _unwrap(src0), _unwrap(offset))
        rhs = micro.vlds(v64f32, _unwrap(src1), _unwrap(offset))
        out = micro.vadd(v64f32, lhs, rhs, mask)
        micro.vsts(out, _unwrap(dst), _unwrap(offset), mask)

    text = str(vadd_demo)

    assert "pto.pset_b32" in text
    assert "pto.vlds" in text
    assert "pto.vadd" in text
    assert "pto.vsts" in text
