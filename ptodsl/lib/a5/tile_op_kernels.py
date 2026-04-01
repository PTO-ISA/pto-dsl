from ... import pto, to_ir_module
from . import ops
from .tile_micro_coverage import TILE_MICRO_COVERAGE


_MATRIX_SHAPE = [8, 64]
_ROW_VECTOR_SHAPE = [1, 64]
_ROW_REDUCE_OUT_SHAPE = [8, 1]
_COL_REDUCE_OUT_SHAPE = [1, 64]
_ROW_EXPAND_SRC_SHAPE = [8, 1]
_SORT_SHAPE = [1, 64]
_SORT32_OUT_SHAPE = [1, 128]
_BLOCK_LEN = 16


def _resolve_dtype(dtype_name):
    return getattr(pto, dtype_name)


def _tensor(ptr_value, *, shape, dtype):
    return pto.make_tensor(ptr_value, shape=shape, dtype=dtype)


def _tile(view, shape):
    return view.slice([0, 0], shape).raw


def _binary_kernel(kernel_name, helper, *, dtype_name, shape):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {"ptr_t": pto.ptr(dtype)}

    def kernel(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        lhs = _tensor(src0, shape=shape, dtype=dtype)
        rhs = _tensor(src1, shape=shape, dtype=dtype)
        out = _tensor(dst, shape=shape, dtype=dtype)

        with pto.vector_section():
            helper(
                _tile(lhs, shape),
                _tile(rhs, shape),
                _tile(out, shape),
                dtype=dtype,
                tile_shape=shape,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _move_kernel(kernel_name, helper, *, dtype_name, shape):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {"ptr_t": pto.ptr(dtype)}

    def kernel(src: "ptr_t", dst: "ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        src_view = _tensor(src, shape=shape, dtype=dtype)
        out_view = _tensor(dst, shape=shape, dtype=dtype)

        with pto.vector_section():
            helper(
                _tile(src_view, shape),
                _tile(out_view, shape),
                dtype=dtype,
                tile_shape=shape,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _unary_kernel(kernel_name, helper, *, dtype_name, shape):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {"ptr_t": pto.ptr(dtype)}

    def kernel(src: "ptr_t", dst: "ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        inp = _tensor(src, shape=shape, dtype=dtype)
        out = _tensor(dst, shape=shape, dtype=dtype)

        with pto.vector_section():
            helper(
                _tile(inp, shape),
                _tile(out, shape),
                dtype=dtype,
                tile_shape=shape,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _row_reduce_kernel(kernel_name, helper, *, dtype_name):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {
            "src_ptr_t": pto.ptr(dtype),
            "dst_ptr_t": pto.ptr(dtype),
        }

    def kernel(src: "src_ptr_t", dst: "dst_ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        src_view = _tensor(src, shape=_MATRIX_SHAPE, dtype=dtype)
        out_view = _tensor(dst, shape=_ROW_REDUCE_OUT_SHAPE, dtype=dtype)

        with pto.vector_section():
            helper(
                _tile(src_view, _MATRIX_SHAPE),
                _tile(out_view, _ROW_REDUCE_OUT_SHAPE),
                dtype=dtype,
                tile_shape=_MATRIX_SHAPE,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _col_reduce_kernel(kernel_name, helper, *, dtype_name):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {
            "src_ptr_t": pto.ptr(dtype),
            "dst_ptr_t": pto.ptr(dtype),
        }

    def kernel(src: "src_ptr_t", dst: "dst_ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        src_view = _tensor(src, shape=_MATRIX_SHAPE, dtype=dtype)
        out_view = _tensor(dst, shape=_COL_REDUCE_OUT_SHAPE, dtype=dtype)

        with pto.vector_section():
            helper(
                _tile(src_view, _MATRIX_SHAPE),
                _tile(out_view, _COL_REDUCE_OUT_SHAPE),
                dtype=dtype,
                tile_shape=_MATRIX_SHAPE,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _row_expand_kernel(kernel_name, helper, *, dtype_name):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {
            "src_ptr_t": pto.ptr(dtype),
            "dst_ptr_t": pto.ptr(dtype),
        }

    def kernel(src: "src_ptr_t", dst: "dst_ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        src_view = _tensor(src, shape=_ROW_EXPAND_SRC_SHAPE, dtype=dtype)
        out_view = _tensor(dst, shape=_MATRIX_SHAPE, dtype=dtype)

        with pto.vector_section():
            helper(
                _tile(src_view, _ROW_EXPAND_SRC_SHAPE),
                _tile(out_view, _MATRIX_SHAPE),
                dtype=dtype,
                tile_shape=_MATRIX_SHAPE,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _row_expand_binary_kernel(kernel_name, helper, *, dtype_name):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {
            "base_ptr_t": pto.ptr(dtype),
            "expand_ptr_t": pto.ptr(dtype),
            "dst_ptr_t": pto.ptr(dtype),
        }

    def kernel(base: "base_ptr_t", expand: "expand_ptr_t", dst: "dst_ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        base_view = _tensor(base, shape=_MATRIX_SHAPE, dtype=dtype)
        expand_view = _tensor(expand, shape=_ROW_EXPAND_SRC_SHAPE, dtype=dtype)
        out_view = _tensor(dst, shape=_MATRIX_SHAPE, dtype=dtype)

        with pto.vector_section():
            helper(
                _tile(base_view, _MATRIX_SHAPE),
                _tile(expand_view, _ROW_EXPAND_SRC_SHAPE),
                _tile(out_view, _MATRIX_SHAPE),
                dtype=dtype,
                tile_shape=_MATRIX_SHAPE,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _col_expand_kernel(kernel_name, helper, *, dtype_name):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {
            "src_ptr_t": pto.ptr(dtype),
            "dst_ptr_t": pto.ptr(dtype),
        }

    def kernel(src: "src_ptr_t", dst: "dst_ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        src_view = _tensor(src, shape=_ROW_VECTOR_SHAPE, dtype=dtype)
        out_view = _tensor(dst, shape=_MATRIX_SHAPE, dtype=dtype)

        with pto.vector_section():
            helper(
                _tile(src_view, _ROW_VECTOR_SHAPE),
                _tile(out_view, _MATRIX_SHAPE),
                dtype=dtype,
                tile_shape=_MATRIX_SHAPE,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _gather_kernel(kernel_name, *, dtype_name):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {
            "src_ptr_t": pto.ptr(dtype),
            "idx_ptr_t": pto.ptr(_resolve_dtype("uint32")),
            "dst_ptr_t": pto.ptr(dtype),
        }

    def kernel(src: "src_ptr_t", idx: "idx_ptr_t", dst: "dst_ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        index_dtype = _resolve_dtype("uint32")
        src_view = _tensor(src, shape=_SORT_SHAPE, dtype=dtype)
        idx_view = _tensor(idx, shape=_SORT_SHAPE, dtype=index_dtype)
        out_view = _tensor(dst, shape=_SORT_SHAPE, dtype=dtype)

        with pto.vector_section():
            ops.tgather(
                _tile(src_view, _SORT_SHAPE),
                _tile(idx_view, _SORT_SHAPE),
                _tile(out_view, _SORT_SHAPE),
                dtype=dtype,
                index_dtype=index_dtype,
                tile_shape=_SORT_SHAPE,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _mrgsort_kernel(kernel_name, *, dtype_name):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {
            "src_ptr_t": pto.ptr(dtype),
            "dst_ptr_t": pto.ptr(dtype),
        }

    def kernel(src: "src_ptr_t", dst: "dst_ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        src_view = _tensor(src, shape=_SORT_SHAPE, dtype=dtype)
        out_view = _tensor(dst, shape=_SORT_SHAPE, dtype=dtype)

        with pto.vector_section():
            ops.tmrgsort(
                _tile(src_view, _SORT_SHAPE),
                _tile(out_view, _SORT_SHAPE),
                dtype=dtype,
                tile_shape=_SORT_SHAPE,
                block_len=_BLOCK_LEN,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


def _sort32_kernel(kernel_name, *, dtype_name):
    def meta_data():
        dtype = _resolve_dtype(dtype_name)
        return {
            "src_ptr_t": pto.ptr(dtype),
            "idx_ptr_t": pto.ptr(_resolve_dtype("uint32")),
            "dst_ptr_t": pto.ptr(dtype),
        }

    def kernel(src: "src_ptr_t", idx: "idx_ptr_t", dst: "dst_ptr_t") -> None:
        dtype = _resolve_dtype(dtype_name)
        index_dtype = _resolve_dtype("uint32")
        src_view = _tensor(src, shape=_SORT_SHAPE, dtype=dtype)
        idx_view = _tensor(idx, shape=_SORT_SHAPE, dtype=index_dtype)
        out_view = _tensor(dst, shape=_SORT32_OUT_SHAPE, dtype=dtype)

        with pto.vector_section():
            ops.tsort32(
                _tile(src_view, _SORT_SHAPE),
                _tile(idx_view, _SORT_SHAPE),
                _tile(out_view, _SORT32_OUT_SHAPE),
                dtype=dtype,
                tile_shape=_SORT_SHAPE,
            )

    kernel.__name__ = kernel_name
    return to_ir_module(meta_data=meta_data)(kernel)


TILE_OP_KERNEL_SPECS = {
    "mov": {
        "builder": lambda: _move_kernel(
            "tile_op_mov", ops.tmov, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vlds", "pto.vsts"],
    },
    "add": {
        "builder": lambda: _binary_kernel(
            "tile_op_add", ops.tadd, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vadd"],
    },
    "sub": {
        "builder": lambda: _binary_kernel(
            "tile_op_sub", ops.tsub, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vsub"],
    },
    "div": {
        "builder": lambda: _binary_kernel(
            "tile_op_div", ops.tdiv, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vdiv"],
    },
    "mul": {
        "builder": lambda: _binary_kernel(
            "tile_op_mul", ops.tmul, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vmul"],
    },
    "or_": {
        "builder": lambda: _binary_kernel(
            "tile_op_or_", ops.tor_, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vor"],
    },
    "gather": {
        "builder": lambda: _gather_kernel("tile_op_gather", dtype_name="float32"),
        "expected_tokens": ["pto.vgather2"],
    },
    "exp": {
        "builder": lambda: _unary_kernel(
            "tile_op_exp", ops.texp, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vexp"],
    },
    "log": {
        "builder": lambda: _unary_kernel(
            "tile_op_log", ops.tlog, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vln"],
    },
    "relu": {
        "builder": lambda: _unary_kernel(
            "tile_op_relu", ops.trelu, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vrelu"],
    },
    "abs": {
        "builder": lambda: _unary_kernel(
            "tile_op_abs", ops.tabs, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vabs"],
    },
    "sqrt": {
        "builder": lambda: _unary_kernel(
            "tile_op_sqrt", ops.tsqrt, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vsqrt"],
    },
    "rsqrt": {
        "builder": lambda: _unary_kernel(
            "tile_op_rsqrt", ops.trsqrt, dtype_name="float32", shape=_MATRIX_SHAPE
        ),
        "expected_tokens": ["pto.vsqrt", "pto.vrec"],
    },
    "reciprocal": {
        "builder": lambda: _unary_kernel(
            "tile_op_reciprocal",
            ops.trecip,
            dtype_name="float32",
            shape=_MATRIX_SHAPE,
        ),
        "expected_tokens": ["pto.vrec"],
    },
    "row_sum": {
        "builder": lambda: _row_reduce_kernel(
            "tile_op_row_sum", ops.trow_sum, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vcadd"],
    },
    "row_min": {
        "builder": lambda: _row_reduce_kernel(
            "tile_op_row_min", ops.trow_min, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vcmin"],
    },
    "row_max": {
        "builder": lambda: _row_reduce_kernel(
            "tile_op_row_max", ops.trow_max, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vcmax"],
    },
    "row_prod": {
        "builder": lambda: _row_reduce_kernel(
            "tile_op_row_prod", ops.trow_prod, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vmul", "pto.vintlv"],
    },
    "row_expand": {
        "builder": lambda: _row_expand_kernel(
            "tile_op_row_expand", ops.trow_expand, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vdup"],
    },
    "row_expand_sub": {
        "builder": lambda: _row_expand_binary_kernel(
            "tile_op_row_expand_sub", ops.trow_expand_sub, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vsub"],
    },
    "row_expand_div": {
        "builder": lambda: _row_expand_binary_kernel(
            "tile_op_row_expand_div", ops.trow_expand_div, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vdiv"],
    },
    "row_expand_mul": {
        "builder": lambda: _row_expand_binary_kernel(
            "tile_op_row_expand_mul", ops.trow_expand_mul, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vmul"],
    },
    "col_sum": {
        "builder": lambda: _col_reduce_kernel(
            "tile_op_col_sum", ops.tcol_sum, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vadd"],
    },
    "col_min": {
        "builder": lambda: _col_reduce_kernel(
            "tile_op_col_min", ops.tcol_min, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vmin"],
    },
    "col_max": {
        "builder": lambda: _col_reduce_kernel(
            "tile_op_col_max", ops.tcol_max, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vmax"],
    },
    "col_prod": {
        "builder": lambda: _col_reduce_kernel(
            "tile_op_col_prod", ops.tcol_prod, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vmul"],
    },
    "col_expand": {
        "builder": lambda: _col_expand_kernel(
            "tile_op_col_expand", ops.tcol_expand, dtype_name="float32"
        ),
        "expected_tokens": ["pto.vlds", "pto.vsts"],
    },
    "mrgsort": {
        "builder": lambda: _mrgsort_kernel("tile_op_mrgsort", dtype_name="float32"),
        "expected_tokens": ["pto.vmrgsort4"],
    },
    "sort32": {
        "builder": lambda: _sort32_kernel("tile_op_sort32", dtype_name="float32"),
        "expected_tokens": ["pto.vbitsort"],
    },
}

TILE_OP_KERNEL_BUILDERS = {
    name: spec["builder"] for name, spec in TILE_OP_KERNEL_SPECS.items()
}


def tile_op_generation_index_markdown():
    lines = [
        "# Tile Op PTO Generation",
        "",
        "| tile op | status | artifact | note |",
        "| --- | --- | --- | --- |",
    ]
    for op_name, entry in TILE_MICRO_COVERAGE.items():
        if op_name in TILE_OP_KERNEL_BUILDERS:
            artifact = f"`tile_ops/{op_name}.pto`"
            status = "generated"
        else:
            artifact = "-"
            status = entry["status"]
        lines.append(f"| `{op_name}` | `{status}` | {artifact} | {entry['note']} |")
    return "\n".join(lines) + "\n"


__all__ = [
    "TILE_OP_KERNEL_BUILDERS",
    "TILE_OP_KERNEL_SPECS",
    "tile_op_generation_index_markdown",
]
