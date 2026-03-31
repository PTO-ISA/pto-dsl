"""Helpers that still map directly to PTO tile/cube ops or simple micro utilities."""

from mlir.dialects import arith, pto
from mlir.ir import BoolAttr, IntegerType

from ... import language as dsl
from ...api.scalar import _unwrap
from ._common import (
    _call,
    full_mask,
    load_tile,
    load_view,
    mask_type,
    move_tile,
    s,
    store_tile,
    store_view,
    vreg_type,
)


def adds(src, scalar, out):
    _call(pto.TAddSOp, src, scalar, out)
    return out


def subs(src, scalar, out):
    _call(pto.TSubSOp, src, scalar, out)
    return out


def muls(src, scalar, out):
    _call(pto.TMulSOp, src, scalar, out)
    return out


def divs(src, scalar, out):
    _call(pto.TDivSOp, src, scalar, out)
    return out


def max(lhs, rhs, out):
    _call(pto.TMaxOp, lhs, rhs, out)
    return out


def maxs(src, scalar, out):
    _call(pto.TMaxSOp, src, scalar, out)
    return out


def min(lhs, rhs, out):
    _call(pto.TMinOp, lhs, rhs, out)
    return out


def mins(src, scalar, out):
    _call(pto.TMinSOp, src, scalar, out)
    return out


def and_(lhs, rhs, out):
    _call(pto.TAndOp, lhs, rhs, out)
    return out


def xor(lhs, rhs, out):
    _call(pto.TXorOp, lhs, rhs, out)
    return out


def shl(lhs, rhs, out):
    _call(pto.TShlOp, lhs, rhs, out)
    return out


def shls(src, scalar, out):
    _call(pto.TShlSOp, src, scalar, out)
    return out


def shr(lhs, rhs, out):
    _call(pto.TShrOp, lhs, rhs, out)
    return out


def shrs(src, scalar, out):
    _call(pto.TShrSOp, src, scalar, out)
    return out


def compare(src0, src1, out, *, mode):
    cmp_mode = (
        pto.CmpModeAttr.get(getattr(pto.CmpMode, mode.upper()))
        if isinstance(mode, str)
        else mode
    )
    _call(pto.TCmpOp, src0, src1, out, cmpMode=cmp_mode)
    return out


def scatter(src, indices, dst):
    _call(pto.TScatterOp, src, indices, dst)
    return dst


def select(mask, lhs, rhs, out):
    _call(pto.TSelOp, mask, lhs, rhs, out)
    return out


def concat(lhs, rhs, dst):
    _call(pto.TConcatOp, lhs, rhs, dst)
    return dst


def extract(source, index_row, index_col, out):
    _call(
        pto.TExtractOp,
        src=source,
        indexRow=_unwrap(index_row),
        indexCol=_unwrap(index_col),
        dst=out,
    )
    return out


def insert(source, index_row, index_col, out):
    _call(
        pto.TInsertOp,
        src=source,
        indexRow=_unwrap(index_row),
        indexCol=_unwrap(index_col),
        dst=out,
    )
    return out


def row_prod(src, tmp, dst):
    _call(pto.TRowProdOp, src=src, tmp=tmp, dst=dst)
    return dst


def col_prod(src, tmp, dst, *, is_binary=True):
    _call(pto.TColProdOp, src=src, dst=dst, tmp=tmp, isBinary=BoolAttr.get(is_binary))
    return dst


def col_expand_mul(src0, src1, dst):
    _call(pto.TColExpandMulOp, src0=src0, src1=src1, dst=dst)
    return dst


def col_expand_max(src0, src1, dst):
    _call(pto.TColExpandMaxOp, src0=src0, src1=src1, dst=dst)
    return dst


def col_expand_min(src0, src1, dst):
    _call(pto.TColExpandMinOp, src0=src0, src1=src1, dst=dst)
    return dst


def trans(src, dst):
    _call(pto.TTransOp, src, dst)
    return dst


def matmul(lhs, rhs, out):
    _call(pto.TMatmulOp, None, lhs, rhs, out)
    return out


def matmul_acc(acc, lhs, rhs, out):
    _call(pto.TMatmulAccOp, None, acc, lhs, rhs, out)
    return out


def matmul_bias(lhs, rhs, bias, out):
    _call(pto.TMatmulBiasOp, None, lhs, rhs, bias, out)
    return out


def matmul_mx(lhs, lhs_scale, rhs, rhs_scale, out):
    _call(pto.TMatmulMxOp, None, lhs, lhs_scale, rhs, rhs_scale, out)
    return out


def matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, out):
    _call(pto.TMatmulMxAccOp, None, acc, lhs, lhs_scale, rhs, rhs_scale, out)
    return out


def matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, out):
    _call(pto.TMatmulMxBiasOp, None, lhs, lhs_scale, rhs, rhs_scale, bias, out)
    return out


def full_mask_b32():
    return pto.pset_b32(mask_type(), "PAT_ALL")


def vload(ptr_value, offset, *, lanes=64, dtype=None):
    dtype = dsl.float32 if dtype is None else dtype
    return pto.vlds(vreg_type(lanes, dtype), _unwrap(ptr_value), _unwrap(offset))


def vstore(vector, ptr_value, offset, *, mask=None):
    if mask is None:
        mask = full_mask(dsl.float32)
    pto.vsts(_unwrap(vector), _unwrap(ptr_value), _unwrap(offset), _unwrap(mask))
    return ptr_value


def vector_copy(src_ptr, dst_ptr, offset, *, lanes=64, dtype=None):
    dtype = dsl.float32 if dtype is None else dtype
    vec = vload(src_ptr, offset, lanes=lanes, dtype=dtype)
    pto.vsts(vec, _unwrap(dst_ptr), _unwrap(offset), full_mask(dtype))
    return vec


__all__ = [
    "adds",
    "and_",
    "col_expand_max",
    "col_expand_min",
    "col_expand_mul",
    "col_prod",
    "compare",
    "concat",
    "divs",
    "extract",
    "full_mask_b32",
    "insert",
    "load_tile",
    "load_view",
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
    "muls",
    "row_prod",
    "scatter",
    "select",
    "shl",
    "shls",
    "shr",
    "shrs",
    "store_tile",
    "store_view",
    "subs",
    "trans",
    "vector_copy",
    "vload",
    "vstore",
    "xor",
]
