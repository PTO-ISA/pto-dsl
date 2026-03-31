"""Public A5 op surface split into small, opcode-focused implementation files."""

from ._common import (
    VF_IMPL_1D_NO_POST_UPDATE,
    VF_IMPL_1D_POST_UPDATE,
    VF_IMPL_2D_NO_POST_UPDATE,
    VF_IMPL_2D_POST_UPDATE,
    VF_IMPL_DEFAULT,
)
from .native import (
    adds,
    and_,
    col_expand_max,
    col_expand_min,
    col_expand_mul,
    col_prod,
    compare,
    concat,
    divs,
    extract,
    full_mask_b32,
    insert,
    load_tile,
    matmul,
    matmul_acc,
    matmul_bias,
    matmul_mx,
    matmul_mx_acc,
    matmul_mx_bias,
    max,
    maxs,
    min,
    mins,
    move_tile,
    muls,
    row_prod,
    scatter,
    select,
    shl,
    shls,
    shr,
    shrs,
    store_tile,
    subs,
    trans,
    vector_copy,
    vload,
    vstore,
    xor,
)
from .tbinary import tadd, tdiv, tmov, tmul, tor_, tsub
from .texpand import tcol_expand, trow_expand, trow_expand_div, trow_expand_mul, trow_expand_sub
from .treduce import (
    tcol_max,
    tcol_min,
    tcol_sum,
    trow_max,
    trow_min,
    trow_sum,
)
from .tsort import tgather, tmrgsort, tsort32
from .tunary import tabs, texp, tlog, trecip, trelu, trsqrt, tsqrt

# Readable aliases that match the public tile op names.
mov = tmov
add = tadd
sub = tsub
mul = tmul
div = tdiv
or_ = tor_
gather = tgather
exp = texp
log = tlog
relu = trelu
abs = tabs
sqrt = tsqrt
rsqrt = trsqrt
reciprocal = trecip
row_sum = trow_sum
row_min = trow_min
row_max = trow_max
row_expand = trow_expand
row_expand_sub = trow_expand_sub
row_expand_mul = trow_expand_mul
row_expand_div = trow_expand_div
col_sum = tcol_sum
col_min = tcol_min
col_max = tcol_max
col_expand = tcol_expand
mrgsort = tmrgsort
sort32 = tsort32

# A5-style aliases.
TLoad = load_tile
TStore = store_tile
TMov = tmov
TAdd = tadd
TAddS = adds
TSub = tsub
TSubS = subs
TMul = tmul
TMulS = muls
TDiv = tdiv
TDivS = divs
TMax = max
TMaxS = maxs
TMin = min
TMinS = mins
TAnd = and_
TOr = tor_
TXor = xor
TShl = shl
TShlS = shls
TShr = shr
TShrS = shrs
TCmp = compare
TExp = texp
TLog = tlog
TRelu = trelu
TAbs = tabs
TSqrt = tsqrt
TRsqrt = trsqrt
TRecip = trecip
TGather = tgather
TScatter = scatter
TSel = select
TConcat = concat
TExtract = extract
TInsert = insert
TRowSum = trow_sum
TRowMin = trow_min
TRowMax = trow_max
TRowExpand = trow_expand
TRowExpandSub = trow_expand_sub
TRowExpandMul = trow_expand_mul
TRowExpandDiv = trow_expand_div
TColSum = tcol_sum
TColMin = tcol_min
TColMax = tcol_max
TColExpand = tcol_expand
TColExpandMul = col_expand_mul
TColExpandMax = col_expand_max
TColExpandMin = col_expand_min
TMrgSort = tmrgsort
TSort32 = tsort32
TTrans = trans
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
    "adds",
    "and_",
    "col_expand",
    "col_expand_max",
    "col_expand_min",
    "col_expand_mul",
    "col_max",
    "col_min",
    "col_prod",
    "col_sum",
    "compare",
    "concat",
    "div",
    "divs",
    "exp",
    "extract",
    "full_mask_b32",
    "gather",
    "insert",
    "load_tile",
    "log",
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
    "mov",
    "mrgsort",
    "mul",
    "muls",
    "or_",
    "reciprocal",
    "relu",
    "row_expand",
    "row_expand_div",
    "row_expand_mul",
    "row_expand_sub",
    "row_max",
    "row_min",
    "row_prod",
    "row_sum",
    "rsqrt",
    "scatter",
    "select",
    "shl",
    "shls",
    "shr",
    "shrs",
    "sort32",
    "sqrt",
    "store_tile",
    "sub",
    "subs",
    "tabs",
    "tadd",
    "tcol_expand",
    "tcol_max",
    "tcol_min",
    "tcol_sum",
    "tdiv",
    "texp",
    "tgather",
    "tlog",
    "tmov",
    "tmrgsort",
    "tmul",
    "tor_",
    "trans",
    "trecip",
    "trelu",
    "trow_expand",
    "trow_expand_div",
    "trow_expand_mul",
    "trow_expand_sub",
    "trow_max",
    "trow_min",
    "trow_sum",
    "trsqrt",
    "tsort32",
    "tsqrt",
    "tsub",
    "vector_copy",
    "vload",
    "vstore",
    "xor",
]
