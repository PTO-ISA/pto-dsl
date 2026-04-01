"""Public A5 op surface split into small, opcode-focused implementation files."""

from ._common import (
    VF_IMPL_1D_NO_POST_UPDATE,
    VF_IMPL_1D_POST_UPDATE,
    VF_IMPL_2D_NO_POST_UPDATE,
    VF_IMPL_2D_POST_UPDATE,
    VF_IMPL_DEFAULT,
)
from .native import (
    compare,
    concat,
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
    move_tile,
    store_tile,
    trans,
    vector_copy,
    vload,
    vstore,
)
from .tindex import tgather, tgatherb, tscatter
from .tselect import tsel, tsels
from .tbinary import (
    tand,
    tadd,
    tdiv,
    tmax,
    tmin,
    tmov,
    tmul,
    tor_,
    tprelu,
    tshl,
    tshr,
    tsub,
    txor,
)
from .texpand import (
    tcol_expand,
    tcol_expand_add,
    tcol_expand_div,
    tcol_expand_max,
    tcol_expand_min,
    tcol_expand_mul,
    tcol_expand_sub,
    trow_expand,
    trow_expand_add,
    trow_expand_div,
    trow_expand_max,
    trow_expand_min,
    trow_expand_mul,
    trow_expand_sub,
)
from .treduce import (
    tcol_max,
    tcol_min,
    tcol_prod,
    tcol_sum,
    trow_max,
    trow_min,
    trow_prod,
    trow_sum,
)
from .tscalar import (
    taxpy,
    texpands,
    tadds,
    tands,
    tdivs,
    tlrelu,
    tmaxs,
    tmins,
    tmuls,
    tors,
    tshls,
    tshrs,
    tsubs,
    txors,
)
from .tsort import tmrgsort, tsort32
from .tunary import tabs, texp, tlog, trecip, trelu, trsqrt, tsqrt

# A5-style aliases.
TLoad = load_tile
TStore = store_tile
TMov = tmov
TAdd = tadd
TAddS = tadds
TSub = tsub
TSubS = tsubs
TMul = tmul
TMulS = tmuls
TDiv = tdiv
TDivS = tdivs
TMax = tmax
TMaxS = tmaxs
TMaxs = tmaxs
TMin = tmin
TMinS = tmins
TMins = tmins
TAnd = tand
TAndS = tands
TOr = tor_
TOrS = tors
TXor = txor
TXorS = txors
TShl = tshl
TShlS = tshls
TShr = tshr
TShrS = tshrs
TCmp = compare
TExp = texp
TLog = tlog
TRelu = trelu
TLRelu = tlrelu
TAbs = tabs
TSqrt = tsqrt
TRsqrt = trsqrt
TRecip = trecip
TAxpy = taxpy
TExpandS = texpands
TGather = tgather
TGatherB = tgatherb
TScatter = tscatter
gatherb = tgatherb
scatter = tscatter
TSel = tsel
TSelS = tsels
TSels = tsels
select = tsel
selects = tsels
TPrelu = tprelu
TConcat = concat
TExtract = extract
TInsert = insert
TRowSum = trow_sum
TRowMin = trow_min
TRowMax = trow_max
TRowProd = trow_prod
row_prod = trow_prod
TRowExpand = trow_expand
TRowExpandAdd = trow_expand_add
TRowExpandSub = trow_expand_sub
TRowExpandMul = trow_expand_mul
TRowExpandDiv = trow_expand_div
TRowExpandMax = trow_expand_max
TRowExpandMin = trow_expand_min
TColSum = tcol_sum
TColMin = tcol_min
TColMax = tcol_max
TColProd = tcol_prod
col_prod = tcol_prod
TColExpand = tcol_expand
TColExpandAdd = tcol_expand_add
TColExpandSub = tcol_expand_sub
TColExpandDiv = tcol_expand_div
TColExpandMul = tcol_expand_mul
TColExpandMax = tcol_expand_max
TColExpandMin = tcol_expand_min
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
    "TAndS",
    "TAxpy",
    "TExpandS",
    "TColExpand",
    "TColExpandAdd",
    "TColExpandDiv",
    "TColExpandMax",
    "TColExpandMin",
    "TColExpandMul",
    "TColExpandSub",
    "TColMax",
    "TColMin",
    "TColProd",
    "TColSum",
    "TConcat",
    "TCmp",
    "TDiv",
    "TDivS",
    "TExp",
    "TExtract",
    "TGather",
    "TGatherB",
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
    "TMaxs",
    "TMin",
    "TMinS",
    "TMins",
    "TMov",
    "TMrgSort",
    "TMul",
    "TMulS",
    "TOr",
    "TOrS",
    "TPrelu",
    "TRecip",
    "TRelu",
    "TRowExpand",
    "TRowExpandAdd",
    "TRowExpandDiv",
    "TRowExpandMax",
    "TRowExpandMin",
    "TRowExpandMul",
    "TRowExpandSub",
    "TRowMax",
    "TRowMin",
    "TRowProd",
    "TRowSum",
    "TRsqrt",
    "TScatter",
    "TSel",
    "TSelS",
    "TSels",
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
    "TXorS",
    "compare",
    "col_prod",
    "concat",
    "extract",
    "full_mask_b32",
    "gatherb",
    "insert",
    "load_tile",
    "matmul",
    "matmul_acc",
    "matmul_bias",
    "matmul_mx",
    "matmul_mx_acc",
    "matmul_mx_bias",
    "move_tile",
    "row_prod",
    "scatter",
    "select",
    "selects",
    "store_tile",
    "tabs",
    "tadd",
    "tadds",
    "tand",
    "tands",
    "taxpy",
    "texpands",
    "tcol_expand",
    "tcol_expand_add",
    "tcol_expand_div",
    "tcol_expand_max",
    "tcol_expand_min",
    "tcol_expand_mul",
    "tcol_expand_sub",
    "tcol_max",
    "tcol_min",
    "tcol_prod",
    "tcol_sum",
    "tdiv",
    "tdivs",
    "texp",
    "tgather",
    "tgatherb",
    "tlrelu",
    "tlog",
    "tmax",
    "tmaxs",
    "tmin",
    "tmins",
    "tmov",
    "tmrgsort",
    "tmul",
    "tmuls",
    "tor_",
    "tors",
    "tprelu",
    "trans",
    "trecip",
    "trelu",
    "trow_expand",
    "trow_expand_add",
    "trow_expand_div",
    "trow_expand_max",
    "trow_expand_min",
    "trow_expand_mul",
    "trow_expand_sub",
    "trow_max",
    "trow_min",
    "trow_prod",
    "trow_sum",
    "tscatter",
    "tsel",
    "tsels",
    "trsqrt",
    "tshl",
    "tshls",
    "tshr",
    "tshrs",
    "tsort32",
    "tsqrt",
    "tsub",
    "tsubs",
    "txor",
    "txors",
    "vector_copy",
    "vload",
    "vstore",
]
