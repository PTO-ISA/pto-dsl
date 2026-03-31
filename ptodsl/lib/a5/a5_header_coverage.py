"""Coverage view over the broader A5 header surface from pto-isa/include/pto/npu/a5."""

A5_HEADER_INVENTORY = [
    "MGather",
    "MScatter",
    "TAdd",
    "TAddS",
    "TAlias",
    "TAnd",
    "TAndS",
    "TAssign",
    "TAxpy",
    "TBinOp",
    "TBinSOp",
    "TCmp",
    "TCmps",
    "TColArgMax",
    "TColArgMin",
    "TColExpand",
    "TColExpandAdd",
    "TColExpandBinOp",
    "TColExpandDiv",
    "TColExpandExpdif",
    "TColExpandMax",
    "TColExpandMin",
    "TColExpandMul",
    "TColExpandSub",
    "TColMax",
    "TColMin",
    "TColProd",
    "TColReduceIdx",
    "TColReduceOps",
    "TColSum",
    "TConcat",
    "TCvt",
    "TDeQuant",
    "TDiv",
    "TDivS",
    "TExpandS",
    "TExtract",
    "TFMod",
    "TFModS",
    "TFillPad",
    "TGather",
    "TGatherB",
    "TGetScaleAddr",
    "THistogram",
    "TImg2col",
    "TInsert",
    "TLRelu",
    "TLoad",
    "TMatmul",
    "TMax",
    "TMaxs",
    "TMin",
    "TMins",
    "TMov",
    "TMrgSort",
    "TMul",
    "TMulS",
    "TOr",
    "TOrS",
    "TPack",
    "TPartAdd",
    "TPartBinOps",
    "TPartMax",
    "TPartMin",
    "TPartMul",
    "TPop",
    "TPrefetch",
    "TPrelu",
    "TPrint",
    "TPush",
    "TQuant",
    "TRandom",
    "TRem",
    "TRemS",
    "TReshape",
    "TRowExpand",
    "TRowExpandAdd",
    "TRowExpandBinOp",
    "TRowExpandDiv",
    "TRowExpandExpdif",
    "TRowExpandMax",
    "TRowExpandMin",
    "TRowExpandMul",
    "TRowExpandSub",
    "TRowProd",
    "TRowReduce",
    "TRowReduceIdx",
    "TRsqrt",
    "TScatter",
    "TSel",
    "TSels",
    "TSetFmatrix",
    "TSetImg2colPadding",
    "TSetImg2colRpt",
    "TShl",
    "TShlS",
    "TShr",
    "TShrS",
    "TSort32",
    "TStore",
    "TSub",
    "TSubS",
    "TSubView",
    "TSync",
    "TTrans",
    "TTri",
    "TUnaryOp",
    "TXor",
    "TXorS",
    "Tci",
    "common",
    "custom/Div754",
    "custom/TSyncCVID",
    "custom/TSync_Custom",
    "datatype",
    "utils",
]

_MICRO_HELPERS = {
    "TAdd": "tadd",
    "TAddS": "tadds",
    "TAnd": "tand",
    "TAndS": "tands",
    "TBinOp": "tbinary._binary_tile_vop",
    "TBinSOp": "tscalar._scalar_tile_vop",
    "TColExpand": "tcol_expand",
    "TColExpandAdd": "tcol_expand_add",
    "TColExpandDiv": "tcol_expand_div",
    "TColExpandMax": "tcol_expand_max",
    "TColExpandMin": "tcol_expand_min",
    "TColExpandMul": "tcol_expand_mul",
    "TColExpandSub": "tcol_expand_sub",
    "TColMax": "tcol_max",
    "TColMin": "tcol_min",
    "TColReduceOps": "treduce._tcol_reduce",
    "TColSum": "tcol_sum",
    "TDiv": "tdiv",
    "TDivS": "tdivs",
    "TLRelu": "tlrelu",
    "TMax": "tmax",
    "TMaxs": "tmaxs",
    "TMin": "tmin",
    "TMins": "tmins",
    "TMov": "tmov",
    "TMrgSort": "tmrgsort",
    "TMul": "tmul",
    "TMulS": "tmuls",
    "TOr": "tor_",
    "TOrS": "tors",
    "TRowExpand": "trow_expand",
    "TRowExpandAdd": "trow_expand_add",
    "TRowExpandDiv": "trow_expand_div",
    "TRowExpandMax": "trow_expand_max",
    "TRowExpandMin": "trow_expand_min",
    "TRowExpandMul": "trow_expand_mul",
    "TRowExpandSub": "trow_expand_sub",
    "TRowReduce": "treduce._trow_reduce",
    "TRsqrt": "trsqrt",
    "TShl": "tshl",
    "TShlS": "tshls",
    "TShr": "tshr",
    "TShrS": "tshrs",
    "TSort32": "tsort32",
    "TSub": "tsub",
    "TSubS": "tsubs",
    "TUnaryOp": "tunary._unary_tile_vop",
    "TXor": "txor",
    "TXorS": "txors",
}

_PARTIAL_HELPERS = {
    "TAxpy": (
        "taxpy",
        "Same-dtype vector-micro path is implemented via vmula; the C++ mixed f32<-f16 variant is still missing.",
    ),
    "TGather": (
        "tgather",
        "Indexed gather is implemented via vgather2; mask-pattern gather still needs missing vsqz-style micro support.",
    ),
}

_NATIVE_HELPERS = {
    "TConcat": (
        "concat",
        "Still forwarded to the native PTO tile builder, not rewritten to micro ops yet.",
    ),
    "TCmp": (
        "compare",
        "Still forwarded to the native PTO tile builder; packed predicate tile lowering is not rewritten yet.",
    ),
    "TCmps": (
        "compare",
        "Still forwarded to the native PTO tile builder; scalar compare packing is not rewritten yet.",
    ),
    "TExtract": ("extract", "Still forwarded to the native PTO tile builder."),
    "TInsert": ("insert", "Still forwarded to the native PTO tile builder."),
    "TLoad": ("load_tile", "Structural staging helper, not a compute rewrite target."),
    "TPrint": ("native print", "Still forwarded to the native PTO tile builder."),
    "TReshape": (
        "native reshape",
        "View/layout helper, not rewritten in the A5 micro layer.",
    ),
    "TStore": (
        "store_tile",
        "Structural staging helper, not a compute rewrite target.",
    ),
    "TTrans": ("trans", "Still forwarded to the native PTO tile builder."),
    "Tci": ("native tci", "Still forwarded to the native PTO tile builder."),
}

_BLOCKED_HEADERS = {
    "TAlias": "C++ helper/meta header, not a tile micro-instruction kernel surface.",
    "TAssign": "C++ helper/meta header, not a tile micro-instruction kernel surface.",
    "TColProd": "No column-product micro lowering is wired yet.",
    "TImg2col": "Hardware layout/state programming path, not a straightforward vector-micro rewrite target.",
    "TMatmul": "Cube/L0 path is not a pure vector-micro rewrite target.",
    "TPop": "Runtime buffer stack/state helper, not a direct vector tile rewrite target.",
    "TPrefetch": "Prefetch/runtime helper, not a direct vector tile rewrite target.",
    "TPush": "Runtime buffer stack/state helper, not a direct vector tile rewrite target.",
    "TRowProd": "No row-product micro lowering is wired yet.",
    "TSetFmatrix": "Hardware state setup header, not a straightforward vector-micro rewrite target.",
    "TSetImg2colPadding": "Hardware state setup header, not a straightforward vector-micro rewrite target.",
    "TSetImg2colRpt": "Hardware state setup header, not a straightforward vector-micro rewrite target.",
    "TSubView": "View helper, not a tile compute rewrite target.",
    "common": "A5 shared infrastructure header.",
    "custom/Div754": "Custom implementation helper header.",
    "custom/TSyncCVID": "Custom sync helper header.",
    "custom/TSync_Custom": "Custom sync helper header.",
    "datatype": "A5 shared datatype infrastructure header.",
    "utils": "A5 shared utility infrastructure header.",
}

_PENDING_HEADERS = {
    "MGather": "Memory gather helper is not yet represented in the PTODSL A5 layer.",
    "MScatter": "Memory scatter helper is not yet represented in the PTODSL A5 layer.",
    "TColArgMax": "Arg-reduction micro lowering is not implemented yet.",
    "TColArgMin": "Arg-reduction micro lowering is not implemented yet.",
    "TColExpandBinOp": "Generic binary broadcast frontend is not exposed yet.",
    "TColExpandExpdif": "Specialized exp-diff broadcast lowering is not implemented yet.",
    "TColReduceIdx": "Indexed column reduction is not implemented yet.",
    "TCvt": "Tile conversion helper is not implemented in the A5 micro layer yet.",
    "TDeQuant": "Quantization/dequantization path is not implemented yet.",
    "TExpandS": "Scalar expand helper is not implemented yet.",
    "TFMod": "Fmod lowering is not implemented yet.",
    "TFModS": "Scalar fmod lowering is not implemented yet.",
    "TFillPad": "Pad/fill helper is not implemented yet.",
    "TGatherB": "GatherB lowering is not implemented yet, even though vgatherb exists in the micro surface.",
    "TGetScaleAddr": "Scale-address helper is not represented in the PTODSL A5 layer.",
    "THistogram": "Histogram lowering is not implemented yet.",
    "TPack": "Pack lowering is not implemented yet.",
    "TPartAdd": "Part-op lowering is not implemented yet.",
    "TPartBinOps": "Part-op lowering is not implemented yet.",
    "TPartMax": "Part-op lowering is not implemented yet.",
    "TPartMin": "Part-op lowering is not implemented yet.",
    "TPartMul": "Part-op lowering is not implemented yet.",
    "TPrelu": "PReLU lowering is not implemented yet.",
    "TQuant": "Quantization path is not implemented yet.",
    "TRandom": "Random-number helper is not implemented yet.",
    "TRem": "Remainder lowering is not implemented yet.",
    "TRemS": "Scalar remainder lowering is not implemented yet.",
    "TRowExpandBinOp": "Generic row-broadcast binary frontend is not exposed yet.",
    "TRowExpandExpdif": "Specialized exp-diff row-broadcast lowering is not implemented yet.",
    "TRowReduceIdx": "Indexed row reduction is not implemented yet.",
    "TScatter": "Scatter lowering is not implemented yet, even though vscatter exists in the micro surface.",
    "TSel": "Packed-mask select lowering is not implemented yet.",
    "TSels": "Scalar/mask select lowering is not implemented yet.",
    "TSync": "Synchronization helper is not represented in the A5 library layer yet.",
    "TTri": "Triangular helper is not implemented yet.",
}


def _entry(name):
    if name in _MICRO_HELPERS:
        return {
            "status": "implemented",
            "helper": _MICRO_HELPERS[name],
            "note": "Rewritten with PTO micro instructions in the PTODSL A5 layer.",
        }
    if name in _PARTIAL_HELPERS:
        helper, note = _PARTIAL_HELPERS[name]
        return {"status": "partial", "helper": helper, "note": note}
    if name in _NATIVE_HELPERS:
        helper, note = _NATIVE_HELPERS[name]
        return {"status": "native", "helper": helper, "note": note}
    if name in _BLOCKED_HEADERS:
        return {"status": "blocked", "helper": None, "note": _BLOCKED_HEADERS[name]}
    if name in _PENDING_HEADERS:
        return {"status": "pending", "helper": None, "note": _PENDING_HEADERS[name]}
    raise KeyError(f"Missing A5 header coverage classification for '{name}'.")


A5_HEADER_COVERAGE = {name: _entry(name) for name in A5_HEADER_INVENTORY}


def a5_header_coverage_summary():
    counts = {}
    for entry in A5_HEADER_COVERAGE.values():
        status = entry["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def a5_header_coverage_markdown():
    counts = a5_header_coverage_summary()
    lines = [
        "# A5 Header Coverage",
        "",
        f"- Total A5 headers tracked: `{len(A5_HEADER_INVENTORY)}`",
        f"- Implemented: `{counts.get('implemented', 0)}`",
        f"- Partial: `{counts.get('partial', 0)}`",
        f"- Native only: `{counts.get('native', 0)}`",
        f"- Pending: `{counts.get('pending', 0)}`",
        f"- Blocked/meta: `{counts.get('blocked', 0)}`",
        "",
        "| header | status | helper | note |",
        "| --- | --- | --- | --- |",
    ]
    for name in A5_HEADER_INVENTORY:
        entry = A5_HEADER_COVERAGE[name]
        helper = entry["helper"] or "-"
        lines.append(
            f"| `{name}` | `{entry['status']}` | `{helper}` | {entry['note']} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "A5_HEADER_COVERAGE",
    "A5_HEADER_INVENTORY",
    "a5_header_coverage_markdown",
    "a5_header_coverage_summary",
]
