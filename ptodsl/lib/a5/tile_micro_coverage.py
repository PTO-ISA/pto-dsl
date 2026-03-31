from ptodsl import tile

TILE_MICRO_COVERAGE = {
    "mov": {
        "status": "implemented",
        "helper": "tmov",
        "note": "UB stage + vlds/vsts copy loop.",
    },
    "add": {
        "status": "implemented",
        "helper": "tadd",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vadd/vsts lowering.",
    },
    "sub": {
        "status": "implemented",
        "helper": "tsub",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vsub/vsts lowering.",
    },
    "div": {
        "status": "implemented",
        "helper": "tdiv",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vdiv/vsts lowering.",
    },
    "mul": {
        "status": "implemented",
        "helper": "tmul",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vmul/vsts lowering.",
    },
    "or_": {
        "status": "implemented",
        "helper": "tor_",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vor/vsts lowering.",
    },
    "gather": {
        "status": "partial",
        "helper": "tgather",
        "note": "Indexed gather is implemented via vgather2 for same-width source/index pairs; mask-pattern gather still needs unsupported vsqz-style micro support.",
    },
    "exp": {
        "status": "implemented",
        "helper": "texp",
        "note": "UB stage + vlds/vexp/vsts loop.",
    },
    "log": {
        "status": "implemented",
        "helper": "tlog",
        "note": "UB stage + vlds/vln/vsts loop.",
    },
    "relu": {
        "status": "implemented",
        "helper": "trelu",
        "note": "UB stage + vlds/vrelu/vsts loop.",
    },
    "abs": {
        "status": "implemented",
        "helper": "tabs",
        "note": "UB stage + vlds/vabs/vsts loop.",
    },
    "sqrt": {
        "status": "implemented",
        "helper": "tsqrt",
        "note": "UB stage + vlds/vsqrt/vsts loop.",
    },
    "rsqrt": {
        "status": "implemented",
        "helper": "trsqrt",
        "note": "UB stage + vsqrt/vrec sequence.",
    },
    "reciprocal": {
        "status": "implemented",
        "helper": "trecip",
        "note": "UB stage + vlds/vrec/vsts loop.",
    },
    "matmul": {
        "status": "blocked",
        "helper": None,
        "note": "Cube/L0 path is not a pure vector-micro rewrite target.",
    },
    "matmul_bias": {
        "status": "blocked",
        "helper": None,
        "note": "Cube/L0 path is not a pure vector-micro rewrite target.",
    },
    "matmul_acc": {
        "status": "blocked",
        "helper": None,
        "note": "Cube/L0 path is not a pure vector-micro rewrite target.",
    },
    "extract": {
        "status": "blocked",
        "helper": None,
        "note": "Layout/L0 extraction op, not a vector-micro compute rewrite.",
    },
    "row_sum": {
        "status": "implemented",
        "helper": "trow_sum",
        "note": "Static-shape row reduction via vcadd + point-store.",
    },
    "row_min": {
        "status": "implemented",
        "helper": "trow_min",
        "note": "Static-shape row reduction via vcmin + point-store.",
    },
    "row_max": {
        "status": "implemented",
        "helper": "trow_max",
        "note": "Static-shape row reduction via vcmax + point-store.",
    },
    "row_prod": {
        "status": "blocked",
        "helper": None,
        "note": "No row-product micro lowering is wired yet.",
    },
    "row_expand": {
        "status": "implemented",
        "helper": "trow_expand",
        "note": "Static-shape canonical broadcast via vldas/vldus/vdup/vsts.",
    },
    "row_expand_sub": {
        "status": "implemented",
        "helper": "trow_expand_sub",
        "note": "Static-shape canonical broadcast via vldas/vldus/vdup/vsub/vsts.",
    },
    "row_expand_div": {
        "status": "implemented",
        "helper": "trow_expand_div",
        "note": "Static-shape canonical broadcast via vldas/vldus/vdup/vdiv/vsts.",
    },
    "row_expand_mul": {
        "status": "implemented",
        "helper": "trow_expand_mul",
        "note": "Static-shape canonical broadcast via vldas/vldus/vdup/vmul/vsts.",
    },
    "col_sum": {
        "status": "implemented",
        "helper": "tcol_sum",
        "note": "Static-shape TColReduceOps-style column reduction via vadd.",
    },
    "col_min": {
        "status": "implemented",
        "helper": "tcol_min",
        "note": "Static-shape TColReduceOps-style column reduction via vmin.",
    },
    "col_max": {
        "status": "implemented",
        "helper": "tcol_max",
        "note": "Static-shape TColReduceOps-style column reduction via vmax.",
    },
    "col_prod": {
        "status": "blocked",
        "helper": None,
        "note": "No column-product micro lowering is wired yet.",
    },
    "col_expand": {
        "status": "implemented",
        "helper": "tcol_expand",
        "note": "Static-shape canonical broadcast via vlds/vsts replication.",
    },
    "mrgsort": {
        "status": "implemented",
        "helper": "tmrgsort",
        "note": "Single-list row-major merge sort via vmrgsort4.",
    },
    "sort32": {
        "status": "implemented",
        "helper": "tsort32",
        "note": "Static-shape block sort via vbitsort.",
    },
    "subset": {
        "status": "not_applicable",
        "helper": None,
        "note": "View helper only, not a tile compute op.",
    },
}


def coverage_summary():
    counts = {}
    for entry in TILE_MICRO_COVERAGE.values():
        status = entry["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def coverage_markdown():
    counts = coverage_summary()
    lines = [
        "# Tile Micro Coverage",
        "",
        f"- Total public tile ops: `{len(tile.__all__)}`",
        f"- Implemented: `{counts.get('implemented', 0)}`",
        f"- Partial: `{counts.get('partial', 0)}`",
        f"- Pending: `{counts.get('pending', 0)}`",
        f"- Blocked: `{counts.get('blocked', 0)}`",
        f"- Not applicable: `{counts.get('not_applicable', 0)}`",
        "",
        "| tile op | helper | note |",
        "| --- | --- | --- |",
    ]
    for name in tile.__all__:
        entry = TILE_MICRO_COVERAGE[name]
        helper = entry["helper"] or "-"
        lines.append(f"| `{name}` | `{helper}` | {entry['note']} |")
    return "\n".join(lines) + "\n"


__all__ = ["TILE_MICRO_COVERAGE", "coverage_markdown", "coverage_summary"]
