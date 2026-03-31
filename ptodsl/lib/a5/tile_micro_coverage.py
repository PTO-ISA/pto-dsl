from ptodsl import tile

TILE_MICRO_COVERAGE = {
    "mov": {
        "status": "implemented",
        "helper": "mov_micro",
        "note": "UB stage + vlds/vsts copy loop.",
    },
    "add": {
        "status": "implemented",
        "helper": "add_micro",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vadd/vsts lowering.",
    },
    "sub": {
        "status": "implemented",
        "helper": "sub_micro",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vsub/vsts lowering.",
    },
    "div": {
        "status": "implemented",
        "helper": "div_micro",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vdiv/vsts lowering.",
    },
    "mul": {
        "status": "implemented",
        "helper": "mul_micro",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vmul/vsts lowering.",
    },
    "or_": {
        "status": "implemented",
        "helper": "or_micro",
        "note": "UB stage + constexpr-specialized TBinOp-style vlds/vor/vsts lowering.",
    },
    "gather": {
        "status": "partial",
        "helper": "gather_micro",
        "note": "Indexed gather is implemented via vgather2 for same-width source/index pairs; mask-pattern gather still needs unsupported vsqz-style micro support.",
    },
    "exp": {
        "status": "implemented",
        "helper": "exp_micro",
        "note": "UB stage + vlds/vexp/vsts loop.",
    },
    "log": {
        "status": "implemented",
        "helper": "log_micro",
        "note": "UB stage + vlds/vln/vsts loop.",
    },
    "relu": {
        "status": "implemented",
        "helper": "relu_micro",
        "note": "UB stage + vlds/vrelu/vsts loop.",
    },
    "abs": {
        "status": "implemented",
        "helper": "abs_micro",
        "note": "UB stage + vlds/vabs/vsts loop.",
    },
    "sqrt": {
        "status": "implemented",
        "helper": "sqrt_micro",
        "note": "UB stage + vlds/vsqrt/vsts loop.",
    },
    "rsqrt": {
        "status": "implemented",
        "helper": "rsqrt_micro",
        "note": "UB stage + vsqrt/vrec micro sequence.",
    },
    "reciprocal": {
        "status": "implemented",
        "helper": "reciprocal_micro",
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
        "helper": "row_sum_micro",
        "note": "Static-shape row reduction via vcadd + point-store.",
    },
    "row_min": {
        "status": "implemented",
        "helper": "row_min_micro",
        "note": "Static-shape row reduction via vcmin + point-store.",
    },
    "row_max": {
        "status": "implemented",
        "helper": "row_max_micro",
        "note": "Static-shape row reduction via vcmax + point-store.",
    },
    "row_expand": {
        "status": "implemented",
        "helper": "row_expand_micro",
        "note": "Static-shape canonical broadcast via vldas/vldus/vdup/vsts.",
    },
    "row_expand_sub": {
        "status": "implemented",
        "helper": "row_expand_sub_micro",
        "note": "Static-shape canonical broadcast via vldas/vldus/vdup/vsub/vsts.",
    },
    "row_expand_div": {
        "status": "implemented",
        "helper": "row_expand_div_micro",
        "note": "Static-shape canonical broadcast via vldas/vldus/vdup/vdiv/vsts.",
    },
    "row_expand_mul": {
        "status": "implemented",
        "helper": "row_expand_mul_micro",
        "note": "Static-shape canonical broadcast via vldas/vldus/vdup/vmul/vsts.",
    },
    "col_sum": {
        "status": "implemented",
        "helper": "col_sum_micro",
        "note": "Static-shape TColReduceOps-style column reduction via vadd.",
    },
    "col_min": {
        "status": "implemented",
        "helper": "col_min_micro",
        "note": "Static-shape TColReduceOps-style column reduction via vmin.",
    },
    "col_max": {
        "status": "implemented",
        "helper": "col_max_micro",
        "note": "Static-shape TColReduceOps-style column reduction via vmax.",
    },
    "col_expand": {
        "status": "implemented",
        "helper": "col_expand_micro",
        "note": "Static-shape canonical broadcast via vlds/vsts replication.",
    },
    "mrgsort": {
        "status": "implemented",
        "helper": "mrgsort_micro",
        "note": "Single-list row-major merge sort via vmrgsort4.",
    },
    "sort32": {
        "status": "implemented",
        "helper": "sort32_micro",
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
        "| tile op | status | helper | note |",
        "| --- | --- | --- | --- |",
    ]
    for name in tile.__all__:
        entry = TILE_MICRO_COVERAGE[name]
        helper = entry["helper"] or "-"
        lines.append(
            f"| `{name}` | `{entry['status']}` | `{helper}` | {entry['note']} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["TILE_MICRO_COVERAGE", "coverage_markdown", "coverage_summary"]
