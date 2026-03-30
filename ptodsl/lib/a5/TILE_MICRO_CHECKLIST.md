# Tile Micro Coverage

- Total public tile ops: `32`
- Implemented: `26`
- Partial: `1`
- Pending: `0`
- Blocked: `4`
- Not applicable: `1`

| tile op | status | helper | note |
| --- | --- | --- | --- |
| `mov` | `implemented` | `mov_micro` | UB stage + vlds/vsts copy loop. |
| `add` | `implemented` | `add_micro` | UB stage + constexpr-specialized TBinOp-style vlds/vadd/vsts lowering. |
| `sub` | `implemented` | `sub_micro` | UB stage + constexpr-specialized TBinOp-style vlds/vsub/vsts lowering. |
| `div` | `implemented` | `div_micro` | UB stage + constexpr-specialized TBinOp-style vlds/vdiv/vsts lowering. |
| `mul` | `implemented` | `mul_micro` | UB stage + constexpr-specialized TBinOp-style vlds/vmul/vsts lowering. |
| `or_` | `implemented` | `or_micro` | UB stage + constexpr-specialized TBinOp-style vlds/vor/vsts lowering. |
| `gather` | `partial` | `gather_micro` | Indexed gather is implemented via vgather2 for same-width source/index pairs; mask-pattern gather still needs unsupported vsqz-style micro support. |
| `exp` | `implemented` | `exp_micro` | UB stage + vlds/vexp/vsts loop. |
| `log` | `implemented` | `log_micro` | UB stage + vlds/vln/vsts loop. |
| `relu` | `implemented` | `relu_micro` | UB stage + vlds/vrelu/vsts loop. |
| `abs` | `implemented` | `abs_micro` | UB stage + vlds/vabs/vsts loop. |
| `sqrt` | `implemented` | `sqrt_micro` | UB stage + vlds/vsqrt/vsts loop. |
| `rsqrt` | `implemented` | `rsqrt_micro` | UB stage + vsqrt/vrec micro sequence. |
| `reciprocal` | `implemented` | `reciprocal_micro` | UB stage + vlds/vrec/vsts loop. |
| `matmul` | `blocked` | `-` | Cube/L0 path is not a pure vector-micro rewrite target. |
| `matmul_bias` | `blocked` | `-` | Cube/L0 path is not a pure vector-micro rewrite target. |
| `matmul_acc` | `blocked` | `-` | Cube/L0 path is not a pure vector-micro rewrite target. |
| `extract` | `blocked` | `-` | Layout/L0 extraction op, not a vector-micro compute rewrite. |
| `row_sum` | `implemented` | `row_sum_micro` | Static-shape row reduction via vcadd + point-store. |
| `row_min` | `implemented` | `row_min_micro` | Static-shape row reduction via vcmin + point-store. |
| `row_max` | `implemented` | `row_max_micro` | Static-shape row reduction via vcmax + point-store. |
| `row_expand` | `implemented` | `row_expand_micro` | Static-shape canonical broadcast via vldas/vldus/vdup/vsts. |
| `row_expand_sub` | `implemented` | `row_expand_sub_micro` | Static-shape canonical broadcast via vldas/vldus/vdup/vsub/vsts. |
| `row_expand_div` | `implemented` | `row_expand_div_micro` | Static-shape canonical broadcast via vldas/vldus/vdup/vdiv/vsts. |
| `row_expand_mul` | `implemented` | `row_expand_mul_micro` | Static-shape canonical broadcast via vldas/vldus/vdup/vmul/vsts. |
| `col_sum` | `implemented` | `col_sum_micro` | Static-shape TColReduceOps-style column reduction via vadd. |
| `col_min` | `implemented` | `col_min_micro` | Static-shape TColReduceOps-style column reduction via vmin. |
| `col_max` | `implemented` | `col_max_micro` | Static-shape TColReduceOps-style column reduction via vmax. |
| `col_expand` | `implemented` | `col_expand_micro` | Static-shape canonical broadcast via vlds/vsts replication. |
| `mrgsort` | `implemented` | `mrgsort_micro` | Single-list row-major merge sort via vmrgsort4. |
| `sort32` | `implemented` | `sort32_micro` | Static-shape block sort via vbitsort. |
| `subset` | `not_applicable` | `-` | View helper only, not a tile compute op. |
