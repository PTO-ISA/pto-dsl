# Tile Micro Coverage

- Total public tile ops: `34`
- Implemented: `28`
- Partial: `1`
- Pending: `0`
- Blocked: `4`
- Not applicable: `1`

| tile op | helper | note |
| --- | --- | --- |
| `mov` | `tmov` | UB stage + vlds/vsts copy loop. |
| `add` | `tadd` | UB stage + constexpr-specialized TBinOp-style vlds/vadd/vsts lowering. |
| `sub` | `tsub` | UB stage + constexpr-specialized TBinOp-style vlds/vsub/vsts lowering. |
| `div` | `tdiv` | UB stage + constexpr-specialized TBinOp-style vlds/vdiv/vsts lowering. |
| `mul` | `tmul` | UB stage + constexpr-specialized TBinOp-style vlds/vmul/vsts lowering. |
| `or_` | `tor_` | UB stage + constexpr-specialized TBinOp-style vlds/vor/vsts lowering. |
| `gather` | `tgather` | Indexed gather is implemented via vgather2 for same-width source/index pairs; mask-pattern gather still needs unsupported vsqz-style micro support. |
| `exp` | `texp` | UB stage + vlds/vexp/vsts loop. |
| `log` | `tlog` | UB stage + vlds/vln/vsts loop. |
| `relu` | `trelu` | UB stage + vlds/vrelu/vsts loop. |
| `abs` | `tabs` | UB stage + vlds/vabs/vsts loop. |
| `sqrt` | `tsqrt` | UB stage + vlds/vsqrt/vsts loop. |
| `rsqrt` | `trsqrt` | UB stage + vsqrt/vrec sequence. |
| `reciprocal` | `trecip` | UB stage + vlds/vrec/vsts loop. |
| `matmul` | `-` | Cube/L0 path is not a pure vector-micro rewrite target. |
| `matmul_bias` | `-` | Cube/L0 path is not a pure vector-micro rewrite target. |
| `matmul_acc` | `-` | Cube/L0 path is not a pure vector-micro rewrite target. |
| `extract` | `-` | Layout/L0 extraction op, not a vector-micro compute rewrite. |
| `row_sum` | `trow_sum` | Static-shape row reduction via vcadd + point-store. |
| `row_min` | `trow_min` | Static-shape row reduction via vcmin + point-store. |
| `row_max` | `trow_max` | Static-shape row reduction via vcmax + point-store. |
| `row_prod` | `trow_prod` | Static-shape row reduction via vmul + vintlv tree reduction + point-store. |
| `row_expand` | `trow_expand` | Static-shape canonical broadcast via vldas/vldus/vdup/vsts. |
| `row_expand_sub` | `trow_expand_sub` | Static-shape canonical broadcast via vldas/vldus/vdup/vsub/vsts. |
| `row_expand_div` | `trow_expand_div` | Static-shape canonical broadcast via vldas/vldus/vdup/vdiv/vsts. |
| `row_expand_mul` | `trow_expand_mul` | Static-shape canonical broadcast via vldas/vldus/vdup/vmul/vsts. |
| `col_sum` | `tcol_sum` | Static-shape TColReduceOps-style column reduction via vadd. |
| `col_min` | `tcol_min` | Static-shape TColReduceOps-style column reduction via vmin. |
| `col_max` | `tcol_max` | Static-shape TColReduceOps-style column reduction via vmax. |
| `col_prod` | `tcol_prod` | Static-shape TColReduceOps-style column reduction via vmul. |
| `col_expand` | `tcol_expand` | Static-shape canonical broadcast via vlds/vsts replication. |
| `mrgsort` | `tmrgsort` | Single-list row-major merge sort via vmrgsort4. |
| `sort32` | `tsort32` | Static-shape block sort via vbitsort. |
| `subset` | `-` | View helper only, not a tile compute op. |
