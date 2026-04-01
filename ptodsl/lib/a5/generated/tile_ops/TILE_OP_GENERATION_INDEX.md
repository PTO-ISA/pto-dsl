# Tile Op PTO Generation

| tile op | status | artifact | note |
| --- | --- | --- | --- |
| `mov` | `generated` | `tile_ops/mov.pto` | UB stage + vlds/vsts copy loop. |
| `add` | `generated` | `tile_ops/add.pto` | UB stage + constexpr-specialized TBinOp-style vlds/vadd/vsts lowering. |
| `sub` | `generated` | `tile_ops/sub.pto` | UB stage + constexpr-specialized TBinOp-style vlds/vsub/vsts lowering. |
| `div` | `generated` | `tile_ops/div.pto` | UB stage + constexpr-specialized TBinOp-style vlds/vdiv/vsts lowering. |
| `mul` | `generated` | `tile_ops/mul.pto` | UB stage + constexpr-specialized TBinOp-style vlds/vmul/vsts lowering. |
| `or_` | `generated` | `tile_ops/or_.pto` | UB stage + constexpr-specialized TBinOp-style vlds/vor/vsts lowering. |
| `gather` | `generated` | `tile_ops/gather.pto` | Indexed gather is implemented via vgather2 for same-width source/index pairs; mask-pattern gather still needs unsupported vsqz-style micro support. |
| `exp` | `generated` | `tile_ops/exp.pto` | UB stage + vlds/vexp/vsts loop. |
| `log` | `generated` | `tile_ops/log.pto` | UB stage + vlds/vln/vsts loop. |
| `relu` | `generated` | `tile_ops/relu.pto` | UB stage + vlds/vrelu/vsts loop. |
| `abs` | `generated` | `tile_ops/abs.pto` | UB stage + vlds/vabs/vsts loop. |
| `sqrt` | `generated` | `tile_ops/sqrt.pto` | UB stage + vlds/vsqrt/vsts loop. |
| `rsqrt` | `generated` | `tile_ops/rsqrt.pto` | UB stage + vsqrt/vrec sequence. |
| `reciprocal` | `generated` | `tile_ops/reciprocal.pto` | UB stage + vlds/vrec/vsts loop. |
| `matmul` | `blocked` | - | Cube/L0 path is not a pure vector-micro rewrite target. |
| `matmul_bias` | `blocked` | - | Cube/L0 path is not a pure vector-micro rewrite target. |
| `matmul_acc` | `blocked` | - | Cube/L0 path is not a pure vector-micro rewrite target. |
| `extract` | `blocked` | - | Layout/L0 extraction op, not a vector-micro compute rewrite. |
| `row_sum` | `generated` | `tile_ops/row_sum.pto` | Static-shape row reduction via vcadd + point-store. |
| `row_min` | `generated` | `tile_ops/row_min.pto` | Static-shape row reduction via vcmin + point-store. |
| `row_max` | `generated` | `tile_ops/row_max.pto` | Static-shape row reduction via vcmax + point-store. |
| `row_prod` | `generated` | `tile_ops/row_prod.pto` | Static-shape row reduction via vmul + vintlv tree reduction + point-store. |
| `row_expand` | `generated` | `tile_ops/row_expand.pto` | Static-shape canonical broadcast via vldas/vldus/vdup/vsts. |
| `row_expand_sub` | `generated` | `tile_ops/row_expand_sub.pto` | Static-shape canonical broadcast via vldas/vldus/vdup/vsub/vsts. |
| `row_expand_div` | `generated` | `tile_ops/row_expand_div.pto` | Static-shape canonical broadcast via vldas/vldus/vdup/vdiv/vsts. |
| `row_expand_mul` | `generated` | `tile_ops/row_expand_mul.pto` | Static-shape canonical broadcast via vldas/vldus/vdup/vmul/vsts. |
| `col_sum` | `generated` | `tile_ops/col_sum.pto` | Static-shape TColReduceOps-style column reduction via vadd. |
| `col_min` | `generated` | `tile_ops/col_min.pto` | Static-shape TColReduceOps-style column reduction via vmin. |
| `col_max` | `generated` | `tile_ops/col_max.pto` | Static-shape TColReduceOps-style column reduction via vmax. |
| `col_prod` | `generated` | `tile_ops/col_prod.pto` | Static-shape TColReduceOps-style column reduction via vmul. |
| `col_expand` | `generated` | `tile_ops/col_expand.pto` | Static-shape canonical broadcast via vlds/vsts replication. |
| `mrgsort` | `generated` | `tile_ops/mrgsort.pto` | Single-list row-major merge sort via vmrgsort4. |
| `sort32` | `generated` | `tile_ops/sort32.pto` | Static-shape block sort via vbitsort. |
| `subset` | `not_applicable` | - | View helper only, not a tile compute op. |
