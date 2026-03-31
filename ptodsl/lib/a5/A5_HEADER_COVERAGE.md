# A5 Header Coverage

- Total A5 headers tracked: `116`
- Implemented: `49`
- Partial: `2`
- Native only: `11`
- Pending: `35`
- Blocked/meta: `19`

| header | status | helper | note |
| --- | --- | --- | --- |
| `MGather` | `pending` | `-` | Memory gather helper is not yet represented in the PTODSL A5 layer. |
| `MScatter` | `pending` | `-` | Memory scatter helper is not yet represented in the PTODSL A5 layer. |
| `TAdd` | `implemented` | `tadd` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TAddS` | `implemented` | `tadds` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TAlias` | `blocked` | `-` | C++ helper/meta header, not a tile micro-instruction kernel surface. |
| `TAnd` | `implemented` | `tand` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TAndS` | `implemented` | `tands` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TAssign` | `blocked` | `-` | C++ helper/meta header, not a tile micro-instruction kernel surface. |
| `TAxpy` | `partial` | `taxpy` | Same-dtype vector-micro path is implemented via vmula; the C++ mixed f32<-f16 variant is still missing. |
| `TBinOp` | `implemented` | `tbinary._binary_tile_vop` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TBinSOp` | `implemented` | `tscalar._scalar_tile_vop` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TCmp` | `native` | `compare` | Still forwarded to the native PTO tile builder; packed predicate tile lowering is not rewritten yet. |
| `TCmps` | `native` | `compare` | Still forwarded to the native PTO tile builder; scalar compare packing is not rewritten yet. |
| `TColArgMax` | `pending` | `-` | Arg-reduction micro lowering is not implemented yet. |
| `TColArgMin` | `pending` | `-` | Arg-reduction micro lowering is not implemented yet. |
| `TColExpand` | `implemented` | `tcol_expand` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColExpandAdd` | `implemented` | `tcol_expand_add` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColExpandBinOp` | `pending` | `-` | Generic binary broadcast frontend is not exposed yet. |
| `TColExpandDiv` | `implemented` | `tcol_expand_div` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColExpandExpdif` | `pending` | `-` | Specialized exp-diff broadcast lowering is not implemented yet. |
| `TColExpandMax` | `implemented` | `tcol_expand_max` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColExpandMin` | `implemented` | `tcol_expand_min` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColExpandMul` | `implemented` | `tcol_expand_mul` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColExpandSub` | `implemented` | `tcol_expand_sub` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColMax` | `implemented` | `tcol_max` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColMin` | `implemented` | `tcol_min` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColProd` | `blocked` | `-` | No column-product micro lowering is wired yet. |
| `TColReduceIdx` | `pending` | `-` | Indexed column reduction is not implemented yet. |
| `TColReduceOps` | `implemented` | `treduce._tcol_reduce` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TColSum` | `implemented` | `tcol_sum` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TConcat` | `native` | `concat` | Still forwarded to the native PTO tile builder, not rewritten to micro ops yet. |
| `TCvt` | `pending` | `-` | Tile conversion helper is not implemented in the A5 micro layer yet. |
| `TDeQuant` | `pending` | `-` | Quantization/dequantization path is not implemented yet. |
| `TDiv` | `implemented` | `tdiv` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TDivS` | `implemented` | `tdivs` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TExpandS` | `pending` | `-` | Scalar expand helper is not implemented yet. |
| `TExtract` | `native` | `extract` | Still forwarded to the native PTO tile builder. |
| `TFMod` | `pending` | `-` | Fmod lowering is not implemented yet. |
| `TFModS` | `pending` | `-` | Scalar fmod lowering is not implemented yet. |
| `TFillPad` | `pending` | `-` | Pad/fill helper is not implemented yet. |
| `TGather` | `partial` | `tgather` | Indexed gather is implemented via vgather2; mask-pattern gather still needs missing vsqz-style micro support. |
| `TGatherB` | `pending` | `-` | GatherB lowering is not implemented yet, even though vgatherb exists in the micro surface. |
| `TGetScaleAddr` | `pending` | `-` | Scale-address helper is not represented in the PTODSL A5 layer. |
| `THistogram` | `pending` | `-` | Histogram lowering is not implemented yet. |
| `TImg2col` | `blocked` | `-` | Hardware layout/state programming path, not a straightforward vector-micro rewrite target. |
| `TInsert` | `native` | `insert` | Still forwarded to the native PTO tile builder. |
| `TLRelu` | `implemented` | `tlrelu` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TLoad` | `native` | `load_tile` | Structural staging helper, not a compute rewrite target. |
| `TMatmul` | `blocked` | `-` | Cube/L0 path is not a pure vector-micro rewrite target. |
| `TMax` | `implemented` | `tmax` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TMaxs` | `implemented` | `tmaxs` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TMin` | `implemented` | `tmin` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TMins` | `implemented` | `tmins` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TMov` | `implemented` | `tmov` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TMrgSort` | `implemented` | `tmrgsort` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TMul` | `implemented` | `tmul` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TMulS` | `implemented` | `tmuls` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TOr` | `implemented` | `tor_` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TOrS` | `implemented` | `tors` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TPack` | `pending` | `-` | Pack lowering is not implemented yet. |
| `TPartAdd` | `pending` | `-` | Part-op lowering is not implemented yet. |
| `TPartBinOps` | `pending` | `-` | Part-op lowering is not implemented yet. |
| `TPartMax` | `pending` | `-` | Part-op lowering is not implemented yet. |
| `TPartMin` | `pending` | `-` | Part-op lowering is not implemented yet. |
| `TPartMul` | `pending` | `-` | Part-op lowering is not implemented yet. |
| `TPop` | `blocked` | `-` | Runtime buffer stack/state helper, not a direct vector tile rewrite target. |
| `TPrefetch` | `blocked` | `-` | Prefetch/runtime helper, not a direct vector tile rewrite target. |
| `TPrelu` | `pending` | `-` | PReLU lowering is not implemented yet. |
| `TPrint` | `native` | `native print` | Still forwarded to the native PTO tile builder. |
| `TPush` | `blocked` | `-` | Runtime buffer stack/state helper, not a direct vector tile rewrite target. |
| `TQuant` | `pending` | `-` | Quantization path is not implemented yet. |
| `TRandom` | `pending` | `-` | Random-number helper is not implemented yet. |
| `TRem` | `pending` | `-` | Remainder lowering is not implemented yet. |
| `TRemS` | `pending` | `-` | Scalar remainder lowering is not implemented yet. |
| `TReshape` | `native` | `native reshape` | View/layout helper, not rewritten in the A5 micro layer. |
| `TRowExpand` | `implemented` | `trow_expand` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TRowExpandAdd` | `implemented` | `trow_expand_add` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TRowExpandBinOp` | `pending` | `-` | Generic row-broadcast binary frontend is not exposed yet. |
| `TRowExpandDiv` | `implemented` | `trow_expand_div` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TRowExpandExpdif` | `pending` | `-` | Specialized exp-diff row-broadcast lowering is not implemented yet. |
| `TRowExpandMax` | `implemented` | `trow_expand_max` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TRowExpandMin` | `implemented` | `trow_expand_min` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TRowExpandMul` | `implemented` | `trow_expand_mul` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TRowExpandSub` | `implemented` | `trow_expand_sub` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TRowProd` | `blocked` | `-` | No row-product micro lowering is wired yet. |
| `TRowReduce` | `implemented` | `treduce._trow_reduce` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TRowReduceIdx` | `pending` | `-` | Indexed row reduction is not implemented yet. |
| `TRsqrt` | `implemented` | `trsqrt` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TScatter` | `pending` | `-` | Scatter lowering is not implemented yet, even though vscatter exists in the micro surface. |
| `TSel` | `pending` | `-` | Packed-mask select lowering is not implemented yet. |
| `TSels` | `pending` | `-` | Scalar/mask select lowering is not implemented yet. |
| `TSetFmatrix` | `blocked` | `-` | Hardware state setup header, not a straightforward vector-micro rewrite target. |
| `TSetImg2colPadding` | `blocked` | `-` | Hardware state setup header, not a straightforward vector-micro rewrite target. |
| `TSetImg2colRpt` | `blocked` | `-` | Hardware state setup header, not a straightforward vector-micro rewrite target. |
| `TShl` | `implemented` | `tshl` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TShlS` | `implemented` | `tshls` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TShr` | `implemented` | `tshr` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TShrS` | `implemented` | `tshrs` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TSort32` | `implemented` | `tsort32` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TStore` | `native` | `store_tile` | Structural staging helper, not a compute rewrite target. |
| `TSub` | `implemented` | `tsub` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TSubS` | `implemented` | `tsubs` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TSubView` | `blocked` | `-` | View helper, not a tile compute rewrite target. |
| `TSync` | `pending` | `-` | Synchronization helper is not represented in the A5 library layer yet. |
| `TTrans` | `native` | `trans` | Still forwarded to the native PTO tile builder. |
| `TTri` | `pending` | `-` | Triangular helper is not implemented yet. |
| `TUnaryOp` | `implemented` | `tunary._unary_tile_vop` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TXor` | `implemented` | `txor` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `TXorS` | `implemented` | `txors` | Rewritten with PTO micro instructions in the PTODSL A5 layer. |
| `Tci` | `native` | `native tci` | Still forwarded to the native PTO tile builder. |
| `common` | `blocked` | `-` | A5 shared infrastructure header. |
| `custom/Div754` | `blocked` | `-` | Custom implementation helper header. |
| `custom/TSyncCVID` | `blocked` | `-` | Custom sync helper header. |
| `custom/TSync_Custom` | `blocked` | `-` | Custom sync helper header. |
| `datatype` | `blocked` | `-` | A5 shared datatype infrastructure header. |
| `utils` | `blocked` | `-` | A5 shared utility infrastructure header. |
