; ModuleID = 'ptoas.hivm.official'
source_filename = "ptoas.hivm.official"

declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare <64 x float> @llvm.hivm.vadd.v64f32.x(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6), i32, i32, i32)

declare { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32)

define void @a5_hivm_vadd_demo(ptr addrspace(6) %0, ptr addrspace(6) %1, ptr addrspace(6) %2, i64 %3) #0 {
  %5 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 64)
  %6 = extractvalue { <256 x i1>, i32 } %5, 0
  %7 = extractvalue { <256 x i1>, i32 } %5, 1
  %8 = trunc i64 %3 to i32
  %9 = mul i32 %8, 4
  %10 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %0, i32 %9, i32 0, i32 0)
  %11 = trunc i64 %3 to i32
  %12 = mul i32 %11, 4
  %13 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %1, i32 %12, i32 0, i32 0)
  %14 = call <64 x float> @llvm.hivm.vadd.v64f32.x(<64 x float> %10, <64 x float> %13, <256 x i1> %6)
  %15 = trunc i64 %3 to i32
  %16 = mul i32 %15, 4
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %14, ptr addrspace(6) %2, i32 %16, i32 2, i32 0, <256 x i1> %6)
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @a5_hivm_vadd_demo, !"kernel", i32 1}
!2 = !{ptr @a5_hivm_vadd_demo, !"kernel_with_simd", i32 1}
