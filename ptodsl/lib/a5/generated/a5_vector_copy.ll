; ModuleID = 'ptoas.hivm.official'
source_filename = "ptoas.hivm.official"

declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32)

declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6), i32, i32, i32)

define void @a5_vector_copy(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2) #0 {
  %4 = trunc i64 %2 to i32
  %5 = mul i32 %4, 4
  %6 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %0, i32 %5, i32 0, i32 0)
  %7 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 64)
  %8 = extractvalue { <256 x i1>, i32 } %7, 0
  %9 = extractvalue { <256 x i1>, i32 } %7, 1
  %10 = trunc i64 %2 to i32
  %11 = mul i32 %10, 4
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %6, ptr addrspace(6) %1, i32 %11, i32 2, i32 0, <256 x i1> %8)
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @a5_vector_copy, !"kernel", i32 1}
!2 = !{ptr @a5_vector_copy, !"kernel_with_simd", i32 1}
