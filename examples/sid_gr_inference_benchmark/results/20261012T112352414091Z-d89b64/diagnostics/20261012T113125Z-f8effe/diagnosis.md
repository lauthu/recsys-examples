# GPU diagnosis

| Field | Value |
| --- | --- |
| Case | `qwen3_1_7B/ctx1000_b1_w256` |
| Source run | [20261012T112352414091Z-d89b64](../../report.md) |
| Timing class | Profiled; excluded from e2e rankings |

| Context | Batch | Beam | Output tokens | Warmups | Profiled repeats |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1 | 256 | 3 | 5 | 1 |

## Measured GPU work

| Candidate | Status | Measured ranges | Window ms | GPU-active ms | Kernel sum ms |
| --- | --- | ---: | ---: | ---: | ---: |
| sid_gr_inference | success | 1 | 16.030 | 13.638 | 13.638 |
| sglang | success | 1 | 48.064 | 39.822 | 39.884 |
| sglang_beam_search | success | 1 | 35.297 | 28.287 | 28.298 |
| sglang_beam_search_new | success | 1 | 34.885 | 20.114 | 20.180 |

| Metric | Scope |
| --- | --- |
| Measured ranges | Complete `sid_bench/measured` NVTX ranges; warmup excluded |
| Window ms | Union of measured NVTX intervals |
| GPU-active ms | Union of kernel intervals inside measured ranges |
| Kernel sum ms | Sum of kernel durations; overlaps count more than once |
| Share % | Kernel duration / total kernel duration |
| Attribution | Raw kernel names; no inferred semantic stages |

## Kernel breakdown

<details>
<summary>sid_gr_inference: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 56 | 3.373 | 24.731 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.615 | 11.845 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.202 | 8.816 |
| `kernel_cutlass_kernel_srcsm120flash_fwdFlashAttentionForwardSm120_object_at__tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16g_0` | 56 | 1.182 | 8.667 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 2 | 1.063 | 7.794 |
| `void at::native::mbtopk::radixFindKthValues<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.943 | 6.912 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.688 | 5.044 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 2 | 0.397 | 2.910 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, unsigned int, T2, at::cuda::detail::TensorInfo<T1, T2>, T2, at::cuda::detail::TensorInfo<long, T2>, T2, unsigned int, unsigned int, T1 *, unsigned int *, unsigned int *, unsigned int)` | 2 | 0.326 | 2.390 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 6 | 0.317 | 2.327 |

</details>

<details>
<summary>sglang: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned int)1, (unsigned int)8, (unsigned int)16, (unsigned int)2, (unsigned int)4, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, Params>(T9)` | 56 | 12.895 | 32.331 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 112 | 11.305 | 28.344 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 30 | 2.900 | 7.272 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.514 | 3.796 |
| `void flashinfer::BatchPrefillWithRaggedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, (unsigned int)128, (unsigned int)2, (unsigned int)2, (unsigned int)8, (unsigned int)8, (unsigned int)4, (unsigned int)1, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, (bool)1>, RaggedParams>(T2)` | 28 | 1.158 | 2.903 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.152 | 2.889 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 1.066 | 2.672 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)` | 28 | 1.001 | 2.510 |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8, (unsigned int)4, __nv_bfloat16, __nv_bfloat16, int>(T5 *, float *, T7 *, T6 *, float *, unsigned int, unsigned int *, unsigned int)` | 28 | 0.757 | 1.897 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.628 | 1.575 |

</details>

<details>
<summary>sglang_beam_search: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned int)1, (unsigned int)8, (unsigned int)16, (unsigned int)2, (unsigned int)4, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, Params>(T9)` | 56 | 12.419 | 43.888 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 56 | 3.484 | 12.313 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` | 84 | 1.822 | 6.439 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.636 | 5.781 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.185 | 4.189 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 2 | 1.057 | 3.736 |
| `void at::native::mbtopk::radixFindKthValues<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.954 | 3.370 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.699 | 2.469 |
| `void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, at::native::<unnamed>::LogSoftMaxForwardEpilogue>(T4 *, const T2 *, int)` | 3 | 0.662 | 2.338 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align8>(T1::Params)` | 28 | 0.548 | 1.935 |

</details>

<details>
<summary>sglang_beam_search_new: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)0, (unsigned int)16, (unsigned int)1, (unsigned int)1, (unsigned int)8, (unsigned int)8, (unsigned int)1, (unsigned int)4, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>>, PagedParams>(T2)` | 84 | 4.073 | 20.184 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 56 | 3.383 | 16.764 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.616 | 8.008 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.172 | 5.808 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 2 | 1.058 | 5.245 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` | 28 | 0.996 | 4.935 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.952 | 4.716 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.695 | 3.445 |
| `void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, at::native::<unnamed>::LogSoftMaxForwardEpilogue>(T4 *, const T2 *, int)` | 3 | 0.674 | 3.338 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)7, (bool)0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>>(T13)` | 29 | 0.666 | 3.301 |

</details>

## Artifacts

[Run archive](../../raw_artifacts.tar.gz)

| Path inside the run | Contents |
| --- | --- |
| `diagnostics/<capture>/diagnosis.json` | Capture status and all kernel totals |
| `diagnostics/<capture>/<candidate>/timeline/` | Request, commands, logs, `.nsys-rep` and SQLite trace |

Use the source run's `report` command to rebuild this view without a GPU.
Rerun `e2e` after an optimization to measure its unprofiled impact.
