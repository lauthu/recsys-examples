# GPU diagnosis

| Field | Value |
| --- | --- |
| Case | `qwen3_1_7B/ctx1000_b1_w256` |
| Source run | [20261012T113832092061Z-c0ad33](../../report.md) |
| Timing class | Profiled; excluded from e2e rankings |

| Context | Batch | Beam | Output tokens | Warmups | Profiled repeats |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1 | 256 | 3 | 5 | 1 |

## Measured GPU work

| Candidate | Status | Measured ranges | Window ms | GPU-active ms | Kernel sum ms |
| --- | --- | ---: | ---: | ---: | ---: |
| sglang_beam_search | success | 1 | 35.402 | 28.153 | 28.163 |
| sglang_beam_search_new | success | 1 | 34.784 | 19.911 | 19.978 |
| sid_gr_inference | success | 1 | 16.048 | 13.628 | 13.628 |
| sglang | success | 1 | 48.740 | 39.925 | 39.998 |

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
<summary>sglang_beam_search: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned int)1, (unsigned int)8, (unsigned int)16, (unsigned int)2, (unsigned int)4, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, Params>(T9)` | 56 | 12.628 | 44.839 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 56 | 3.386 | 12.021 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` | 84 | 1.778 | 6.312 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.573 | 5.585 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.172 | 4.160 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 2 | 1.061 | 3.768 |
| `void at::native::mbtopk::radixFindKthValues<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.959 | 3.406 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.667 | 2.370 |
| `void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, at::native::<unnamed>::LogSoftMaxForwardEpilogue>(T4 *, const T2 *, int)` | 3 | 0.665 | 2.362 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align8>(T1::Params)` | 28 | 0.540 | 1.918 |

</details>

<details>
<summary>sglang_beam_search_new: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)0, (unsigned int)16, (unsigned int)1, (unsigned int)1, (unsigned int)8, (unsigned int)8, (unsigned int)1, (unsigned int)4, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>>, PagedParams>(T2)` | 84 | 3.903 | 19.536 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 56 | 3.384 | 16.941 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.617 | 8.094 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.170 | 5.857 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 2 | 1.060 | 5.304 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` | 28 | 0.997 | 4.991 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.943 | 4.723 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.696 | 3.485 |
| `void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, at::native::<unnamed>::LogSoftMaxForwardEpilogue>(T4 *, const T2 *, int)` | 3 | 0.668 | 3.342 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)7, (bool)0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<const __nv_bfloat16>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>>(T13)` | 29 | 0.667 | 3.336 |

</details>

<details>
<summary>sid_gr_inference: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 56 | 3.376 | 24.771 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.615 | 11.852 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.197 | 8.785 |
| `kernel_cutlass_kernel_srcsm120flash_fwdFlashAttentionForwardSm120_object_at__tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16g_0` | 56 | 1.180 | 8.655 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 2 | 1.062 | 7.792 |
| `void at::native::mbtopk::radixFindKthValues<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.937 | 6.874 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.687 | 5.044 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 2 | 0.403 | 2.960 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, unsigned int, T2, at::cuda::detail::TensorInfo<T1, T2>, T2, at::cuda::detail::TensorInfo<long, T2>, T2, unsigned int, unsigned int, T1 *, unsigned int *, unsigned int *, unsigned int)` | 2 | 0.323 | 2.368 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, (bool)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, (bool)1, (bool)0, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)` | 112 | 0.316 | 2.321 |

</details>

<details>
<summary>sglang: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned int)1, (unsigned int)8, (unsigned int)16, (unsigned int)2, (unsigned int)4, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, Params>(T9)` | 56 | 12.463 | 31.160 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 112 | 11.671 | 29.180 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 30 | 2.954 | 7.386 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.634 | 4.086 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.178 | 2.946 |
| `void flashinfer::BatchPrefillWithRaggedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, (unsigned int)128, (unsigned int)2, (unsigned int)2, (unsigned int)8, (unsigned int)8, (unsigned int)4, (unsigned int)1, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, (bool)1>, RaggedParams>(T2)` | 28 | 1.172 | 2.930 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 1.065 | 2.662 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)` | 28 | 1.024 | 2.559 |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8, (unsigned int)4, __nv_bfloat16, __nv_bfloat16, int>(T5 *, float *, T7 *, T6 *, float *, unsigned int, unsigned int *, unsigned int)` | 28 | 0.768 | 1.920 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.692 | 1.730 |

</details>

## Artifacts

[Run archive](../../raw_artifacts.tar.gz)

| Path inside the run | Contents |
| --- | --- |
| `diagnostics/<capture>/diagnosis.json` | Capture status and all kernel totals |
| `diagnostics/<capture>/<candidate>/timeline/` | Request, commands, logs, `.nsys-rep` and SQLite trace |

Use the source run's `report` command to rebuild this view without a GPU.
Rerun `e2e` after an optimization to measure its unprofiled impact.
