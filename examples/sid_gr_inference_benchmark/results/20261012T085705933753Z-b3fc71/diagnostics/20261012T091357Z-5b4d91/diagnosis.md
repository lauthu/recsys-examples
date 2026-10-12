# GPU diagnosis

| Field | Value |
| --- | --- |
| Case | `qwen3_1_7B/ctx1000_b1_w256` |
| Source run | [20261012T085705933753Z-b3fc71](../../report.md) |
| Timing class | Profiled; excluded from e2e rankings |

| Context | Batch | Beam | Output tokens | Warmups | Profiled repeats |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1 | 256 | 3 | 5 | 1 |

## Measured GPU work

| Candidate | Status | Measured ranges | Window ms | GPU-active ms | Kernel sum ms |
| --- | --- | ---: | ---: | ---: | ---: |
| sid_gr_inference | success | 1 | 15.583 | 12.689 | 12.689 |
| sglang_beam_search | success | 1 | 37.144 | 25.021 | 25.030 |

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
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 56 | 3.182 | 25.074 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.473 | 11.612 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.107 | 8.724 |
| `kernel_cutlass_kernel_srcsm120flash_fwdFlashAttentionForwardSm120_object_at__tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16g_0` | 56 | 1.059 | 8.348 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 2 | 1.032 | 8.135 |
| `void at::native::mbtopk::radixFindKthValues<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.920 | 7.254 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.621 | 4.898 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>>(int, T2, T3)` | 2 | 0.397 | 3.132 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 6 | 0.309 | 2.433 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, unsigned int, T2, at::cuda::detail::TensorInfo<T1, T2>, T2, at::cuda::detail::TensorInfo<long, T2>, T2, unsigned int, unsigned int, T1 *, unsigned int *, unsigned int *, unsigned int)` | 2 | 0.305 | 2.403 |

</details>

<details>
<summary>sglang_beam_search: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned int)1, (unsigned int)8, (unsigned int)16, (unsigned int)2, (unsigned int)4, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, Params>(T9)` | 56 | 10.574 | 42.244 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 56 | 3.253 | 12.997 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)` | 84 | 1.746 | 6.977 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.453 | 5.807 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.065 | 4.256 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 2 | 1.019 | 4.070 |
| `void at::native::mbtopk::radixFindKthValues<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.885 | 3.534 |
| `void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, at::native::<unnamed>::LogSoftMaxForwardEpilogue>(T4 *, const T2 *, int)` | 3 | 0.652 | 2.605 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.618 | 2.467 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align8>(T1::Params)` | 28 | 0.538 | 2.149 |

</details>

## Artifacts

[Run archive](../../raw_artifacts.tar.gz)

| Path inside the run | Contents |
| --- | --- |
| `diagnostics/<capture>/diagnosis.json` | Capture status and all kernel totals |
| `diagnostics/<capture>/<candidate>/timeline/` | Request, commands, logs, `.nsys-rep` and SQLite trace |

Use the source run's `report` command to rebuild this view without a GPU.
Rerun `e2e` after an optimization to measure its unprofiled impact.
