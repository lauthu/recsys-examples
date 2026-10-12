# GPU diagnosis

| Field | Value |
| --- | --- |
| Case | `qwen3_1_7B/ctx1000_b1_w256` |
| Source run | [20261012T074633875336Z-dcc97c](../../report.md) |
| Timing class | Profiled; excluded from e2e rankings |

| Context | Batch | Beam | Output tokens | Warmups | Profiled repeats |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1 | 256 | 3 | 5 | 1 |

## Measured GPU work

| Candidate | Status | Measured ranges | Window ms | GPU-active ms | Kernel sum ms |
| --- | --- | ---: | ---: | ---: | ---: |
| sid_gr_inference | success | 1 | 28.153 | 25.199 | 25.199 |
| sglang_beam_search | success | 1 | 43.185 | 34.809 | 34.815 |

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
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 112 | 10.450 | 41.470 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 30 | 2.717 | 10.784 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.472 | 5.840 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.094 | 4.343 |
| `kernel_cutlass_kernel_srcsm120flash_fwdFlashAttentionForwardSm120_object_at__tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16g_0` | 56 | 1.059 | 4.202 |
| `void at::native::mbtopk::radixFindKthValues<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.920 | 3.651 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)` | 28 | 0.894 | 3.548 |
| `void pytorch_flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)128, (int)64, (int)128, (int)4, cutlass::bfloat16_t>>, (bool)1, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0, (bool)1, (bool)0>(pytorch_flash::Flash_fwd_params)` | 28 | 0.859 | 3.408 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(c10::BFloat16) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 280 | 0.672 | 2.666 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(T1::Params)` | 56 | 0.619 | 2.457 |

</details>

<details>
<summary>sglang_beam_search: top 10 kernels</summary>

| Kernel | Calls | GPU ms | Share % |
| --- | ---: | ---: | ---: |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(T1::Params)` | 112 | 10.642 | 30.567 |
| `void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned int)1, (unsigned int)8, (unsigned int)16, (unsigned int)2, (unsigned int)4, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>, Params>(T9)` | 56 | 10.593 | 30.428 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(T1::Params)` | 30 | 2.707 | 7.776 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_64x3_tn_align8>(T1::Params)` | 56 | 1.455 | 4.178 |
| `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, (unsigned int)128, (unsigned int)2, (unsigned int)2, (unsigned int)8, (unsigned int)8, (unsigned int)4, (unsigned int)1, (flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, int, flashinfer::DefaultAttention<(bool)0, (bool)0, (bool)0, (bool)0>>, PagedParams>(T2)` | 28 | 1.095 | 3.144 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(T1::Params)` | 56 | 1.066 | 3.062 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>(T1::Params)` | 28 | 0.890 | 2.556 |
| `void at::native::mbtopk::radixFindKthValues<float, unsigned int, unsigned int, (int)2>(at::cuda::detail::TensorInfo<const T1, T2>, unsigned int, unsigned int *, unsigned int, T2, int, int, unsigned int, T3, T3 *, short *)` | 8 | 0.878 | 2.523 |
| `void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8, (unsigned int)4, __nv_bfloat16, __nv_bfloat16, int>(T5 *, float *, T7 *, T6 *, float *, unsigned int, unsigned int *, unsigned int)` | 28 | 0.693 | 1.992 |
| `void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, at::native::<unnamed>::LogSoftMaxForwardEpilogue>(T4 *, const T2 *, int)` | 3 | 0.652 | 1.872 |

</details>

## Artifacts

[Run archive](../../raw_artifacts.tar.gz)

| Path inside the run | Contents |
| --- | --- |
| `diagnostics/<capture>/diagnosis.json` | Capture status and all kernel totals |
| `diagnostics/<capture>/<candidate>/timeline/` | Request, commands, logs, `.nsys-rep` and SQLite trace |

Use the source run's `report` command to rebuild this view without a GPU.
Rerun `e2e` after an optimization to measure its unprofiled impact.
