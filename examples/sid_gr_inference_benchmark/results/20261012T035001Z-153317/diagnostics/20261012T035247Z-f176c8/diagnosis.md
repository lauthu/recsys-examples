# GPU diagnosis

| Field | Value |
| --- | --- |
| Case | `qwen3_0_6B/ctx1000_b1_w64` |
| Source run | [20261012T035001Z-153317](../../report.md) |
| Timing class | Profiled; excluded from e2e rankings |

| Context | Batch | Beam | Output tokens | Warmups | Profiled repeats |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1 | 64 | 3 | 2 | 1 |

## Measured GPU work

| Candidate | Status | Measured ranges | Window ms | GPU-active ms | Kernel sum ms |
| --- | --- | ---: | ---: | ---: | ---: |
| sid_gr_inference | success | 1 | 15.868 | 13.425 | 13.529 |

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
| `nvjet_sm120_tst_mma_224x128x64_2_112x32x64_tmaAB_alignCD4_bz_TNNN` | 30 | 1.294 | 9.568 |
| `nvjet_sm120_tst_mma_128x144x64_2_32x72x64_tmaAB_alignCD4_bz_TNNN` | 28 | 1.162 | 8.589 |
| `void at::native::<unnamed>::vectorized_layer_norm_kernel<c10::BFloat16, float, (bool)1>(int, T2, const T1 *, const T1 *, const T1 *, T2 *, T2 *, T1 *)` | 339 | 1.149 | 8.491 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 672 | 1.138 | 8.409 |
| `nvjet_sm120_tst_mma_128x144x64_2_32x72x64_tmaAB_alignCD4_splitK_TNNN` | 56 | 1.075 | 7.948 |
| `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)128, (int)64, (int)128, (int)4, cutlass::bfloat16_t>>, (bool)1, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0, (bool)1, (bool)0>(flash::Flash_fwd_params)` | 28 | 0.833 | 6.161 |
| `nvjet_sm120_tst_mma_128x80x64_3_32x40x64_tmaAB_alignCD4_splitK_TNNN` | 56 | 0.792 | 5.856 |
| `kernel_cutlass_kernel_srcsm120flash_fwdFlashAttentionForwardSm120_object_at__tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16gmemalign16oi64div81i64div8i64div8_tensorptrbf16g_0` | 56 | 0.741 | 5.477 |
| `nvjet_sm120_tst_mma_64x64x64_6_16x64x64_tmaAB_alignCD4_splitK_TNNN` | 112 | 0.641 | 4.741 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, __nv_bfloat16, (bool)1, (bool)0, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)` | 224 | 0.623 | 4.606 |

</details>

## Artifacts

[Run archive](../../raw_artifacts.tar.gz)

| Path inside the run | Contents |
| --- | --- |
| `diagnostics/<capture>/diagnosis.json` | Capture status and all kernel totals |
| `diagnostics/<capture>/<candidate>/timeline/` | Request, commands, logs, `.nsys-rep` and SQLite trace |

Use the source run's `report` command to rebuild this view without a GPU.
Rerun `e2e` after an optimization to measure its unprofiled impact.
