# Benchmark images

The benchmark mounts each engine's source read-only at its pinned commit and puts
it first on `PYTHONPATH`, and the client checks that the server imported it from
there. An image therefore only has to supply the dependencies, and Nsight Systems.

A release may use its official image, pinned by digest. Code on a dev branch is
built from that branch's source at the commit pinned in `configs/solutions/`; the
image is tagged `<owner>-<repo>:<branch>-<commit>` and labeled with the source
URL, commit and branch. `build.sh ENGINE [CHECKOUT]` builds either kind:

| Engine | Source | Recipe | Tag |
| --- | --- | --- | --- |
| `sglang` | release v0.5.22 | [Dockerfile.sglang](Dockerfile.sglang): the official image, by digest | `sid-gr-bench:sglang` |
| `sglang-beam-search-new` | [cswuyg/sglang@df191d6](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891), `feature/beam_search_update_0801` | the source's `docker/Dockerfile`, target `framework_final` | `cswuyg-sglang:feature-beam_search_update_0801-df191d672f19` |
| `sglang-sidtrie` | [vedantjh2/sglang@646a7f5](https://github.com/vedantjh2/sglang/tree/646a7f5ce6f94be84237c50cd41bb2e7beca7788), `trie-head` | the source's `docker/Dockerfile`, target `framework_final` | `vedantjh2-sglang:trie-head-646a7f5ce6f9` |
| `sid-gr` | [NVIDIA/recsys-examples@c6eba91](https://github.com/NVIDIA/recsys-examples/tree/c6eba911231d69d8cabaab94626bcb5a25c5a719), `main` | [Dockerfile.sid-gr](Dockerfile.sid-gr): an SGLang release plus the dependencies the commit declares | `nvidia-recsys-examples-sid-gr-inference:main-c6eba911231d` |

```bash
images/build.sh sglang
images/build.sh sglang-beam-search-new /path/to/cswuyg-sglang   # clean, at the pin
images/build.sh sglang-sidtrie /path/to/vedantjh2-sglang        # clean, at the pin
images/build.sh sid-gr ../..                                    # any clone with the commit
```

The SGLang forks are built with the recipe SGLang releases with: `BRANCH_TYPE=local`
installs the checkout itself together with the dependencies that commit pins, on
the CUDA image the fork's Dockerfile names (pinned by tag, not digest). The
checkout must be clean and at the pinned commit.

## SID-GR

SID-GR's quickstart (`scripts/run_container.sh`, then
`scripts/bootstrap_container_env.sh --project-kernels`) installs
`sid-gr-inference[kernels]` into the moving `lmsysorg/sglang:dev-cu13` image.
Dockerfile.sid-gr pins that setup to the SGLang release whose kernel stack those
pins match, v0.5.12.post1, current when SID-GR set them: FlashInfer 0.6.11.post1,
CUTLASS DSL 4.5.1 and Quack 0.4.1. On it the pins change only `packaging` (26.2
to 25.0). Later releases ship CUTLASS DSL 4.5.2 or newer (and, from v0.5.16,
Quack 0.6), which the pins would downgrade under SGLang's own pins. `build.sh`
reads `pyproject.toml` from the pinned commit, not from the work tree.

In this image SID-GR selects no SGLang kernel (`tools/check_kernel_backends.py`):
FlashInfer for the norms, RoPE and top-k, its own JIT kernels for the MLP, and
`corelib/gr_decode_atten` for decode attention. Prefill attention uses torch
SDPA, as before: the image's `flash_attn` is FA4's namespace package, which has
no `flash_attn_func`, so `auto` falls back.

`examples/sid-gr-inference/Dockerfile` is not used. It copies a `third_party/`
submodule that exists only in SID-GR's standalone repository, so it does not build
here. Its default base, PyTorch 2.6.0 with CUDA 12.4, has no sm_120 kernels, no
nvcc for SID-GR's JIT kernels and no Nsight Systems.

## Versions

Read from the images built on 2026-10-21:

| Engine | torch | FlashInfer | CUTLASS DSL | Quack | sglang-kernel | Nsight Systems |
| --- | --- | --- | --- | --- | --- | --- |
| `sglang` | 2.14.1+cu130 | 0.7.1 | 4.8.0 | 0.6.5 | 0.4.9 | 2026.4.1 |
| `sglang-beam-search-new` | 2.11.0+cu130 | 0.6.15.post1 | 4.6.0 | 0.6.3 | 0.4.5 | 2026.4.1 |
| `sglang-sidtrie` | 2.13.0+cu130 | 0.6.18 | 4.6.2 | 0.6.4 | 0.4.6.post1 | 2026.4.1 |
| `sid-gr` | 2.11.0+cu130 | 0.6.11.post1 | 4.5.1 | 0.4.1 | 0.4.2.post2 | 2026.2.1 |

Measurements before 2026-10-21 ran SID-GR and beam-search-new in images layered on
an SGLang nightly (`lmsysorg/sglang@sha256:b5c48869…`, dev build `e6072935` of
2026-10-12): SID-GR with torch 2.9.1, FlashInfer 0.6.8.post1, CUTLASS DSL 4.5.1
and Quack 0.4.1; beam-search-new with torch 2.11.0, FlashInfer 0.6.15.post1,
CUTLASS DSL 4.6.0 and Quack 0.6.1.

The runner never pulls or builds images. It records each image's ID, tags,
repository digests and labels in the report. A local build gets a new ID every
time, so the recipe, the pinned base and the source commit are the reproducible
parts, together with the package versions the report reads from the running
server.
