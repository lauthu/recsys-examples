# Benchmark images

The benchmark mounts each engine's source read-only at its pinned commit and puts
it first on `PYTHONPATH`, and the client checks that the server imported it from
there. An image therefore only has to supply the dependencies, and Nsight Systems.

A release may use its official image, pinned by digest and written
`name:tag@digest`: Docker pulls the digest, and the tag names the release for
readers. Code on a dev branch is built from that branch's source at the commit
pinned in `configs/solutions/`; the image is tagged
`<owner>-<repo>:<branch>-<commit>` and labeled with the source URL, commit and
branch. `build.sh ENGINE [CHECKOUT]` builds either kind:

| Engine | Source | Recipe | Tag |
| --- | --- | --- | --- |
| `sglang` | release v0.5.22 | [Dockerfile.sglang](Dockerfile.sglang): the official image, by digest | `sid-gr-bench:sglang` |
| `sglang-beam-search-new` | [cswuyg/sglang@df191d6](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891), `feature/beam_search_update_0801` | the source's `docker/Dockerfile` with one pin (below), target `framework_final` | `cswuyg-sglang:feature-beam_search_update_0801-df191d672f19` |
| `vedantjh2-sglang` | [vedantjh2/sglang@646a7f5](https://github.com/vedantjh2/sglang/tree/646a7f5ce6f94be84237c50cd41bb2e7beca7788), `trie-head` | the source's `docker/Dockerfile` with one pin (below), target `framework_final` | `vedantjh2-sglang:trie-head-646a7f5ce6f9` |
| `sid-gr` | [NVIDIA/recsys-examples@c6eba91](https://github.com/NVIDIA/recsys-examples/tree/c6eba911231d69d8cabaab94626bcb5a25c5a719), `main` | [Dockerfile.sid-gr](Dockerfile.sid-gr): an SGLang release plus the dependencies the commit declares | `nvidia-recsys-examples-sid-gr-inference:main-c6eba911231d` |
| `gr-inference-internal` | [cb521/gr-inference-internal@facff3d](https://github.com/cb521/gr-inference-internal/tree/facff3db9c4d94ba5a5f1676169e318898cf82ba), `perf_precision_fix` | the source's `Dockerfile`: an SGLang release plus the commit's own install (below) | `cb521-gr-inference-internal:perf_precision_fix-facff3db9c4d` |

```bash
images/build.sh sglang
images/build.sh sglang-beam-search-new /path/to/cswuyg-sglang   # clean, at the pin
images/build.sh vedantjh2-sglang /path/to/vedantjh2-sglang        # clean, at the pin
images/build.sh sid-gr ../..                                    # any clone with the commit
images/build.sh gr-inference-internal /path/to/gr-inference-internal  # any clone with the commit
```

The SGLang forks are built with the recipe SGLang releases with: `BRANCH_TYPE=local`
installs the checkout itself together with the dependencies that commit pins, on
the CUDA image the fork's Dockerfile names (pinned by tag, not digest). The
checkout must be clean and at the pinned commit. `build.sh` adds one pin: the step
that resolves the dependencies also requires `python-dateutil<3`. The recipe
freezes what that step resolves into `constraints.txt` for the later steps, and
every botocore requires python-dateutil below 3, so with python-dateutil 3.0 (on
PyPI since 2026-10-21) the later `runai-model-streamer[s3]` install cannot resolve.

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

## gr-inference-internal

From its `perf_precision_fix` branch on, gr-inference-internal's own Dockerfile
starts from an SGLang release, v0.5.16 by digest, installs the commit with its
`[kernels]` pins, which match that release, and runs `pip check` and the
engine's own stack check. `build.sh` builds that file from the pinned tree and
`third_party/gr-decode-attention` submodule, whatever the work tree holds, with
one change: the image keeps the base's entrypoint instead of the server script,
so the benchmark starts its client. The image therefore also holds an installed
copy of the engine; the client checks that the server imported the mounted one.
The engine compiles its beam-selection CUDA extension at first use, into the
cache volume; if that fails, it selects beams in torch, and the run fails. The
checkout the benchmark mounts needs the submodule (`git submodule update --init`).
Earlier revisions' Dockerfile starts from PyTorch 2.6.0 with CUDA 12.4, which has
no sm_120 kernels, and `build.sh` rejects it.

## Versions

Read from the images built on 2026-10-21, and on 2026-10-26 for
gr-inference-internal:

| Engine | torch | FlashInfer | CUTLASS DSL | Quack | sglang-kernel | Nsight Systems |
| --- | --- | --- | --- | --- | --- | --- |
| `sglang` | 2.14.1+cu130 | 0.7.1 | 4.8.0 | 0.6.5 | 0.4.9 | 2026.4.1 |
| `sglang-beam-search-new` | 2.11.0+cu130 | 0.6.15.post1 | 4.6.0 | 0.6.3 | 0.4.5 | 2026.4.1 |
| `vedantjh2-sglang` | 2.13.0+cu130 | 0.6.18 | 4.6.2 | 0.6.4 | 0.4.6.post1 | 2026.4.1 |
| `sid-gr` | 2.11.0+cu130 | 0.6.11.post1 | 4.5.1 | 0.4.1 | 0.4.2.post2 | 2026.2.1 |
| `gr-inference-internal` | 2.11.0+cu130 | 0.6.14 | 4.6.0 | 0.6.1 | 0.4.5 | 2026.4.1 |

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
