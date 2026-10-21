# Benchmark images

One recipe per engine. The benchmark mounts the engine source read-only at its
pinned commit and puts it first on `PYTHONPATH`, and the client checks that the
server imported it from there. An image therefore only has to supply the
dependencies, and Nsight Systems. The Dockerfiles here pin their base images by
digest.

`sglang-sidtrie` is a dev branch, so `build.sh` builds it from source. It uses
the fork's own `docker/Dockerfile` (target `framework_final`, the recipe SGLang
releases with), which installs the checkout at the pinned commit together with
the dependencies that commit pins. The checkout must be clean and at
`source.commit` of `configs/solutions/sglang_sidtrie.yaml`. The image is tagged
`<owner>-<repo>:<branch>-<commit>`, for example
`vedantjh2-sglang:trie-head-646a7f5ce6f9`, and labeled with the source URL,
commit and branch. Its base is the CUDA image that the fork's Dockerfile names,
`nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, which is pinned by tag, not digest.

| Engine | Recipe | Build context | Versions in the image used so far |
| --- | --- | --- | --- |
| `sglang` | [Dockerfile.sglang](Dockerfile.sglang) | none | SGLang 0.5.22 image: torch 2.14.1, FlashInfer 0.7.1, CUTLASS DSL 4.8.0 |
| `sglang-beam-search-new` | [Dockerfile.sglang-beam-search-new](Dockerfile.sglang-beam-search-new) | [cswuyg/sglang@df191d6](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) | torch 2.11.0, FlashInfer 0.6.15.post1, CUTLASS DSL 4.6.0 |
| `sglang-sidtrie` | the source's `docker/Dockerfile`, built from source by [build.sh](build.sh) | [vedantjh2/sglang@646a7f5](https://github.com/vedantjh2/sglang/tree/646a7f5ce6f94be84237c50cd41bb2e7beca7788) (`trie-head`) | CUDA 13.0.3, torch 2.13.0, FlashInfer 0.6.18 with its JIT cache, sglang-kernel 0.4.6.post1, CUTLASS DSL 4.6.2, Quack 0.6.4 |
| `sid-gr` | [Dockerfile.sid-gr](Dockerfile.sid-gr) | [cswuyg/sglang@af4e3f4](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) | torch 2.9.1, FlashInfer 0.6.8.post1, CUTLASS DSL 4.5.1, Quack 0.4.1 |

```bash
images/build.sh sglang
images/build.sh sglang-beam-search-new /path/to/sglang_beam_search_new
images/build.sh sglang-sidtrie /path/to/sglang_sidtrie
images/build.sh sid-gr /path/to/sglang_beam_search   # at af4e3f4
```

The other images are tagged `sid-gr-bench:ENGINE`. Pass the tag as the `IMAGE`
of `--solution`. The runner never pulls or builds images. It records each image's ID,
tags, repository digests and base-image labels in the report. A local build gets a
new ID every time, so the recipe and the pinned base digest are the reproducible
parts, together with the source commit and the package versions the report reads
from the running server.

`build.sh sglang-sidtrie` was run on 2026-10-21 and produced image
`sha256:1dc210a64c2e`, the image used for the SIDTrie measurements of
2026-10-20 (PR #3 tools). The other recipes are the ones the earlier suite built.
