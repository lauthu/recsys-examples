# Benchmark images

One recipe per dependency set. The engine source is mounted into the container
by the benchmark, not installed, so the build context only supplies the
dependency list. Base images are pinned by digest. Nsight Systems comes with the
base image.

| Engine | Recipe | Build context | Versions in the image used so far |
| --- | --- | --- | --- |
| `sglang` | [Dockerfile.sglang](Dockerfile.sglang) | none | SGLang 0.5.22 image: torch 2.14.1, FlashInfer 0.7.1, CUTLASS DSL 4.8.0 |
| `sglang-beam-search-new` | [Dockerfile.sglang-beam-search-new](Dockerfile.sglang-beam-search-new) | [cswuyg/sglang@df191d6](https://github.com/cswuyg/sglang/tree/df191d672f1955d273bf56fe21684b60dcf6a891) | torch 2.11.0, FlashInfer 0.6.15.post1, CUTLASS DSL 4.6.0 |
| `sglang-sidtrie` | [Dockerfile.sglang-sidtrie](Dockerfile.sglang-sidtrie) | [vedantjh2/sglang@646a7f5](https://github.com/vedantjh2/sglang/tree/646a7f5ce6f94be84237c50cd41bb2e7beca7788) | torch 2.13.0, FlashInfer 0.6.18, CUTLASS DSL 4.6.2, Quack 0.6.4 |
| `sid-gr` | [Dockerfile.sid-gr](Dockerfile.sid-gr) | [cswuyg/sglang@af4e3f4](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1a79ffc82d01a42f7e96f74294b8d) | torch 2.9.1, FlashInfer 0.6.8.post1, CUTLASS DSL 4.5.1, Quack 0.4.1 |

```bash
images/build.sh sglang
images/build.sh sglang-beam-search-new /path/to/sglang_beam_search_new
images/build.sh sglang-sidtrie /path/to/sglang_sidtrie
images/build.sh sid-gr /path/to/sglang_beam_search   # at af4e3f4
```

Images are tagged `sid-gr-bench:ENGINE`; pass that tag as the `IMAGE` of
`--solution`. The runner never pulls or builds images. It records each image's ID,
tags, repository digests and base-image labels in the report. A local build gets a
new ID every time, so the recipe and the pinned base digest are the reproducible
parts, together with the source commit and the package versions the report reads
from the running server.

`Dockerfile.sglang-sidtrie` was reconstructed from the layer history of the image
the published SIDTrie runs used; it has not been rebuilt and compared. The other
recipes are the ones the earlier suite built.
