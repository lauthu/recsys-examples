#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build one engine's image: images/build.sh ENGINE [BUILD_CONTEXT]
# ENGINE is sglang, sglang-beam-search-new, sglang-sidtrie or sid-gr; the context
# is the engine's checkout (none for sglang).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine="${1:?usage: build.sh ENGINE [BUILD_CONTEXT]}"
context="${2:-$here}"

if [[ "$engine" == sglang-sidtrie ]]; then
  # A dev branch, so it is built from source: the fork's own recipe (the
  # docker/Dockerfile SGLang releases with) installs the checkout itself. The
  # checkout must be clean at the commit pinned in the solution file, and the
  # tag names the repository, the pinned branch and the commit.
  solution="$here/../configs/solutions/sglang_sidtrie.yaml"
  read -r url pinned branch < <(python3 -c 'import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))["source"]
print(s["url"], s["commit"], s["branch"])' "$solution")
  commit="$(git -C "$context" rev-parse HEAD)"
  [[ "$commit" == "$pinned" ]] || {
    echo "$context is at $commit, not $pinned from $solution" >&2; exit 1; }
  [[ -z "$(git -C "$context" status --porcelain --untracked-files=no)" ]] || {
    echo "$context has uncommitted changes" >&2; exit 1; }
  repo="$(basename "$(dirname "$url")")-$(basename "$url" .git)"
  echo "building $url@$commit ($branch) from source in $context"
  DOCKER_BUILDKIT=1 exec docker build -f "$context/docker/Dockerfile" \
    --target framework_final --build-arg BRANCH_TYPE=local \
    --build-arg CUDA_VERSION=13.0.3 --build-arg BUILD_TYPE=all \
    --build-arg INSTALL_FLASHINFER_JIT_CACHE=1 \
    --label "org.opencontainers.image.source=$url" \
    --label "org.opencontainers.image.revision=$commit" \
    --label "org.opencontainers.image.ref.name=$branch" \
    -t "$repo:${branch//\//-}-${commit:0:12}" "$context"
fi

[[ -f "$here/Dockerfile.$engine" ]] || { echo "no recipe for $engine" >&2; exit 1; }
if [[ -d "$context/.git" ]]; then
  echo "building from $(git -C "$context" rev-parse HEAD) in $context"
fi
exec docker build -f "$here/Dockerfile.$engine" -t "sid-gr-bench:$engine" "$context"
