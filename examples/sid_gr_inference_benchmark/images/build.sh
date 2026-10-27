#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build one engine's image: images/build.sh ENGINE [CHECKOUT]
#   sglang                  a release: the official image, pinned by digest
#   sglang-beam-search-new  dev branches, built from source with the fork's own
#   vedantjh2-sglang          docker/Dockerfile; CHECKOUT is the fork at its pin
#   sid-gr                  Dockerfile.sid-gr plus the dependencies of SID-GR's
#                             pinned commit; CHECKOUT is a recsys-examples clone
#   gr-inference-internal   the source's own Dockerfile, which starts from an
#                             SGLang release; CHECKOUT is a clone with its
#                             submodule
# Images of pinned source are tagged <owner>-<repo>:<branch>-<commit> and
# labeled with the source URL, commit and branch from configs/solutions/.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine="${1:?usage: build.sh ENGINE [CHECKOUT]}"
checkout="${2:-}"

# Read the engine's source pin; set the image name, tag and labels from it.
pin() {
  local solution="$here/../configs/solutions/${engine//-/_}.yaml"
  read -r url pinned branch < <(python3 -c 'import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))["source"]
print(s["url"], s["commit"], s.get("branch", ""))' "$solution")
  [[ -n "$branch" ]] || { echo "$solution needs source.branch" >&2; exit 1; }
  [[ -d "$checkout" ]] || { echo "usage: build.sh $engine CHECKOUT" >&2; exit 1; }
  name="$(basename "$(dirname "$url")")-$(basename "$url" .git)"
  name="$(tr '[:upper:]' '[:lower:]' <<<"$name")"
  tag="${branch//\//-}-${pinned:0:12}"
  labels=(--label "org.opencontainers.image.source=$url"
    --label "org.opencontainers.image.revision=$pinned"
    --label "org.opencontainers.image.ref.name=$branch")
}

case "$engine" in
  sglang)
    exec docker build -f "$here/Dockerfile.sglang" -t sid-gr-bench:sglang "$here"
    ;;
  sglang-beam-search-new | vedantjh2-sglang)
    # The recipe SGLang releases with (target framework_final) installs the
    # checkout itself, so it must be clean at the pinned commit.
    pin
    commit="$(git -C "$checkout" rev-parse HEAD)"
    [[ "$commit" == "$pinned" ]] || {
      echo "$checkout is at $commit, not the pinned $pinned" >&2; exit 1; }
    [[ -z "$(git -C "$checkout" status --porcelain --untracked-files=no)" ]] || {
      echo "$checkout has uncommitted changes" >&2; exit 1; }
    # The recipe freezes the resolved dependencies into constraints.txt. Since
    # python-dateutil 3.0 that includes it, and botocore needs <3, so the later
    # runai-model-streamer[s3] install cannot resolve: hold it below 3.
    recipe="$(mktemp)"
    trap 'rm -f "$recipe"' EXIT
    sed 's|pip install .*"\.\[\${BUILD_TYPE}]"|& "python-dateutil<3"|' \
      "$checkout/docker/Dockerfile" >"$recipe"
    [[ "$(grep -c '"python-dateutil<3"' "$recipe")" == 1 ]] || {
      echo "$checkout/docker/Dockerfile: no single dependency install" >&2; exit 1; }
    echo "building $url@$pinned ($branch) from source in $checkout"
    DOCKER_BUILDKIT=1 docker build -f "$recipe" \
      --target framework_final --build-arg BRANCH_TYPE=local \
      --build-arg BUILD_TYPE=all --build-arg INSTALL_FLASHINFER_JIT_CACHE=1 \
      "${labels[@]}" -t "$name:$tag" "$checkout"
    ;;
  sid-gr)
    # The image gets only the dependencies; the benchmark mounts the source.
    # They are read from the pinned commit, whatever the work tree holds.
    pin
    context="$(mktemp -d)"
    trap 'rm -rf "$context"' EXIT
    git -C "$checkout" show "$pinned:examples/sid-gr-inference/pyproject.toml" \
      >"$context/pyproject.toml"
    echo "building SID-GR's dependencies at $url@$pinned ($branch)"
    DOCKER_BUILDKIT=1 docker build -f "$here/Dockerfile.sid-gr" \
      "${labels[@]}" -t "$name-sid-gr-inference:$tag" "$context"
    ;;
  gr-inference-internal)
    # Its own recipe, from the pinned tree and submodule whatever the work tree
    # holds. Its entrypoint, the server script, gives way to the SGLang base's
    # again: the benchmark starts its client, which starts the server from the
    # mounted source.
    pin
    context="$(mktemp -d)"
    trap 'rm -rf "$context"' EXIT
    git -C "$checkout" archive "$pinned" | tar -x -C "$context"
    grep -q '^ARG BASE_IMAGE=lmsysorg/sglang:' "$context/Dockerfile" || {
      echo "$url@$pinned: its Dockerfile does not start from an SGLang release" >&2
      exit 1; }
    submodule=third_party/gr-decode-attention
    mkdir -p "$context/$submodule"
    git -C "$checkout/$submodule" archive \
      "$(git -C "$checkout" rev-parse "$pinned:$submodule")" |
      tar -x -C "$context/$submodule"
    recipe="$context/Dockerfile.benchmark"
    sed 's|^ENTRYPOINT .*|ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]|' \
      "$context/Dockerfile" >"$recipe"
    [[ "$(grep -c '^ENTRYPOINT' "$recipe")" == 1 ]] || {
      echo "$url@$pinned: no single ENTRYPOINT in its Dockerfile" >&2; exit 1; }
    echo "building gr-inference-internal at $url@$pinned ($branch) from source"
    DOCKER_BUILDKIT=1 docker build -f "$recipe" \
      "${labels[@]}" -t "$name:$tag" "$context"
    ;;
  *)
    echo "unknown engine $engine" >&2
    exit 1
    ;;
esac
