#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build one engine's image: images/build.sh ENGINE [BUILD_CONTEXT]
# ENGINE is sglang, sglang-beam-search-new, sglang-sidtrie or sid-gr; the context
# is the checkout named at the top of its Dockerfile (none for sglang).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine="${1:?usage: build.sh ENGINE [BUILD_CONTEXT]}"
context="${2:-$here}"
[[ -f "$here/Dockerfile.$engine" ]] || { echo "no recipe for $engine" >&2; exit 1; }
if [[ -d "$context/.git" ]]; then
  echo "building from $(git -C "$context" rev-parse HEAD) in $context"
fi
exec docker build -f "$here/Dockerfile.$engine" -t "sid-gr-bench:$engine" "$context"
