#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Building SPIR-V for Vulkan 1.1 ==="

# Direct kernel
glslangValidator -V --target-env vulkan1.1 -o lmz_rans.spv lmz_rans.comp
spirv-val lmz_rans.spv
echo "Built and validated lmz_rans.spv ($(wc -c < lmz_rans.spv | tr -d ' ') bytes)"

# Grid-stride kernel
glslangValidator -V --target-env vulkan1.1 -DSTRIDE -o lmz_rans_stride.spv lmz_rans.comp
spirv-val lmz_rans_stride.spv
echo "Built and validated lmz_rans_stride.spv ($(wc -c < lmz_rans_stride.spv | tr -d ' ') bytes)"

echo "=== SPIR-V build complete ==="
