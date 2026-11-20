#!/usr/bin/env bash
# Build, run, and dump PTX + SASS for both unpack arms.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ')
echo "GPU compute cap: $ARCH"   # GB10 -> 12.1? use sm_121 or sm_120
SASS_ARCH="sm_${ARCH/./}"
mkdir -p out

nvcc -O3 -std=c++17 -arch "$SASS_ARCH" gemv_fp16x2.cu -o out/gemv_fp16x2
./out/gemv_fp16x2 | tee out/run.txt

# PTX (virtual arch)
PTX_ARCH="compute_${ARCH/./}"
nvcc -O3 -std=c++17 -ptx -arch "$PTX_ARCH" gemv_fp16x2.cu -o out/gemv_fp16x2.ptx
# SASS for both kernels
cuobjdump -sass out/gemv_fp16x2 > out/gemv_fp16x2.sass
echo "=== files ==="
ls -la out/
