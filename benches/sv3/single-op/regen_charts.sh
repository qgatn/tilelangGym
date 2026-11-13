#!/bin/bash
# Regen all charts referenced by simt-uarch.md + lds_fp32_vs_bf16.md (clip to last issue).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYV=/home/onasismelchiorjeyakumar/venv/bin/python
TOOLS=${ROOT}/../simt-uarch/tools
PLOTS=${ROOT}/../simt-uarch/plots
pick() { find "build/npusim_$1" -name instr.bin 2>/dev/null | head -1; }

pair() {  # tag chart label
  local bin chart label
  bin=$(pick "$1"); chart="$2"; label="$3"
  [ -z "$bin" ] && { echo "SKIP $1 (no bin)"; return; }
  "$PYV" "$TOOLS/instr_vs_ipc.py" "$bin" simt "$PLOTS/$chart" --label "$label"
  "$PYV" "$TOOLS/warpchart.py" "$bin" "$PLOTS/${chart}_warps" --label "$label, warp activity"
}

# simt-uarch.md
pair pure_ldsk0_t256   lds_s1_scalar_t256   "LDS S1 scalar fp16 (pure), 256 threads"
pair pure_ldsk1_t256   lds_s2_u32_t256      "LDS S2 uint32 pack (pure), 256 threads"
pair pure_ldsk2_t256   lds_s3_half2_t256    "LDS S3 half2 typed (pure), 256 threads"
pair pure_ldsk3_t256   lds_s4_bf16x2_t256   "LDS S4 bfloat16x2 typed (pure), 256t"
pair pure_f2fk0_t256   f2f_s1_scalar_t256   "F2F S1 scalar convert (pure), 256t"
pair pure_f2fk1_t256   f2f_s2_packed_t256   "F2F S2 packed+PRMT (pure), 256 threads"
pair pure_f2fk2_t256   f2f_s3_word_t256     "F2F S3 word-preload (pure), 256t"
pair pure_f2fk3_t256   f2f_s4_barrier_t256  "F2F S4 barrier-aligned (pure), 256t"
pair pure_ffmak0_t256  ffma_s1_scalar_t256  "FFMA S1 scalar (pure), 256 threads"
pair pure_ffmak1_t256  ffma_s2_f2src_t256   "FFMA S2 float2-src (pure), 256t"
pair pure_ffmak2_t256  ffma_s3_barrier_t256 "FFMA S3 barrier-aligned (pure), 256t"
STS=$(find build/npusim_hide32_sts_t256 -name instr.bin 2>/dev/null | head -1)
if [ -n "$STS" ]; then
  "$PYV" "$TOOLS/instr_vs_ipc.py" "$STS" simt "$PLOTS/sts_s1_t256" --label "STS S1 coalesced (pure), 256 threads"
  "$PYV" "$TOOLS/warpchart.py" "$STS" "$PLOTS/sts_s1_t256_warps" --label "STS S1 coalesced, 256t, warp activity"
fi

# lds_fp32_vs_bf16.md
pair lds_pure_fp32_t256   lds_fp32_t256        "LDS fp32 scalar (pure), 256 threads"
pair lds_pure_bf16_t256   lds_bf16_scalar_t256 "LDS bf16 scalar (pure), 256 threads"
pair lds_pure_bf16x2_t256 lds_bf16x2_t256      "LDS bfloat16x2 (pure), 256 threads"
pair lds_pure_fp32_t1024  lds_fp32_t1024       "LDS fp32 scalar (pure), 1024 threads"
pair lds_pure_bf16x2_t1024 lds_bf16x2_t1024    "LDS bfloat16x2 (pure), 1024 threads"

echo REGEN_DONE
