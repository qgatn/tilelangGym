#!/bin/bash
# Numbered, named charts. Run from simt_ops.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYV="/home/onasismelchiorjeyakumar/venv/bin/python"
PLOTS="${ROOT}/../simt-uarch/plots"
mkdir -p "${PLOTS}"
pick() { find "build/npusim_$1" -name instr.bin 2>/dev/null | head -1; }
STS256=$(find build/npusim_hide32_sts_t256 -name instr.bin 2>/dev/null | head -1)
CASE=$(find ../schedules/runs/ub-n2048-k8-t256 -name instr.bin 2>/dev/null | head -1)

# Titles inside the PNG are short names only; figure numbers live in the .md captions.
${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick lds_packed_t256)" simt "${PLOTS}/lds_packed_t256" --label "LDS packed v2, 256 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "$(pick lds_packed_t256)" "${PLOTS}/lds_packed_t256_warps" --label "LDS packed v2, 256 threads, warp activity"
${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick lds_packed_t128)" simt "${PLOTS}/lds_packed_t128" --label "LDS packed v2, 128 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "$(pick lds_packed_t128)" "${PLOTS}/lds_packed_t128_warps" --label "LDS packed v2, 128 threads, warp activity"

${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick sv3_f2f_t128)" simt "${PLOTS}/f2f_t128" --label "F2F scalar burst, 128 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "$(pick sv3_f2f_t128)" "${PLOTS}/f2f_t128_warps" --label "F2F scalar burst, 128 threads, warp activity"
${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick f2f_packed_t128)" simt "${PLOTS}/f2f_packed_t128" --label "F2F packed B, 128 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "$(pick f2f_packed_t128)" "${PLOTS}/f2f_packed_t128_warps" --label "F2F packed, 128 threads, warp activity"
${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick f2f_packed_t256)" simt "${PLOTS}/f2f_packed_t256" --label "F2F packed B, 256 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "$(pick f2f_packed_t256)" "${PLOTS}/f2f_packed_t256_warps" --label "F2F packed B, 256 threads, warp activity"
${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick f2f_packed_v3_t256)" simt "${PLOTS}/f2f_packed_v3_t256" --label "F2F packed v3 (uint32 preload), 256 threads"
${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick f2f_packed_v4_t256)" simt "${PLOTS}/f2f_packed_v4_t256" --label "F2F packed v4 (+ syncthreads), 256 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "$(pick f2f_packed_v4_t256)" "${PLOTS}/f2f_packed_v4_t256_warps" --label "F2F packed v4 (+ syncthreads), 256 threads, warp activity"

${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick sv3_ffma_t128)" simt "${PLOTS}/ffma_t128" --label "FFMA scalar burst, 128 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "$(pick sv3_ffma_t128)" "${PLOTS}/ffma_t128_warps" --label "FFMA scalar burst, 128 threads, warp activity"
${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick ffma_packed_t256)" simt "${PLOTS}/ffma_packed_t256" --label "FFMA float2-src, 256 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "$(pick ffma_packed_t256)" "${PLOTS}/ffma_packed_t256_warps" --label "FFMA float2-src, 256 threads, warp activity"
${PYV} ../simt-uarch/tools/instr_vs_ipc.py "$(pick ffma_packed_v4_t256)" simt "${PLOTS}/ffma_packed_v4_t256" --label "FFMA v4 (+ syncthreads), 256 threads"

${PYV} ../simt-uarch/tools/instr_vs_ipc.py "${STS256}" simt "${PLOTS}/sts_t256" --label "STS coalesced, 256 threads"
${PYV} ../simt-uarch/tools/warpchart.py  "${STS256}" "${PLOTS}/sts_t256_warps" --label "STS coalesced, 256 threads, warp activity"

${PYV} ../simt-uarch/tools/instr_vs_ipc.py "${CASE}" simt "${PLOTS}/case_ub_k8" --label "Base case, ub-n2048-k8-t256"
for c in methodo-n2048-k8-t256 serial-n2048-k8-t256; do
  BIN=$(find ../schedules/runs/$c -name instr.bin 2>/dev/null | head -1)
  [ -n "${BIN}" ] && ${PYV} ../simt-uarch/tools/instr_vs_ipc.py "${BIN}" simt "${PLOTS}/${c%%-*}_ub_k8" --label "$c"
done
echo CHARTS_DONE
