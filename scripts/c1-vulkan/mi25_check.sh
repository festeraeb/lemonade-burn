#!/bin/bash
# Run on c1 the moment the MI25 (Vega 10, gfx900) is in the box. Read-only apart from the log files it writes under $V/results.
# 1 detect  2 vulkaninfo  3 kernel parity  4 decode test  5 temperature log
# Usage: mi25_check.sh [wgpu-index]   (no argument: the index is found from vulkaninfo's device order)
V=${V:-/mnt/models/vk}; R=$V/results; mkdir -p $R; D=$1
echo "== 1 detect"; lspci -nn | grep -i -E '1002:6860|Vega 10|Instinct' || echo "MI25 NOT on the bus (lspci shows no 1002:6860): reseat the card / check the 8-pin power / 'Above 4G decoding' in RBSU"
dmesg 2>/dev/null | grep -i -E 'amdgpu.*(vega|gfx9|VRAM|fail|error)' | tail -8 || true
for c in /sys/class/drm/card[0-9]*/device; do b=$(basename $(readlink -f $c)); echo "$c $b vram $(( $(cat $c/mem_info_vram_total 2>/dev/null || echo 0) >> 20 )) MiB, link $(cat $c/current_link_speed 2>/dev/null) x$(cat $c/current_link_width 2>/dev/null), driver $(basename $(readlink -f $c/driver 2>/dev/null))"; done
echo "== 2 vulkaninfo (the order below is the wgpu discrete:N order; read it, do not assume)"
vulkaninfo --summary 2>/dev/null | grep -E '^GPU|deviceName|deviceType|driverName|driverInfo' | paste - - - - - 2>/dev/null | head
if [ -z "$D" ]; then D=$(vulkaninfo --summary 2>/dev/null | awk '/deviceType/ && /DISCRETE/ {n++} /deviceName/ && /Vega|MI25|Instinct/ {print n-1; exit}'); fi
[ -n "$D" ] || { echo "no Vega device in vulkaninfo (RADV may need the card initialised by amdgpu first: check dmesg)"; exit 2; }
echo "MI25 = wgpu discrete:$D"
vulkaninfo 2>/dev/null | awk '/GPU id/ {g++} g==('$D'+1) && /shaderFloat16|shaderInt8|shaderInt64|subgroupSize|shaderIntegerDotProduct|integerDotProduct8BitSignedAccelerated|storageBuffer16BitAccess|maxStorageBufferRange/ {print}' | head -14
echo "== 3 kernel parity vs the CUDA P100 dump (identical seeded inputs)"
LB_WGPU_DEVICE=discrete:$D $V/bin/kernel_parity dump $R/kp-wgsl-mi25.bin 2>&1 | tail -2
$V/bin/kernel_parity cpu $R/kp-wgsl-mi25.bin
[ -f $V/kp-cuda-0.bin ] && $V/bin/kernel_parity cmp $R/kp-wgsl-mi25.bin $V/kp-cuda-0.bin
echo "== 4 decode test: 24B teacher-forced vs the CUDA reference, first prompt (322 positions), cache 8192 MiB"
( NP=1 $V/c1_agree.sh mi25-8192 $D 8192 ) &
APID=$!
echo "== 5 temperature log every 10 s while the decode runs (edge/junction/mem, PPT) -> $R/mi25-temps.log"
H=""; for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = amdgpu ] && [ "$(readlink -f $h/device | grep -c $(lspci -D | grep -i 'Vega 10' | cut -d' ' -f1 | head -1))" = 1 ] && H=$h; done
while kill -0 $APID 2>/dev/null; do echo "$(date +%T) $(for f in temp1_input temp2_input temp3_input; do [ -f $H/$f ] && echo -n "$f=$(( $(cat $H/$f) / 1000 ))C "; done) $( [ -f $H/power1_average ] && echo -n "P=$(( $(cat $H/power1_average) / 1000000 ))W")" >> $R/mi25-temps.log; ping -c 10 127.0.0.1 >/dev/null; done
wait $APID; tail -5 $R/mi25-temps.log
echo "WATCH: passive card in a tower: if edge > 85C or junction > 100C stop the run; amdgpu throttles at the crit values and powers off at 'emergency'."
