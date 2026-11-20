#!/bin/bash
# memwatch.sh <card dir name e.g. card2> <seconds>: print vram/gtt used (MiB) every 2 s
C=/sys/class/drm/$1/device
for i in $(seq 1 $(( $2 / 2 ))); do echo "$(date +%T) vram=$(( $(cat $C/mem_info_vram_used) >> 20 )) gtt=$(( $(cat $C/mem_info_gtt_used) >> 20 )) vis=$(( $(cat $C/mem_info_vis_vram_used) >> 20 )) busy=$(cat $C/gpu_busy_percent) mclk=$(grep '\*' $C/pp_dpm_mclk | cut -d: -f2 | tr -d ' *')"; ping -c 2 127.0.0.1 >/dev/null; done
