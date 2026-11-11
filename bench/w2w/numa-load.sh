#!/usr/bin/env bash
# Put the GGUF's page cache on one NUMA node: drop this file's cached pages (dd's nocache advice, this file
# only), read it back bound to the node, show the split. usage: numa-load.sh <file> <node>
set -u
F=$1; N=$2
dd if="$F" iflag=nocache count=0 status=none
numactl --membind=$N dd if="$F" of=/dev/null bs=16M status=none
numastat -m | grep -E "FilePages"
