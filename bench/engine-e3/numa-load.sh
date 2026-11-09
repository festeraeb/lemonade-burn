#!/usr/bin/env bash
# Move the GGUF's page cache onto the card's NUMA node: drop this file's cached pages, read it back
# bound to that node, report the node split. Only touches this file's cache. usage: numa-load.sh
set -u
. "$(dirname "$0")/env.sh"
[ -n "$NUMA" ] || { echo "single NUMA node: nothing to bind"; exit 0; }
python3 -c "
import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.fsync(fd) if False else None; os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" "$LB_MODEL"
numastat -m | grep -E "FilePages"
numactl --membind=$CARD_NODE dd if="$LB_MODEL" of=/dev/null bs=16M status=none
numastat -m | grep -E "FilePages"
