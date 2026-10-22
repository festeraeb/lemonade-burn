#!/bin/bash
# Start a private lemond (custom build with the lemonade-burn backend) on a spare
# port, CPU only. lemonade-cake-bridge and lemonade-burn-server must be in BIN.
# Register the model first in $RUN/config/user_models.json, e.g.
#   "bitnet-b1.58-2B-4T-burn": {"checkpoint": "/data/lemonade-burn/models/bitnet-b1.58-2B-4T-lbt",
#     "recipe": "lemonade-burn", "labels": ["chat","custom"], "source": "huggingface",
#     "registry_source": "huggingface", "suggested": true}
# ("source" must be huggingface/modelscope for lemond to accept the entry; the
#  backend's BurnOps treats the checkpoint as a local dir and never downloads.)
set -euo pipefail
LEMOND="${LEMOND:-/data/lemonade-custom/build/lemond}"
RUN="${RUN:-/data/lemonade-custom/run}"
BIN="${BIN:-/data/lemonade-burn/bin}"
PORT="${PORT:-13399}"
cd "$RUN"
env PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES= GGML_DISABLE_VULKAN=1 LB_THREADS="${LB_THREADS:-16}" \
  HOME="$RUN/home" XDG_CACHE_HOME="$RUN/xdg/cache" XDG_CONFIG_HOME="$RUN/xdg/config" \
  XDG_DATA_HOME="$RUN/xdg/data" XDG_STATE_HOME="$RUN/xdg/state" \
  setsid -f "$LEMOND" --port "$PORT" --host 127.0.0.1 --no-broadcast --log-file disabled \
  "$RUN/cache" "$RUN/config" > "$RUN/lemond-burn.log" 2>&1 < /dev/null
curl -s --retry 20 --retry-connrefused --retry-delay 1 -o /dev/null -w "lemond health %{http_code}\n" "http://127.0.0.1:$PORT/api/v1/health"
ss -ltnp | grep ":$PORT " || true
