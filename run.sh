#!/usr/bin/env bash
set -euo pipefail

export BROWSER_ROUTER_GROUP_CHAT_URL="${BROWSER_ROUTER_GROUP_CHAT_URL:-https://chatgpt.com/gg/6a1dc9eb0ab8819590488a5e9d1a859c}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --forever)
      export QUANTALE_LOOP_FOREVER=1
      shift
      ;;
    --ticks)
      export QUANTALE_MAX_TICKS="$2"
      shift 2
      ;;
    --check-topology)
      EXTRA_ARGS+=(--check-topology)
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--forever] [--ticks N] [--check-topology]" >&2
      exit 1
      ;;
  esac
done

rtk cargo run --bin quantale_semiring_v2 -- "${EXTRA_ARGS[@]}"
