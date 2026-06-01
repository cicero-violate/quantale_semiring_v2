#!/usr/bin/env bash
set -euo pipefail

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
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--forever] [--ticks N]" >&2
      exit 1
      ;;
  esac
done

rtk cargo run --bin quantale_semiring_v2
