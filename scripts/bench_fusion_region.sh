#!/usr/bin/env bash
set -euo pipefail

iterations="${1:-1000}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="state/benchmarks/fusion_region_${stamp}.json"

mkdir -p state/benchmarks
cargo run --release --bin bench_fusion_region -- "${iterations}" | tee "${out}"
printf 'wrote %s\n' "${out}" >&2
