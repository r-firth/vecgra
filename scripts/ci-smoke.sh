#!/usr/bin/env bash
set -euo pipefail

vecgra_binary="${1:-target/release/vecgra}"
if [[ ! -x "$vecgra_binary" ]]; then
  echo "Vecgra binary is not executable: $vecgra_binary" >&2
  exit 1
fi

"$vecgra_binary" --version | grep -Eq '^vecgra [0-9]+\.[0-9]+\.[0-9]+'

VECGRA_SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vecgra-smoke.XXXXXX")"
export VECGRA_SMOKE_DIR
trap 'rm -rf -- "$VECGRA_SMOKE_DIR"' EXIT

"$vecgra_binary" import-jsonl --help | grep -q 'Each nonblank node line is JSON'

"$vecgra_binary" import-jsonl \
  examples/custom-data/nodes.jsonl \
  examples/custom-data/edges.jsonl \
  "$VECGRA_SMOKE_DIR/smoke.vg" 4 f16

stats="$($vecgra_binary stats "$VECGRA_SMOKE_DIR/smoke.vg")"
grep -q $'nodes\t3' <<<"$stats"
grep -q $'edges\t2' <<<"$stats"
grep -q $'vectors\t5' <<<"$stats"

if "$vecgra_binary" stats "$VECGRA_SMOKE_DIR/smoke.vg" unexpected >/dev/null 2>&1; then
  echo "Vecgra accepted an unexpected command argument" >&2
  exit 1
fi

integrity="$($vecgra_binary check "$VECGRA_SMOKE_DIR/smoke.vg")"
grep -q $'status\tok' <<<"$integrity"

query="$($vecgra_binary query "$VECGRA_SMOKE_DIR/smoke.vg" \
  'MATCH (c:Customer)-[r:PURCHASED]->(p:Product) RETURN c,r,p LIMIT 10')"
grep -q 'PURCHASED' <<<"$query"

"$vecgra_binary" compact \
  "$VECGRA_SMOKE_DIR/smoke.vg" \
  "$VECGRA_SMOKE_DIR/compact.vg" f32
compact_integrity="$($vecgra_binary check "$VECGRA_SMOKE_DIR/compact.vg")"
grep -q $'status\tok' <<<"$compact_integrity"

echo "Vecgra smoke workflow passed"
