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

printf '%s\n' \
  '{"id":"repo","label":"Repository","properties":{"name":"smoke","stars":3},"vectors":[[1.0,0.0,0.0,0.0]]}' \
  '{"id":"issue","label":"Issue","properties":{"title":"bounded recovery","open":true},"vectors":[[0.9,0.1,0.0,0.0]]}' \
  '{"id":"pr","label":"PullRequest","properties":{"title":"repair torn tail","open":false},"vectors":[[0.8,0.2,0.0,0.0]]}' \
  > "$VECGRA_SMOKE_DIR/nodes.jsonl"

printf '%s\n' \
  '{"source":"repo","target":"issue","label":"HAS_ISSUE","properties":{"position":1},"vectors":[[0.7,0.3,0.0,0.0]]}' \
  '{"source":"pr","target":"issue","label":"CLOSES","properties":{"confidence":1.0},"vectors":[[0.6,0.4,0.0,0.0]]}' \
  > "$VECGRA_SMOKE_DIR/edges.jsonl"

"$vecgra_binary" import-jsonl \
  "$VECGRA_SMOKE_DIR/nodes.jsonl" \
  "$VECGRA_SMOKE_DIR/edges.jsonl" \
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
  'MATCH (p:PullRequest)-[r:CLOSES]->(i:Issue) RETURN p,r,i LIMIT 10')"
grep -q 'CLOSES' <<<"$query"

"$vecgra_binary" compact \
  "$VECGRA_SMOKE_DIR/smoke.vg" \
  "$VECGRA_SMOKE_DIR/compact.vg" f32
compact_integrity="$($vecgra_binary check "$VECGRA_SMOKE_DIR/compact.vg")"
grep -q $'status\tok' <<<"$compact_integrity"

echo "Vecgra smoke workflow passed"
