#!/usr/bin/env bash
# Confirmatory run (EXPERIMENTS_AND_BASELINES.md §4). Requires results/FREEZE.json (written by `freeze`).
# Usage: bash run_confirm.sh HIDDEN LR UPDATES
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
OUT=../results
LOG=$OUT/confirm_logs
mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
H=$1; LR=$2; UPD=$3
AGENTS=ira,ira_noactive,ira_closed,ira_nocuetrack,ira_nochange,ira_nolibrary,entity_bayes,ucb,random
chain() {
  local U=$1
  $PY -m u1.experiment train-metarl --universe "$U" --hidden "$H" --lr "$LR" --updates "$UPD" --seed 0 --tag s0 --out $OUT
  $PY -m u1.experiment train-ira --universe "$U" --beta "$(python3 -c "import json;print(json.load(open('$OUT/settings.json'))['beta'])")" --final --out $OUT
  $PY -m u1.experiment eval --universe "$U" --phase confirm --agents $AGENTS,metarl_s0 --resume --out $OUT
  $PY -m u1.experiment accumulate --universe "$U" --phase confirm --out $OUT
}
second_seed() {
  for U in 1 2 3; do
    $PY -m u1.experiment train-metarl --universe "$U" --hidden "$H" --lr "$LR" --updates "$UPD" --seed 1 --tag s1 --out $OUT
  done
}
for U in 1 2 3; do chain "$U" > "$LOG/chain_u$U.log" 2>&1 & done
second_seed > "$LOG/metarl_s1.log" 2>&1 &
wait
for U in 1 2 3; do
  $PY -m u1.experiment eval --universe "$U" --phase confirm --agents metarl_s1 --resume --out $OUT >> "$LOG/metarl_s1.log" 2>&1 &
done
wait
$PY -m u1.experiment summarize --out $OUT --phase confirm > "$LOG/summary.log" 2>&1
echo "confirmatory run complete"
