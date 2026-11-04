#!/usr/bin/env bash
# R1 confirmatory run (EXPERIMENTS_AND_BASELINES.md §R1.4). Requires results_r1/FREEZE.json.
set -uo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
OUT=../results_r1
LOG=$OUT/confirm_logs
mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BETA=$(python3 -c "import json;print(json.load(open('$OUT/settings.json'))['beta'])")
IRA_AGENTS=ira,ira_noactive,ira_closed,ira_nocuetrack,ira_nochange,ira_nolibrary,entity_bayes,ucb,random
JOBS=$LOG/jobs.txt
: > "$JOBS"
# Stage 1: IRA (all variants) + gru_narrow + baselines + accumulation
for U in 4 5 6 7 8; do echo "$PY -m u1.experiment train-ira --universe $U --beta $BETA --final --out $OUT > $LOG/ira_u$U.log 2>&1 && $PY -m u1.experiment eval --universe $U --phase confirm --agents $IRA_AGENTS --resume --out $OUT >> $LOG/ira_u$U.log 2>&1 && $PY -m u1.experiment accumulate --universe $U --phase confirm --out $OUT >> $LOG/ira_u$U.log 2>&1" >> "$JOBS"; done
for U in 4 5 6 7 8; do echo "$PY -m u1.experiment train-metarl --universe $U --arch gru --dist train --tag gru_narrow --out $OUT > $LOG/gru_narrow_u$U.log 2>&1 && $PY -m u1.experiment eval --universe $U --phase confirm --agents gru_narrow --resume --out $OUT >> $LOG/gru_narrow_u$U.log 2>&1" >> "$JOBS"; done
xargs -P 4 -I{} bash -c "{}" < "$JOBS"
$PY -m u1.experiment summarize --out $OUT --phase confirm > $LOG/summary_stage1.log 2>&1
echo "stage 1 summary exit $?"
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/summary_confirm.json'))['stage2_warranted'] else 1)"; then
  echo "stage 2: H1 holds against gru_narrow -> training gru_broad and rel_broad"
  : > "$LOG/jobs2.txt"
  for U in 4 5 6 7 8; do
    echo "$PY -m u1.experiment train-metarl --universe $U --arch rel --dist train_broad --hidden 64 --tag rel_broad --out $OUT > $LOG/rel_broad_u$U.log 2>&1 && $PY -m u1.experiment eval --universe $U --phase confirm --agents rel_broad --resume --out $OUT >> $LOG/rel_broad_u$U.log 2>&1" >> "$LOG/jobs2.txt"
    echo "$PY -m u1.experiment train-metarl --universe $U --arch gru --dist train_broad --tag gru_broad --out $OUT > $LOG/gru_broad_u$U.log 2>&1 && $PY -m u1.experiment eval --universe $U --phase confirm --agents gru_broad --resume --out $OUT >> $LOG/gru_broad_u$U.log 2>&1" >> "$LOG/jobs2.txt"
  done
  xargs -P 4 -I{} bash -c "{}" < "$LOG/jobs2.txt"
  $PY -m u1.experiment summarize --out $OUT --phase confirm > $LOG/summary.log 2>&1
else
  echo "stage 2 skipped: H1 does not hold against gru_narrow, so H1 is not supported whatever stage 2 would show"
fi
echo "R1 confirmatory run complete"
