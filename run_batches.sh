#!/bin/zsh
# Cache JVS batches 1..7 (3 MPS workers) and evaluate cumulatively after each.
set -e
files=()
for b in 1 2 3 4 5 6 7; do
  uv run python cache_parallel.py batch${b}_jobs.json jvs_batch${b}.pkl 3 1 mps 2>/dev/null | tail -1 | sed "s/^/batch $b: /" >> batch_throughput.log
  files+=(jvs_batch${b}.pkl)
  uv run python batch_eval.py "+${b} batch" ${files[@]} 2>/dev/null
done
echo ALL DONE >> batch_throughput.log
