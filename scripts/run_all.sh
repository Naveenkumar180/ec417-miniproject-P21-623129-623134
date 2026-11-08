#!/usr/bin/env bash
# Run every configuration for 3 seeds. Appends to results/metrics.csv.
# Delete results/metrics.csv first for a clean run.
set -e
for seed in 0 1 2; do
  python -m src.train --model majority --seed $seed
  python -m src.train --model rf       --seed $seed
  python -m src.train --model lstm     --seed $seed
  python -m src.train --model bilstm   --seed $seed
  python -m src.train --model cnnlstm            --seed $seed   # ablation: conv front end only
  python -m src.train --model bilstm   --weighted --seed $seed  # ablation: weights only
  python -m src.train --model cnnlstm  --weighted --seed $seed  # full improvement
done
python -m src.evaluate
