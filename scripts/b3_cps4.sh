#!/usr/bin/env bash
# Run B3 = B2-R with the stress-concentration prior (SCP) features instead of the anchors, CPS4 SI dataset:
#
#   bash scripts/b3_cps4.sh quick       20-epoch timing run -> results_si_cps4_b3_quick/ (delete it afterwards)
#   bash scripts/b3_cps4.sh train       the full run in the background (stops by itself; safe to close SSH)
#   bash scripts/b3_cps4.sh status      is it running? last lines of both logs
#   bash scripts/b3_cps4.sh curves      R2 vs epoch plot, train + validation (works while training runs)
#   bash scripts/b3_cps4.sh compare     B2, B2-R and B3 curves on one plot (the runs you have)
#   bash scripts/b3_cps4.sh resume      continue a stopped run, in the background
#   bash scripts/b3_cps4.sh eval        the three tests, 8 edge draws, contours + scatter plots
#   bash scripts/b3_cps4.sh fitcheck    over/under-fitting verdict + plots (after eval)
#
# EPOCHS=4000 bash scripts/b3_cps4.sh train    changes the upper limit (default 5000; early stopping
#                                              ends the run 1000 epochs after the best validation R2)
# Same training graphs as B1 / B2 (data/si_cps4/dataset_si_cps4_aug20.pt) - no rebuild.
# Activate the environment first:   source ~/mgn/.venv/bin/activate

set -e
cd "$(dirname "$0")/.."

D=data/si_cps4
DATA=$D/dataset_si_cps4_aug20.pt
RES=results_si_cps4_b3
CKPT=$RES/mgn_best_val.pt
EPOCHS=${EPOCHS:-5000}
TESTS="--train-data $DATA --loads-dir $D/unseen_loads --interp-dir $D/unseen_interp --extrap-dir $D/unseen_extrap --manifest $D/si_manifest.csv --units MPa --draws 8"

if [ -z "$1" ] || [ "$1" = "help" ]; then sed -n '2,17p' "$0"; exit 0; fi
if ! python -c "import torch" 2>/dev/null; then
    echo "python cannot import torch - activate the environment first:  source ~/mgn/.venv/bin/activate"
    exit 1
fi
if [ ! -f "$DATA" ]; then
    echo "No $DATA - build it once (the same graphs as B1):  python -m mgn.dataset --raw $D/train --aug-perc 0.2 --out $DATA"
    exit 1
fi
PAT='^[^ ]*python[^ ]* (-u )?scripts/train_b3_val\.py'          # a python process running the B3 trainer
running() { pgrep -f "$PAT" > /dev/null; }

case "$1" in
  quick)
    python scripts/train_b3_val.py --data $DATA --val-dir $D/val --epochs 20 --eval-every 10 --results ${RES}_quick ;;
  train|resume)
    if running; then echo "A B3 training run is already going (bash scripts/b3_cps4.sh status). Not starting a second one."; exit 1; fi
    mkdir -p $RES
    EXTRA=""; [ "$1" = "resume" ] && EXTRA="--resume"
    nohup python -u scripts/train_b3_val.py --data $DATA --val-dir $D/val --results $RES --epochs $EPOCHS $EXTRA >> $RES/train.out 2>&1 &
    sleep 5
    if running; then echo "started (PID $!). Output: $RES/train.out   Check with: bash scripts/b3_cps4.sh status";
    else echo "it stopped straight away - the error:"; tail -n 20 $RES/train.out; exit 1; fi ;;
  status)
    if running; then echo "RUNNING:"; pgrep -af "$PAT"; else echo "not running"; fi
    echo "--- last epochs: epoch, loss, seconds"
    if [ -f $RES/loss_log.csv ]; then tail -n 3 $RES/loss_log.csv; else echo "none yet"; fi
    echo "--- last R2 checks: epoch, train R2, validation R2 (averaged weights), lr, validation R2 (raw weights)"
    if [ -f $RES/r2_log.csv ]; then tail -n 3 $RES/r2_log.csv | cut -d, -f1,3,8,14,15; else echo "none yet"; fi
    grep -E "EARLY STOP|learning rate cut" $RES/train.out 2>/dev/null | tail -n 3 || true ;;
  curves)
    python scripts/check_fit.py --curves-only --ckpt $CKPT
    echo "plot: $RES/fit_check/r2_vs_epoch.png" ;;
  compare)
    RUNS=""; LABS=""
    for r in b2:B2 b2r:B2-R b3:B3; do f=results_si_cps4_${r%%:*}; [ -f $f/r2_log.csv ] && RUNS="$RUNS $f" && LABS="$LABS ${r##*:}"; done
    python scripts/compare_runs.py $RUNS --labels $LABS ;;
  eval)
    python scripts/eval_unseen_v2.py --ckpt $CKPT $TESTS ;;
  fitcheck)
    python scripts/check_fit.py --ckpt $CKPT --train-data $DATA --val-dir $D/val --draws 8 ;;
  *)
    echo "unknown step '$1'"; sed -n '2,17p' "$0"; exit 1 ;;
esac
