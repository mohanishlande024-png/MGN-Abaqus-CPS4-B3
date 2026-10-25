#!/usr/bin/env bash
# ======================================================================================
#  RUN B3 - the complete pipeline in ONE script  (Ubuntu, from the repo folder)
#
#    bash scripts/run_b3.sh start      everything, in the background (safe to close SSH):
#                                        1 SCP check   (how good the 7 numbers are before training)
#                                        2 training    (stops by itself 1000 epochs after the best epoch)
#                                        3 evaluation  with the BEST-EPOCH model (mgn_best_val.pt):
#                                          metrics, HD contour plots, one scatter plot per test case
#                                        4 fit check   (over/under-fitting verdict, with the best-epoch model)
#                                        5 comparison  with B2 and B2-R (the runs you have)
#    bash scripts/run_b3.sh status     what is running, last training lines, which step
#    bash scripts/run_b3.sh stop       stop safely (the trainer saves a checkpoint first)
#    bash scripts/run_b3.sh start      again after a stop: continues where it stopped
#
#    single steps (when you want to redo one):
#    bash scripts/run_b3.sh check | eval | fitcheck | compare
#
#  Settings (put in front of the command, e.g.  DPI=600 bash scripts/run_b3.sh eval):
#    EPOCHS=5000   upper limit of training epochs (early stopping usually ends sooner)
#    DPI=300       resolution of every PNG (300 = 4650 x 2220 px contour plots; 600 doubles it)
#    DRAWS=8       random-edge draws averaged per prediction
#    SCATTER=test  one scatter plot per case at the unseen test loads (8, 60, 130 MPa), every
#                  shape of every test set; SCATTER=all also does every training load (~500 files)
#
#  Contour plots: the standard banded style (about 20 colour bands, as in B1), no smoothing and no
#  extra lines - only the resolution is high.
#  Activate the environment first:   source ~/mgn/.venv/bin/activate
# ======================================================================================

set -e
cd "$(dirname "$0")/.."

D=${DATA_DIR:-data/si_cps4}
DATA=$D/dataset_si_cps4_aug20.pt
RES=${RES:-results_si_cps4_b3}
CKPT=$RES/mgn_best_val.pt
EPOCHS=${EPOCHS:-5000}
DPI=${DPI:-300}
DRAWS=${DRAWS:-8}
SCATTER=${SCATTER:-test}
TRAIN_OPTS=${TRAIN_OPTS:-}            # advanced: extra options for train_b3_val.py
TESTS="--train-data $DATA --loads-dir $D/unseen_loads --interp-dir $D/unseen_interp --extrap-dir $D/unseen_extrap --manifest $D/si_manifest.csv --units MPa --draws $DRAWS"
EVAL_OPTS="--dpi $DPI --bands 20 --contour-loads test --scatter-loads $SCATTER"
EVAL_DIR=$RES/unseen_eval_v2
if [ "$DRAWS" -gt 1 ]; then EVAL_DIR=${EVAL_DIR}_draws$DRAWS; fi
PIPE='run_b3\.sh pipeline'
TRAINER='scripts/train_b3_val\.py'
LOG=$RES/pipeline.out

usage() { sed -n '2,31p' "$0"; }
running() { pgrep -f "$PIPE" > /dev/null || pgrep -f "$TRAINER" > /dev/null; }
step() { echo; echo "=================================================================================="; echo "$(date '+%Y-%m-%d %H:%M')   $*"; echo "=================================================================================="; }

checks() {
    if ! python -c "import torch" 2>/dev/null; then
        echo "python cannot import torch - activate the environment first:  source ~/mgn/.venv/bin/activate"; exit 1
    fi
    for f in "$DATA" "$D/si_manifest.csv"; do
        [ -f "$f" ] || { echo "missing $f"; exit 1; }
    done
    for d in "$D/val" "$D/unseen_loads" "$D/unseen_interp" "$D/unseen_extrap"; do
        [ -d "$d" ] || { echo "missing folder $d"; exit 1; }
    done
    for f in mgn/scp_features.py mgn/scp_trainer.py mgn/regularize.py scripts/train_b3_val.py scripts/eval_unseen_v2.py \
             scripts/check_fit.py scripts/compare_runs.py scripts/scp_check.py; do
        [ -f "$f" ] || { echo "missing $f - extract the B3 package into the repo first"; exit 1; }
    done
}

do_check()    { python scripts/scp_check.py --train-data "$DATA" --dirs "$D/val" "$D/unseen_interp" "$D/unseen_extrap" --out "$RES/scp_check"; }
do_eval()     { [ -f "$CKPT" ] || { echo "no $CKPT yet - train first"; exit 1; }
                python scripts/eval_unseen_v2.py --ckpt "$CKPT" $TESTS $EVAL_OPTS; }
do_fitcheck() { [ -f "$CKPT" ] || { echo "no $CKPT yet - train first"; exit 1; }
                python scripts/check_fit.py --ckpt "$CKPT" --train-data "$DATA" --val-dir "$D/val" --draws "$DRAWS"; }
do_compare()  { RUNS=""; LABS=""
                for r in b2:B2 b2r:B2-R b3:B3; do f=results_si_cps4_${r%%:*}; [ "$r" = "b3:B3" ] && f=$RES
                    if [ -f "$f/r2_log.csv" ]; then RUNS="$RUNS $f"; LABS="$LABS ${r##*:}"; fi; done
                [ -n "$RUNS" ] && python scripts/compare_runs.py $RUNS --labels $LABS --out "$RES/fit_check/compare_runs.png"; }

case "$1" in
  ""|help) usage ;;

  start|resume)
    checks
    if running; then echo "B3 is already running:"; pgrep -af "$PIPE|$TRAINER"; exit 1; fi
    mkdir -p "$RES"
    nohup bash scripts/run_b3.sh pipeline >> "$LOG" 2>&1 &
    sleep 5
    if running; then echo "started (PID $!).  Progress:  bash scripts/run_b3.sh status      Full log: $LOG"
    else echo "it stopped straight away - the error:"; tail -n 25 "$LOG"; exit 1; fi ;;

  pipeline)                                     # called by 'start'; runs in the background
    checks
    step "B3 PIPELINE   epochs <= $EPOCHS, dpi $DPI, draws $DRAWS, scatter loads: $SCATTER"
    step "1/5  SCP check before training"
    do_check || echo "(SCP check failed - continuing)"
    if [ -f "$RES/multi_geometry_mgn.pt" ]; then
        step "2/5  training - already finished ($RES/multi_geometry_mgn.pt exists), skipped"
    else
        EXTRA=""; [ -f "$RES/mgn_latest.pt" ] && EXTRA="--resume"
        step "2/5  training ${EXTRA:+(continuing from the last checkpoint)}"
        python -u scripts/train_b3_val.py --data "$DATA" --val-dir "$D/val" --results "$RES" --epochs "$EPOCHS" $EXTRA $TRAIN_OPTS
        if [ ! -f "$RES/multi_geometry_mgn.pt" ]; then
            echo "training was stopped before the end - run 'bash scripts/run_b3.sh start' to continue"; exit 0
        fi
    fi
    step "3/5  evaluation with the best-epoch model ($CKPT)"
    do_eval
    step "4/5  fit check with the best-epoch model"
    do_fitcheck
    step "5/5  comparison with the other runs"
    do_compare || echo "(comparison skipped)"
    step "DONE - results:"
    grep -h "^best validation R2" "$LOG" | tail -n 1 | sed 's/^/  /'
    python - "$RES/fit_check/fit_ladder.csv" <<'PY' || true
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print("  %-15s %6s %9s %9s  %s" % ("level", "files", "mean R2", "worst R2", "mean |peak error|"))
for r in rows:
    print("  %-15s %6s %9.4f %9.4f  %.1f %%" % (r["level"], r["files"], float(r["mean_r2"]), float(r["worst_r2"]),
                                             float(r["mean_abs_peak_err_pct"])))
PY
    echo "  metrics (every test case)   : $EVAL_DIR/metrics_all.csv"
    echo "  contour plots (HD)          : $EVAL_DIR/contours/<set>/<shape>__<load>.png and _zoom.png"
    echo "  scatter, one per case       : $EVAL_DIR/scatter/<set>/<shape>__<load>.png"
    echo "  scatter, all together       : $EVAL_DIR/scatter_grid_*.png, scatter_overlay_*.png"
    echo "  summaries                   : $EVAL_DIR/summary_*.png"
    echo "  fit check                   : $RES/fit_check/"
    echo "  before-training check       : $RES/scp_check/"
    ;;

  status)
    if running; then echo "RUNNING:"; pgrep -af "$PIPE|$TRAINER" | grep -v pgrep || true; else echo "not running"; fi
    echo "--- current step"
    grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]+ " "$LOG" 2>/dev/null | tail -n 1 || echo "none yet"
    echo "--- last epochs: epoch, loss, seconds"
    [ -f "$RES/loss_log.csv" ] && tail -n 3 "$RES/loss_log.csv" || echo "none yet"
    echo "--- last R2 checks: epoch, train R2, validation R2, lr"
    [ -f "$RES/r2_log.csv" ] && tail -n 3 "$RES/r2_log.csv" | cut -d, -f1,3,8,14 || echo "none yet"
    grep -hE "best, saved|EARLY STOP|learning rate cut" "$LOG" 2>/dev/null | tail -n 3 || true ;;

  stop)
    if ! running; then echo "not running"; exit 0; fi
    pkill -f "$PIPE" 2>/dev/null || true
    pkill -f "$TRAINER" 2>/dev/null || true               # SIGTERM: the trainer saves mgn_latest.pt, then exits
    for i in $(seq 1 60); do pgrep -f "$TRAINER" > /dev/null || break; sleep 2; done
    pkill -f "scripts/eval_unseen_v2\.py --ckpt $CKPT" 2>/dev/null || true
    echo "stopped. Continue later with:  bash scripts/run_b3.sh start" ;;

  check)    checks; do_check ;;
  eval)     checks; do_eval ;;
  fitcheck) checks; do_fitcheck ;;
  compare)  do_compare ;;
  *) echo "unknown step '$1'"; usage; exit 1 ;;
esac
