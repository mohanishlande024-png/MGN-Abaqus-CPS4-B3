# Run B3 - stress-concentration prior (SCP) features  (version 3: seven simple numbers)

B3 replaces the 14 anchor features with **7 new numbers per node** built only from classical
closed-form solutions. Everything else is B2-R (Kt target, fresh random edges per graph, mesh
jitter, weight averaging, lr cut + early stop, no load input). Same dataset, no rebuild.

## The idea

Every node is told, with textbook formulas, how strong the stress concentration near it should be;
the network learns the correction. Load along x. At every hole-edge node: local radius rho
(1 / curvature), hole half-size across the load a, and the direction nu the edge faces.

For node i, with b its nearest hole-edge node and d the distance to it:

| # | Number | Formula | In words |
|---|---|---|---|
| 1 | sharpness | S = 1 + 2 sqrt(a / rho) | how sharp the nearest edge point is (Inglis). Straight edge = 1 |
| 2 | facing | c = -cos 2nu | +1 faces across the load, -1 along it (Kirsch) |
| 3 | edge_Kt | K = 1 + (S-1) max(c,0) + 2 min(c,0) | rough Kt on the edge: S if facing across, -1 if along. Circle: exactly Kirsch |
| 4 | near_hole | F(a / (a+d)) | how close to the hole: 1 on the edge, 0 far away |
| 5 | near_tip | F(r / (r+d)), r = min(rho, a) | how close to the sharp spot: drops within mm of a tight corner |
| 6 | Kt_here | 1 + (K-1) near_tip | rough Kt at this node |
| 7 | width_ratio | 2a / W | how much of the plate width the hole blocks (Peterson) |

F(q) = (q^2 + 3q^4)/4 is Kirsch's fall-off (exact above a circular hole). Corners need no separate
treatment: a tight corner has a small rho, so a large S.

## Checked before training

On your Abaqus training data (27 shapes, `python scripts/scp_check.py`): |edge_Kt| at the FE peak vs
FE peak Kt r = 0.94 (log), median error 19 %; Kt_here field median R2 0.41. Von Mises has no sign,
so the check compares |edge_Kt|; this matters only for E_H8, whose FE peak sits at the tip that
lies along the load (edge_Kt = -1 there, magnitude 1 vs FE 1.44).

On 49 local CPS4 geometries:

- The prior's peak follows the FE peak Kt with r = 0.95 (log scale), median error 21 %, with no learning.
- A generic model predicting the peak Kt of a held-out shape from the node's numbers:
  SCP (7 numbers) R2 0.955 / 0.925 (extra trees / random forest) vs the anchors (14) 0.51 / 0.46.
- P3 (triangle) falls inside the training range of edge_Kt (6.5 vs 1.3-12.8); E_V8 (16.7) and
  E_H8 remain outside it.

## Files

| File | What |
|---|---|
| `scripts/run_b3.sh` | NEW - the whole run in one script: check, training, evaluation, fit check, comparison |
| `mgn/scp_features.py` | NEW - the 7 SCP numbers per node (pure numpy, own boundary detection) |
| `mgn/scp_trainer.py` | NEW - `ScpKtMGN`: the B2 model with SCP instead of anchors |
| `scripts/train_b3_val.py` | NEW - B3 training (B2-R settings + SCP); `run_b3.sh stop` (SIGTERM) and Ctrl-C both save a checkpoint |
| `scripts/b3_cps4.sh` | NEW - the single steps: quick, train, status, curves, compare, resume, eval, fitcheck |
| `scripts/scp_check.py` | NEW - how good the 7 numbers are before training, on your Abaqus data (numbers for the thesis) |
| `scripts/eval_unseen_v2.py` | UPDATED - also reads B3 checkpoints ("Run B3"); B1, B2, B2-R unchanged |
| `scripts/check_fit.py` | fit check (unchanged, included so the package is complete) |
| `scripts/plot_style.py` | UPDATED - no more "cursive font not found" messages |
| `mgn/regularize.py`, `scripts/compare_runs.py` | from the B2-R package (unchanged) |

A B3 run trained with an earlier version (1 or 2) cannot be resumed or evaluated with this code:
the scripts stop with a clear message. Delete `results_si_cps4_b3` and train again.

## The whole run in one script (Ubuntu, repo folder, environment active)

    bash scripts/run_b3.sh start       check -> training -> evaluation -> fit check -> comparison
    bash scripts/run_b3.sh status      what is running, last epochs, last R2 checks
    bash scripts/run_b3.sh stop        stops safely (the trainer saves mgn_latest.pt first)
    bash scripts/run_b3.sh start       after a stop: continues from mgn_latest.pt

All predictions (evaluation and fit check) use **`mgn_best_val.pt`**, the weights of the epoch with
the best validation R2. Settings in front of the command, e.g. `DPI=600 bash scripts/run_b3.sh eval`:

| Setting | Default | Meaning |
|---|---|---|
| `EPOCHS` | 5000 | upper limit (early stopping ends the run 1000 epochs after the best epoch) |
| `DPI` | 300 | resolution of every PNG: contour plot 4650 x 2220 px, zoom 4500 x 1680 px |
| `DRAWS` | 8 | random-edge draws averaged per prediction |
| `SCATTER` | test | one scatter plot per case at the unseen loads 8, 60, 130 MPa, for every shape of every test set; `all` also does every training load |

Contour plots keep the standard banded style (about 20 colour bands with round-number edges, as in B1), with no smoothing and no
extra lines; only the resolution is higher. Single steps: `bash scripts/run_b3.sh check | eval | fitcheck | compare`.

Where the results go (`results_si_cps4_b3/`):

| Folder / file | What |
|---|---|
| `unseen_eval_v2_draws8/metrics_all.csv` | R2, RMSE, peak error of every test case |
| `unseen_eval_v2_draws8/contours/<set>/<shape>__<load>.png`, `_zoom.png` | HD contour plots (FE, MGN, error) |
| `unseen_eval_v2_draws8/scatter/<set>/<shape>__<load>.png` | one scatter plot per case |
| `unseen_eval_v2_draws8/scatter_grid_*.png`, `scatter_overlay_*.png`, `summary_*.png` | the overview figures |
| `fit_check/` | fit ladder, verdict, R2 vs epoch, comparison with B2 / B2-R |
| `scp_check/` | the before-training check |
| `pipeline.out` | the full log |

## Single steps with b3_cps4.sh (as before)

    bash scripts/b3_cps4.sh quick
    rm -rf results_si_cps4_b3_quick
    bash scripts/b3_cps4.sh train
    bash scripts/b3_cps4.sh status
    bash scripts/b3_cps4.sh compare
    bash scripts/b3_cps4.sh eval
    bash scripts/b3_cps4.sh fitcheck

## Tested (CPU, small model)

- a circular hole gives edge_Kt = 3 at the top and -1 at the side (Kirsch) and sharpness 3;
  all 27 training shapes finite, 2 s in total; an older-version checkpoint is refused.
- training, early stop, `--resume`, weight averaging; `eval_unseen_v2.py` gives identical results
  from `mgn_best_val.pt` and `multi_geometry_mgn.pt`; B2 and B2-R checkpoints still evaluate as before.
- `run_b3.sh` on a small copy (2-layer model, 4 epochs): start, stop during training (checkpoint
  saved), start again (continued at epoch 2), then evaluation with the best epoch (epoch 3, not the
  last), 18 contour plots at 4650 x 2220 px, 9 single scatter plots, fit check and comparison.
