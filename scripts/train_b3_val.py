r"""
Train RUN B3  =  B2-R (Kt target + regularisation) with the STRESS-CONCENTRATION PRIOR (SCP)
features of mgn/scp_features.py in place of the anchor features. Same dataset.

    bash scripts/b3_cps4.sh train                                   (the easy way)
    python scripts/train_b3_val.py                                  defaults = the CPS4 SI dataset
    python scripts/train_b3_val.py --epochs 20 --eval-every 10 --results results_si_cps4_b3_quick
    python scripts/train_b3_val.py --resume                         continue after a stop

THE GEOMETRY INPUT  (mgn/scp_features.py explains each number)
    Every node gets 7 simple numbers from textbook formulas (Kirsch, Inglis, Peterson):
    sharpness, facing, edge_Kt, near_hole, near_tip, Kt_here, width_ratio. They enter every
    layer through the global encoder: global input [load, 7 SCP numbers].

EVERYTHING ELSE = B2-R
    model (20 layers, hidden 64), Adam lr 1e-5, batch 1, the paper's _train_epoch(), Kt target,
    prediction = load x Kt, fresh random edges per graph, mesh jitter, weight averaging,
    lr cut on plateau + early stop, no load input. Switches as in train_b2r_val.py.

WRITES  (results_si_cps4_b3/)
    loss_log.csv, r2_log.csv, mgn_best_val.pt, mgn_best.pt, mgn_latest.pt, scp_summary.csv,
    multi_geometry_mgn.pt (= the best-validation weights)
    scripts/eval_unseen_v2.py and scripts/check_fit.py read them (reported as "Run B3").
"""

import argparse
import os
import signal
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mgn.dataset import load_dataset                                         # noqa: E402
from mgn.kt_trainer import kt_targets                                        # noqa: E402
from mgn.scp_trainer import ScpKtMGN                                         # noqa: E402
from mgn.scp_features import SCP_NAMES, scp_features                         # noqa: E402
from mgn.regularize import GraphRefresher, WeightEMA, adamw_groups           # noqa: E402
from train_b2_val import read_val, evaluate, truncate_log, fmt, short, R2_COLUMNS  # noqa: E402

DEFAULT_DATA = os.path.join("data", "si_cps4", "dataset_si_cps4_aug20.pt")
DEFAULT_VAL = os.path.join("data", "si_cps4", "val")
DEFAULT_RESULTS = "results_si_cps4_b3"
COLUMNS = R2_COLUMNS + ["lr", "val_r2_raw"]


def scp_summary(mgn, cases, path=None, title="SCP FEATURES (stress-concentration prior)"):
    """One line per shape: holes, half-size across the load, width ratio, largest edge_Kt."""
    seen, rows = set(), []
    for c in cases:
        if c.geometry in seen:
            continue
        seen.add(c.geometry)
        _, info = scp_features(np.asarray(c.coordinates), mgn._mesh_edges(c), return_info=True)
        rows.append((c.geometry, info["n_holes"], info["a"], info["beta"], info["max_edge_Kt"]))
    if path:
        with open(path, "w") as fh:
            fh.write("geometry,holes,half_size_across_load,width_ratio,max_edge_Kt\n")
            for r in rows:
                fh.write("%s,%d,%.4f,%.4f,%.4f\n" % r)
    print("\n%s  (edge_Kt = Kirsch-Inglis estimate of Kt on the hole edge)" % title)
    print("  %-26s %6s %10s %8s %12s" % ("geometry", "holes", "a", "2a/W", "max edge_Kt"))
    for r in rows:
        print("  %-26s %6d %10.2f %8.3f %12.2f" % r)


def one_per_shape(cases):
    seen, out = set(), []
    for c in cases:
        if c.geometry not in seen:
            seen.add(c.geometry)
            out.append(c)
    return out


def _stop_request(signum, frame):
    """'bash scripts/run_b3.sh stop' sends SIGTERM: stop like Ctrl-C, so the checkpoint is saved."""
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _stop_request)
    signal.signal(signal.SIGINT, signal.default_int_handler)     # Ctrl-C works even when started with nohup
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--val-dir", default=DEFAULT_VAL)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--epochs", type=int, default=5000, help="upper limit; early stopping usually ends sooner")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embedding", type=int, default=16)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # ---- regularisation
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="max interior-node move, as a fraction of its shortest mesh edge (0 = off)")
    ap.add_argument("--no-per-copy", action="store_true",
                    help="one random-edge draw per shape per epoch, as in B2 (default: one per graph)")
    ap.add_argument("--ema", type=float, default=0.9995,
                    help="weight-averaging decay per step (0 = off; 0.9995 ~ the last 2000 steps)")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0 = plain Adam, as B2)")
    ap.add_argument("--no-plateau", action="store_true", help="constant learning rate, no early stop (as B2)")
    ap.add_argument("--lr-patience", type=int, default=300, help="epochs without a better validation R2 -> lr x 0.5")
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--stop-patience", type=int, default=1000, help="epochs without a better validation R2 -> stop")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="smallest validation R2 gain that counts")
    ap.add_argument("--load-input", action="store_true", help="keep the load as an input (as in B2)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    if not os.path.isfile(a.data):
        print("No %s.\nBuild it first:  python -m mgn.dataset --raw data/si_cps4/train --aug-perc 0.2 --out %s"
              % (a.data, a.data))
        return 1
    cases, aug = load_dataset(a.data, return_meta=True)
    if aug["aug_perc"] <= 0:
        print("%s has no random edges (aug_perc = 0) - B2-R needs the edge-augmented dataset." % a.data)
        return 1
    val_cases = read_val(a.val_dir, aug) if os.path.isdir(a.val_dir) else []
    if not val_cases:
        print("No validation files in %s." % a.val_dir)
        return 1
    geoms = sorted(set(c.geometry for c in cases))
    val_geoms = sorted(set(c.geometry for c in val_cases))
    both = sorted(set(geoms) & set(val_geoms))
    if both:
        print("STOP: %s is in the training data AND in the validation folder." % ", ".join(both))
        return 1

    RESULTS = a.results
    LATEST, BEST = os.path.join(RESULTS, "mgn_latest.pt"), os.path.join(RESULTS, "mgn_best.pt")
    BEST_VAL = os.path.join(RESULTS, "mgn_best_val.pt")
    LOG, R2LOG = os.path.join(RESULTS, "loss_log.csv"), os.path.join(RESULTS, "r2_log.csv")
    os.makedirs(RESULTS, exist_ok=True)

    y = kt_targets(cases)
    load_input = bool(a.load_input)
    mgn = ScpKtMGN(num_layers=a.layers, hidden_channels=a.hidden, embedding_dim=a.embedding,
                learning_rate=a.lr, epochs=a.epochs, global_features=["load"],
                aug_perc=aug["aug_perc"], aug_seed=aug["aug_seed"], anchors=True, redraw=True,
                load_input=load_input)
    reg = dict(run="B3", per_copy=not a.no_per_copy, jitter=a.jitter, ema=a.ema, weight_decay=a.weight_decay,
               plateau=not a.no_plateau, lr_patience=a.lr_patience, lr_min=a.lr_min,
               stop_patience=a.stop_patience, min_delta=a.min_delta)
    config = mgn.config()
    config["regularization"] = reg
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print("MGN TRAINING - RUN B3 (SCP FEATURES + Kt TARGET + REGULARISATION) WITH VALIDATION")
    print("=" * 78)
    print("training       %d graphs  (%d shapes x %d loads)  %s"
          % (len(cases), len(geoms), len(cases) // max(len(geoms), 1), a.data))
    print("validation     %d graphs  (%d shapes: %s)" % (len(val_cases), len(val_geoms),
                                                         ", ".join(short(g) for g in val_geoms)))
    print("device         %s%s" % (dev, "  (" + torch.cuda.get_device_name(0) + ")" if dev == "cuda" else ""))
    print("model          %d layers, hidden %d, Adam%s lr %g, batch %d, SCP features in every layer, Kt target"
          % (a.layers, a.hidden, "W (decay %g)" % a.weight_decay if a.weight_decay > 0 else "", a.lr, a.batch_size))
    print("REGULARISATION")
    print("  1 random edges   %s" % ("a fresh draw for EVERY graph every epoch (%d draws / epoch)" % len(cases)
                                    if reg["per_copy"] else "one draw per shape per epoch (as B2)"))
    print("  2 mesh jitter    %s" % ("interior nodes moved <= %.0f%% of their shortest edge, target moved along"
                                    % (100 * a.jitter) if a.jitter > 0 else "off"))
    print("  3 weight average %s" % ("EMA decay %g (~%d steps = ~%.1f epochs); averaged weights are scored and saved"
                                    % (a.ema, round(1 / (1 - a.ema)), 1 / (1 - a.ema) / len(cases))
                                    if a.ema > 0 else "off"))
    print("  4 plateau        %s" % ("lr x 0.5 after %d epochs without a better validation R2 (min %g), "
                                    "stop after %d" % (a.lr_patience, a.lr_min, a.stop_patience)
                                    if reg["plateau"] else "off (constant lr, all epochs)"))
    print("  5 load input     %s" % ("on (as B2)" if load_input else "OFF - Kt does not depend on the load"))
    print("results        %s   (at most %d epochs, R2 every %d)" % (RESULTS, a.epochs, a.eval_every))
    print("=" * 78)

    t_prep = time.time()
    mgn._train_fem = cases
    mgn._y_train = torch.tensor(y, dtype=torch.float).squeeze()
    fem_data = mgn._preprocess_fems(cases)
    mgn._compute_normalization_stats(fem_data)
    mgn._build_model()
    scp_summary(mgn, cases, os.path.join(RESULTS, "scp_summary.csv"))
    scp_summary(mgn, val_cases, None, title="SCP FEATURES - VALIDATION SHAPES")
    print("\nglobal inputs  %s" % ", ".join(["load (zeroed)" if not load_input else "load"] + SCP_NAMES))
    refresher = GraphRefresher(mgn, cases, fem_data, per_copy=reg["per_copy"], jitter=a.jitter)
    print("parameters     %d" % sum(p.numel() for p in mgn._model.parameters()))
    print("preprocessing  %.0f s" % (time.time() - t_prep))

    # scoring sets: with the load input off every load of a shape gives the same Kt -> score one per shape
    tr_eval = cases if load_input else one_per_shape(cases)
    va_eval = val_cases if load_input else one_per_shape(val_cases)

    optimizer = torch.optim.AdamW(adamw_groups(mgn._model, a.weight_decay), lr=a.lr)
    ema = WeightEMA(mgn._model, a.ema) if a.ema > 0 else None
    if ema:
        ema.attach(optimizer, mgn._model)
    loss_fn = torch.nn.MSELoss()
    st = dict(best=float("inf"), best_val=-float("inf"), best_val_epoch=0, ref=-float("inf"), ref_epoch=0,
              lr=a.lr, lr_mark=0)
    start_epoch = 0

    if a.resume and os.path.isfile(LATEST):
        ck = torch.load(LATEST, map_location=mgn.device, weights_only=False)
        old = ck.get("config", {})
        if ck.get("augmentation") != aug or old.get("target") != "kt" or old.get("regularization") != reg \
                or old.get("load_input", True) != load_input \
                or (old.get("anchor_params") or {}).get("version") != config["anchor_params"]["version"]:
            print("\nREFUSING TO RESUME: %s was trained with different settings (or SCP version %s, now %s):\n  %s\n  now: %s"
                  % (LATEST, (old.get("anchor_params") or {}).get("version"), config["anchor_params"]["version"],
                     old.get("regularization"), reg))
            return 1
        mgn._model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ema and ck.get("ema"):
            ema.load_state(ck["ema"], mgn.device)
        start_epoch = ck["epoch"]
        st.update(ck["state"])
        for g in optimizer.param_groups:
            g["lr"] = st["lr"]
        mgn.train_losses = ck.get("train_losses", [])
        np.random.seed(a.seed + start_epoch)
        truncate_log(LOG, start_epoch)
        truncate_log(R2LOG, start_epoch)
        print("\nRESUMED from epoch %d  (best validation R2 %.4f at epoch %d, lr %g)"
              % (start_epoch, st["best_val"], st["best_val_epoch"], st["lr"]))
    elif a.resume:
        print("\n--resume given but no %s found; starting fresh." % LATEST)

    if start_epoch == 0 or not os.path.isfile(LOG):
        open(LOG, "w").write("epoch,loss,seconds\n")
    if start_epoch == 0 or not os.path.isfile(R2LOG):
        open(R2LOG, "w").write(",".join(COLUMNS) + "\n")

    def eval_weights():
        """The weights that are scored and saved: the averaged ones if EMA is on."""
        return ema.state_dict_for(mgn._model) if ema else mgn._model.state_dict()

    def state(epoch):
        return {"model": mgn._model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "ema": ema.state() if ema else None, "state": dict(st), "train_losses": mgn.train_losses,
                "augmentation": aug, "config": config}

    t0 = time.time()

    def score(epoch, loss):
        """Train + validation R2 with the averaged weights -> r2_log.csv; best -> mgn_best_val.pt;
        plateau -> lr cut; returns True when training should stop."""
        raw_val = evaluate(mgn, va_eval)["r2_mean"] if ema else float("nan")
        raw = ema.swap_in(mgn._model) if ema else None
        try:
            tr, va = evaluate(mgn, tr_eval), evaluate(mgn, va_eval)
        finally:
            if ema:
                ema.swap_out(mgn._model, raw)
        with open(R2LOG, "a") as fh:
            fh.write(",".join(["%d" % epoch, "" if loss is None else "%.8f" % loss] +
                              ["%.6f" % tr[k] for k in ("r2_mean", "r2_median", "r2_pooled")] +
                              ["%.8f" % tr["loss"], "%.4f" % tr["peak"]] +
                              ["%.6f" % va[k] for k in ("r2_mean", "r2_median", "r2_pooled")] +
                              ["%.8f" % va["loss"], "%.4f" % va["peak"], "%.1f" % (time.time() - t0),
                               "%.3g" % st["lr"], "%.6f" % raw_val]) + "\n")
        mark = ""
        if va["r2_mean"] > st["best_val"]:                                # any gain: keep these weights
            st["best_val"], st["best_val_epoch"] = va["r2_mean"], epoch
            torch.save({"model": eval_weights(), "epoch": epoch, "loss": float("nan") if loss is None else loss,
                        "val_r2": va["r2_mean"], "augmentation": aug, "config": config}, BEST_VAL)
            mark = "  <- best, saved"
        if va["r2_mean"] > st["ref"] + a.min_delta:                       # a real gain: reset the patience clocks
            st["ref"], st["ref_epoch"], st["lr_mark"] = va["r2_mean"], epoch, epoch
        print("   R2 at epoch %d:  train %.4f   validation %.4f%s   (gap %+.4f, lr %.2g)%s"
              % (epoch, tr["r2_mean"], va["r2_mean"], "  [raw %.4f]" % raw_val if ema else "",
                 tr["r2_mean"] - va["r2_mean"], st["lr"], mark))
        if not reg["plateau"] or epoch == 0:
            return False
        if epoch - st["ref_epoch"] >= a.stop_patience:
            print("   EARLY STOP: validation R2 not better by %g for %d epochs (best %.4f at epoch %d)"
                  % (a.min_delta, epoch - st["ref_epoch"], st["best_val"], st["best_val_epoch"]))
            return True
        if epoch - st["lr_mark"] >= a.lr_patience and st["lr"] > a.lr_min * 1.0001:
            st["lr"] = max(st["lr"] * 0.5, a.lr_min)
            st["lr_mark"] = epoch
            for g in optimizer.param_groups:
                g["lr"] = st["lr"]
            print("   validation R2 flat for %d epochs -> learning rate cut to %.2g" % (a.lr_patience, st["lr"]))
        return False

    if start_epoch == 0:
        print("\nscoring the untrained model (epoch 0) ...")
        score(0, None)

    print("\nepoch      loss        elapsed     eta (at most)")
    print("-" * 56)
    interrupted = stopped = False
    try:
        for epoch in range(start_epoch, a.epochs):
            refresher.refresh(epoch)                                          # new edges + jitter for every graph
            loss = mgn._train_epoch(cases, fem_data, a.batch_size, optimizer, loss_fn)   # the paper's step
            mgn.train_losses.append(loss)
            elapsed = time.time() - t0
            eta = elapsed / (epoch - start_epoch + 1) * (a.epochs - epoch - 1)
            open(LOG, "a").write("%d,%.8f,%.1f\n" % (epoch + 1, loss, elapsed))
            if loss < st["best"]:
                st["best"] = loss
                torch.save({"model": eval_weights(), "epoch": epoch + 1, "loss": loss,
                            "augmentation": aug, "config": config}, BEST)
            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print("%6d   %.6f   %s   %s" % (epoch + 1, loss, fmt(elapsed), fmt(eta)))
            if (epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs:
                stopped = score(epoch + 1, loss)
            if (epoch + 1) % a.ckpt_every == 0 or stopped:
                torch.save(state(epoch + 1), LATEST)
            if stopped:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted. Saving checkpoint...")

    torch.save(state(len(mgn.train_losses)), LATEST)
    print("\n" + "=" * 78)
    print("%s at epoch %d after %s" % ("Stopped" if interrupted else "Early-stopped" if stopped else "Finished",
                                      len(mgn.train_losses), fmt(time.time() - t0)))
    if mgn.train_losses:
        print("final loss            %.6f\nbest loss             %.6f" % (mgn.train_losses[-1], st["best"]))
    print("best validation R2    %.4f at epoch %d   -> %s" % (st["best_val"], st["best_val_epoch"], BEST_VAL))
    if interrupted:
        print("\nContinue with:  bash scripts/run_b3.sh start     (or: bash scripts/b3_cps4.sh resume)")
    else:
        ck = torch.load(BEST_VAL, map_location=mgn.device, weights_only=False)
        mgn._model.load_state_dict(ck["model"])                       # the final file = the best-validation weights
        mgn.save(os.path.join(RESULTS, "multi_geometry_mgn.pt"))
        print("multi_geometry_mgn.pt = the best-validation weights (epoch %d)" % st["best_val_epoch"])
    print("\nnext:  bash scripts/b3_cps4.sh curves     bash scripts/b3_cps4.sh compare")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
