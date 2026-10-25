r"""
Over-fitting side by side: train and validation R2 vs epoch of several runs, from their r2_log.csv.

    python scripts/compare_runs.py results_si_cps4_b2 results_si_cps4_b2r
    bash scripts/b2r_cps4.sh compare                       (the same, B2 vs B2-R)

Writes <last run>/fit_check/compare_runs.png and prints, per run:
    best validation R2 and its epoch, the last validation R2, how far validation fell
    after its best (the over-fitting drop), and the train - validation gap at the end.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib                                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
import plot_style                                                            # noqa: E402

COLORS = ["#9a9a94", "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]


def read(folder):
    path = os.path.join(folder, "r2_log.csv")
    if not os.path.isfile(path):
        sys.exit("no %s" % path)
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])   # noqa: E731
    return dict(epoch=f("epoch"), train=f("train_r2_mean"), val=f("val_r2_mean"), peak=f("val_peak_err_pct"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="results folders, e.g. results_si_cps4_b2 results_si_cps4_b2r")
    ap.add_argument("--labels", nargs="*", default=None, help="names in the legend (default: folder names)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    plot_style.use_arial(quiet=True)
    labels = a.labels or [os.path.basename(os.path.normpath(r)).replace("results_si_cps4_", "").upper() for r in a.runs]

    print("\n%-10s %10s %8s %10s %12s %10s %12s %8s" % ("run", "best val", "epoch", "last val", "typical val",
                                                       "drop", "last gap", "epochs"))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.4))
    for run, lab, col in zip(a.runs, labels, COLORS):
        c = read(run)
        k = c["epoch"] > 0
        ib = int(np.nanargmax(c["val"]))
        drop = c["val"][ib] - c["val"][-1]
        gap = c["train"][-1] - c["val"][-1]
        vk = c["val"][k]
        typical = float(np.nanmedian(vk[len(vk) // 2:]))
        print("%-10s %10.4f %8d %10.4f %12.4f %10.4f %+12.4f %8d"
              % (lab, c["val"][ib], c["epoch"][ib], c["val"][-1], typical, drop, gap, c["epoch"][-1]))
        ax = axs[0]
        ax.plot(c["epoch"][k], c["val"][k], color=col, lw=2, label="%s validation (best %.4f @ %d)"
                % (lab, c["val"][ib], c["epoch"][ib]))
        ax.plot(c["epoch"][k], c["train"][k], color=col, lw=1.2, ls="--", label="%s train" % lab)
        ax.plot([c["epoch"][ib]], [c["val"][ib]], "o", color=col, ms=7)
        ax = axs[1]
        ax.semilogy(c["epoch"][k], np.clip(1 - c["val"][k], 1e-5, None), color=col, lw=2, label="%s validation" % lab)
        ax.semilogy(c["epoch"][k], np.clip(1 - c["train"][k], 1e-5, None), color=col, lw=1.2, ls="--",
                    label="%s train" % lab)
    print("\ntypical val = median validation R2 over the second half of the run (a single best point can be a lucky spike)")
    print("drop = best validation R2 - last validation R2  (over-fitting shows as a large drop)")
    print("gap  = train R2 - validation R2 at the last epoch")

    lo = min(np.nanmin(read(r)["val"][read(r)["epoch"] > 0]) for r in a.runs)
    axs[0].set_ylim(max(lo - 0.01, 0.9), 1.001)
    axs[0].set_title("(a) R² vs epoch: solid = validation, dashed = train", loc="left")
    axs[0].set_ylabel("R²  (mean over files)")
    axs[1].set_title("(b) 1 − R² (log): the space between solid and dashed = the gap", loc="left")
    axs[1].set_ylabel("1 − R²  (lower is better)")
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.grid(color="#ecebe6", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8.5, loc="lower right" if ax is axs[0] else "upper right")
    fig.tight_layout()
    out = a.out or os.path.join(a.runs[-1], "fit_check", "compare_runs.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=plot_style.DPI, facecolor="white")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
