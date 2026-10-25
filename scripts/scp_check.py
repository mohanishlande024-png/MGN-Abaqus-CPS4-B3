r"""
How good are the SCP numbers BEFORE any training?  (the numbers for the thesis, on YOUR Abaqus data)

    python scripts/scp_check.py
    python scripts/scp_check.py --dirs data/si_cps4/val data/si_cps4/unseen_interp data/si_cps4/unseen_extrap

For every geometry (one load each - Kt does not depend on the load):
    FE peak Kt            max(von Mises) / load
    edge_Kt at the peak   the Kirsch-Inglis estimate (number 3) at the node of the FE peak, compared as
                          |edge_Kt|: von Mises has no sign, so a compressive edge (edge_Kt = -1, e.g. the
                          tip of a slender hole lying along the load) counts as magnitude 1
    Kt_here R2            how well the rough field (number 6) matches the whole FE field
Prints the table, the correlation (log scale) and the median error, and writes
results_scp_check/scp_check.csv and scp_check.png.
"""

import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mgn.dataset import load_dataset, _read_name                    # noqa: E402
from mgn.graph import build_edges                                    # noqa: E402
from mgn.scp_features import scp_features, VERSION                   # noqa: E402

D = os.path.join("data", "si_cps4")
warnings.filterwarnings("ignore", message=".*Python 2.*")


def short(name):
    for p in ("plate_si_", "plate_own_", "plate_"):
        if name.startswith(p):
            return name[len(p):]
    return name


def from_dataset(path):
    """One graph per training geometry from the .pt file: (name, coords, mesh edges, Kt)."""
    cases, _ = load_dataset(path, return_meta=True)
    seen = {}
    for c in cases:
        if c.geometry in seen:
            continue
        ei, fl = np.asarray(c.edge_index), np.asarray(c.edge_flag)
        seen[c.geometry] = (np.asarray(c.coordinates, float), ei[:, fl < 0.5],
                            np.asarray(c.von_mises, float) / float(c.metadata["load"]))
    return [(short(k),) + v for k, v in seen.items()]


def from_folder(folder):
    out, seen = [], set()
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        d = np.load(f, allow_pickle=False)
        name = short(_read_name(d["geometry"]) if "geometry" in d.files
                     else os.path.splitext(os.path.basename(f))[0].rsplit("__", 1)[0])
        if name in seen:
            continue
        seen.add(name)
        out.append((name, d["coords"].astype(float), build_edges(d["tris"].astype(np.int64)),
                    d["mises"].astype(float) / float(d["load"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-data", default=os.path.join(D, "dataset_si_cps4_aug20.pt"))
    ap.add_argument("--dirs", nargs="*", default=[os.path.join(D, s) for s in ("val", "unseen_interp", "unseen_extrap")])
    ap.add_argument("--out", default="results_scp_check")
    a = ap.parse_args()

    sets = []
    if os.path.isfile(a.train_data):
        sets.append(("train", from_dataset(a.train_data)))
    for folder in a.dirs:
        if os.path.isdir(folder):
            sets.append((os.path.basename(os.path.normpath(folder)), from_folder(folder)))
    if not sets:
        sys.exit("no data found - run from the repo folder")

    rows = []
    for label, items in sets:
        for name, xy, mesh, kt in items:
            F = scp_features(xy, mesh)
            i = int(np.argmax(kt))
            ss = float(((kt - kt.mean()) ** 2).sum())
            r2 = 1.0 - float(((kt - np.abs(F[:, 5])) ** 2).sum()) / ss if ss > 0 else float("nan")
            rows.append(dict(set=label, geometry=name, fe_peak=float(kt[i]), edge_Kt_at_peak=float(F[i, 2]),
                             sharpness_at_peak=float(F[i, 0]), facing_at_peak=float(F[i, 1]),
                             width_ratio=float(F[i, 6]), kt_here_r2=r2))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "scp_check.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("SCP features version %d - check before training (%d geometries)" % (VERSION, len(rows)))
    print("%-14s %-10s %9s %15s %9s" % ("set", "geometry", "FE peak", "edge_Kt @peak", "error"))
    for r in sorted(rows, key=lambda r: (r["set"], r["fe_peak"])):
        print("%-14s %-10s %9.2f %15.2f %+8.0f%%" % (r["set"], r["geometry"], r["fe_peak"], r["edge_Kt_at_peak"],
                                                   100 * (max(abs(r["edge_Kt_at_peak"]), 1e-3) / r["fe_peak"] - 1)))
    fe = np.array([r["fe_peak"] for r in rows])
    est = np.array([max(abs(r["edge_Kt_at_peak"]), 1e-3) for r in rows])      # von Mises is a magnitude
    rr = float(np.corrcoef(np.log(fe), np.log(est))[0, 1])
    med = float(np.median(np.abs(est / fe - 1)))
    r2s = np.array([r["kt_here_r2"] for r in rows])
    print("\n|edge_Kt| at the FE peak vs FE peak Kt:  r (log) = %.3f,  median |error| = %.0f %%   (all %d geometries)"
          % (rr, 100 * med, len(rows)))
    tr = np.array([r["set"] == "train" for r in rows])
    if 2 < tr.sum() < len(rows):
        print("                                      r (log) = %.3f,  median |error| = %.0f %%   (the %d training shapes)"
              % (float(np.corrcoef(np.log(fe[tr]), np.log(est[tr]))[0, 1]), 100 * float(np.median(np.abs(est[tr] / fe[tr] - 1))),
                 int(tr.sum())))
    print("Kt_here field vs FE field: median R2 = %.3f" % float(np.nanmedian(r2s)))

    try:
        sys.path.insert(0, HERE)
        import plot_style
        plot_style.use_arial(quiet=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 6))
        cols = {"train": "#2a78d6", "val": "#9a9a94", "unseen_interp": "#eb6834", "unseen_extrap": "#1baf7a"}
        for label in dict.fromkeys(r["set"] for r in rows):
            m = np.array([r["set"] == label for r in rows])
            ax.scatter(fe[m], est[m], s=34, color=cols.get(label, "#e87ba4"), edgecolor="white", lw=0.6, label=label, zorder=3)
        lo, hi = 0.8 * min(fe.min(), est.min()), 1.25 * max(fe.max(), est.max())
        ax.plot([lo, hi], [lo, hi], color="#1f1f1e", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("FE peak K_t"); ax.set_ylabel("|edge_Kt| at the FE peak (no training)")
        ax.set_title("SCP estimate vs FE, %d geometries\nr = %.2f (log), median error %.0f %%" % (len(rows), rr, 100 * med), loc="left")
        ax.legend(fontsize=9); ax.grid(color="#ecebe6", which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "scp_check.png"), dpi=plot_style.DPI, facecolor="white")
        print("wrote %s/scp_check.csv and scp_check.png" % a.out)
    except Exception as exc:                     # the numbers above are the important part
        print("(figure skipped: %s)" % exc)


if __name__ == "__main__":
    main()
