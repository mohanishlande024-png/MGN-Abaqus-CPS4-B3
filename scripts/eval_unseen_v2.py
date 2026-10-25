r"""
Unseen LOADS and unseen GEOMETRIES - numbers, high-resolution contour plots and
scatter (parity) plots, all in Arial. Works for Run B1 AND Run B2 (Kt target):
the kind of model is read from the checkpoint.

    bash scripts/b2_cps4.sh eval        B2, the easy way
    bash scripts/b2_cps4.sh eval-b1     the same plots for your finished B1 run

    python scripts/eval_unseen_v2.py --ckpt results_si_cps4_b2/mgn_best_val.pt --draws 8
    python scripts/eval_unseen_v2.py --ckpt results_si_cps4_b1/mgn_best_val.pt --draws 8
    options:
      --dpi 600                       sharper PNGs (default 300)
      --bands 30 / --lines / --smooth contour style (default: 20 colour bands, as in B1)
      --formats png,pdf               also vector PDFs (fonts embedded, contours rasterised at --dpi)
      --contour-loads all             contours at every load, not only 8 / 60 / 130
      --contour-sets interp,extrap    contours only for the unseen geometries (skip the 81 unseen-load files)
      --no-contours                   numbers and scatter plots only (fast)
      --scatter-loads 60              scatter plots only at these loads (default: the test loads)
      --overlay-log                   log axes for the all-geometries-in-one scatter plot

R2 - SAME DEFINITION FOR B1 AND B2
    R2 = 1 - sum (FE - MGN)^2 / sum (FE - mean FE)^2, per file (one geometry at one
    load), on the von Mises stress. For one file R2 on stress = R2 on Kt (load^2
    cancels), so B1 and B2 are scored identically. What B2 changes is the model,
    not the score. Next to R2 every file now also gets
        NRMSE = RMSE / load  [%]   - the typical error as a fraction of the applied
                                      traction, which stays meaningful when the FE
                                      field is nearly uniform (thin slots, flat ellipses)
                                      and R2 exaggerates.

WRITES  (into <checkpoint folder>/unseen_eval_v2[_drawsK]/)
    metrics_all.csv            every file: R2, RMSE, MAE, max error, peak truth / prediction,
                               peak error %, NRMSE % (new, last column)
    summary_unseen_loads.png, summary_r2_vs_load.png, summary_unseen_geometries_heatmap.png
    contours/<set>/<geometry>__<load>.png         the B1 layout: FE / MGN / error in rows,
                                                  whole plate + zoom column + colour bar
    contours/<set>/<geometry>__<load>_zoom.png    a closer zoom on the hole: FE | MGN | error side by side
        Colour BANDS with round-number edges (~20 bands, as in the B1 plots); FE and MGN share
        one colour scale; peak marked with a circle; hole and plate outline drawn from the mesh.
        --bands 30 more bands, --lines thin iso-lines on the band edges, --smooth 100 levels.
    scatter/<set>/<geometry>__<load>.png          SEPARATE: one parity plot per file
    scatter_grid_<group>_<load>.png               TOGETHER: one panel per geometry
    scatter_overlay_<group>_<load>.png            TOGETHER: every geometry on one axes,
                                                  coloured by family, pooled R2
        <group> = unseen_geometries (interpolation + extrapolation) or unseen_loads

CHECKPOINTS
    mgn_best_val.pt / mgn_best.pt hold weights only: the normalisation is rebuilt from
    --train-data exactly as in training (Kt for B2, stress for B1).
    multi_geometry_mgn.pt carries its own normalisation and is loaded directly.
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")                     # no display over SSH
import matplotlib.pyplot as plt           # noqa: E402
import matplotlib.tri as mtri             # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import plot_style                                                         # noqa: E402
from mgn.dataset import Case, _read_name, build_topology, load_dataset   # noqa: E402
from mgn.augmented_trainer import AugMGN                                  # noqa: E402
from mgn.anchor_trainer import AnchorMGN                                  # noqa: E402
from mgn.kt_trainer import KtMGN, kt_targets                              # noqa: E402
from mgn.scp_trainer import ScpKtMGN, check_version as check_scp_version  # noqa: E402

FONT = plot_style.use_arial()

HOLE_ID = 3                                # NODE_TYPE_TO_ID["hole"]
R2_FLOOR = -1.0                            # R2 axes stop here; worse values are marked, not plotted to scale

# colours: validated categorical slots; text stays in ink
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de", "#2a78d6"
SET_COLOUR = {"unseen_loads": BLUE, "unseen_interp": ORANGE, "unseen_extrap": AQUA}
SET_TITLE = {"unseen_loads": "unseen LOAD (training geometry)",
             "unseen_interp": "unseen GEOMETRY - interpolation",
             "unseen_extrap": "unseen GEOMETRY - extrapolation"}
FAMILIES = [("E_H", "horizontal ellipse", BLUE), ("E_V", "vertical ellipse", ORANGE),
            ("R_W", "rectangle, width shrinks", AQUA), ("R_H", "rectangle, height shrinks", YELLOW),
            ("P", "polygon", MAGENTA)]
plt.rcParams.update({"text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11, "axes.titlecolor": INK})


# ----------------------------------------------------------------- metrics

def r2_score(t, p):
    ss = float(((t - t.mean()) ** 2).sum())
    return 1.0 - float(((t - p) ** 2).sum()) / ss if ss > 0 else float("nan")


def metrics(t, p, load=None):
    """Per-file scores. load (optional) adds NRMSE = RMSE / load in %."""
    e = p - t
    ip, it = int(np.argmax(p)), int(np.argmax(t))
    rmse = float(np.sqrt((e ** 2).mean()))
    return dict(r2=r2_score(t, p), rmse=rmse, mae=float(np.abs(e).mean()),
                max_abs_err=float(np.abs(e).max()), peak_true=float(t[it]), peak_pred=float(p[ip]),
                peak_err_pct=100.0 * (float(p[ip]) - float(t[it])) / float(t[it]),
                nrmse_load_pct=100.0 * rmse / float(load) if load else float("nan"),
                i_peak_true=it, i_peak_pred=ip)


# ----------------------------------------------------------------- model

def snapshot(path):
    """Copy the checkpoint (training may be rewriting it right now) and load the copy."""
    tmp = os.path.join(tempfile.gettempdir(), "mgn_eval_snapshot_%d.pt" % os.getpid())
    last = None
    for _ in range(6):
        try:
            shutil.copy2(path, tmp)
            return tmp, torch.load(tmp, map_location="cpu", weights_only=False)
        except Exception as exc:          # caught mid-write: wait and copy again
            last = exc
            time.sleep(3)
    raise RuntimeError("could not read %s: %s" % (path, last))


def model_kind(ck):
    """'b3' (SCP features, Kt target), 'b2' (Kt target), 'b1' (anchors) or 'aug' (plain edge-augmented)."""
    if isinstance(ck, dict) and "model_state_dict" in ck:
        return "b3" if "b3" in ck else "b2" if "b2" in ck else "b1" if "b1" in ck else "aug"
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    if cfg.get("features") == "scp":
        return "b3"
    if cfg.get("target") == "kt":
        return "b2"
    return "b1" if "anchors" in cfg else "aug"


KIND_TEXT = {"b3": "Run B3 (SCP features, Kt target)", "b2": "Run B2 (Kt target)", "b1": "Run B1",
             "aug": "edge-augmented MGN"}


def load_model(a):
    """(model, description, training loads). a needs: ckpt, train_data, layers, hidden, embedding."""
    snap, ck = snapshot(a.ckpt)
    kind = model_kind(ck)
    train_loads = []
    if os.path.isfile(a.train_data):
        cases, aug = load_dataset(a.train_data, return_meta=True)
        train_loads = sorted(set(float(c.metadata["load"]) for c in cases))
    else:
        cases, aug = None, None

    if isinstance(ck, dict) and "model_state_dict" in ck:          # multi_geometry_mgn.pt
        try:
            mgn = {"b3": ScpKtMGN, "b2": KtMGN, "b1": AnchorMGN, "aug": AugMGN}[kind].load(snap)
        except ValueError as exc:
            sys.exit(str(exc).replace(snap, a.ckpt))
        desc = "final model (%d epochs), %s" % (len(ck.get("train_losses", [])), KIND_TEXT[kind])
        return mgn, desc, train_loads

    if cases is None:
        sys.exit("%s holds weights only, so the training dataset is needed to rebuild the "
                 "normalisation - pass --train-data." % a.ckpt)
    if isinstance(ck, dict) and "augmentation" in ck and ck["augmentation"] != aug:
        sys.exit("MISMATCH: %s was trained with augmentation %s but %s has %s."
                 % (a.ckpt, ck["augmentation"], a.train_data, aug))
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    epoch = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
    loss = ck.get("loss", float("nan")) if isinstance(ck, dict) else float("nan")
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}

    print("rebuilding the normalisation from %s (%d training cases, %s) ..."
          % (a.train_data, len(cases), "target Kt = stress / load" if kind in ("b2", "b3") else "target stress"))
    y = kt_targets(cases) if kind in ("b2", "b3") else np.concatenate([c.von_mises for c in cases])
    kw = dict(num_layers=cfg.get("layers", a.layers), hidden_channels=cfg.get("hidden", a.hidden),
              embedding_dim=cfg.get("embedding", a.embedding), learning_rate=1e-5, epochs=1,
              global_features=["load"], aug_perc=aug["aug_perc"], aug_seed=aug["aug_seed"])
    if kind == "b3":
        try:
            check_scp_version(cfg.get("anchor_params"), a.ckpt)
        except ValueError as exc:
            sys.exit(str(exc))
        mgn = ScpKtMGN(anchors=cfg["anchors"], redraw=cfg["redraw"], load_input=cfg.get("load_input", True), **kw)
    elif kind == "b2":
        mgn = KtMGN(anchors=cfg["anchors"], redraw=cfg["redraw"], load_input=cfg.get("load_input", True), **kw)
    elif kind == "b1":
        mgn = AnchorMGN(anchors=cfg["anchors"], redraw=cfg["redraw"], **kw)
    else:
        mgn = AugMGN(**kw)
    mgn._train_fem = cases
    mgn._y_train = torch.tensor(y, dtype=torch.float).squeeze()
    fem_data = mgn._preprocess_fems(cases)
    mgn._compute_normalization_stats(fem_data)
    del fem_data
    mgn._build_model()
    try:
        mgn._model.load_state_dict(state)
    except RuntimeError as exc:
        sys.exit("The weights do not fit a %d-layer / %d-hidden model. If you trained with "
                 "--layers or --hidden, pass the same values here.\n%s"
                 % (kw["num_layers"], kw["hidden_channels"], str(exc).splitlines()[0]))
    mgn._model.eval()
    mgn._train_fem = None                     # free the training cases
    extra = ""
    if kind in ("b2", "b3"):
        extra = ", load input %s" % ("on" if cfg.get("load_input", True) else "OFF")
    desc = "best weights: epoch %s, training loss %.3e, %s%s" % (epoch, loss, KIND_TEXT[kind], extra)
    return mgn, desc, train_loads


# ----------------------------------------------------------------- data

def read_manifest(path):
    info, order = {}, {}
    if path and os.path.isfile(path):
        with open(path) as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                info[r["geometry"]] = r
                order[r["geometry"]] = i
    return info, order


def short(name):
    for p in ("plate_own_", "plate_si_", "plate_"):
        if name.startswith(p):
            return name[len(p):]
    return name


def label(name, info):
    r = info.get(name)
    return "%s  (%s)" % (short(name), r["param"]) if r and r.get("param") else short(name)


def family_of(name):
    s = short(name)
    for pre, title, col in FAMILIES:
        if s.startswith(pre):
            return pre, title, col
    return "?", "other", GREEN


def read_set(folder):
    import warnings
    warnings.filterwarnings("ignore", message="Reading `.npy` or `.npz` file required additional header")
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        d = np.load(f, allow_pickle=False)
        name = (_read_name(d["geometry"]) if "geometry" in d.files
                else os.path.splitext(os.path.basename(f))[0].rsplit("__", 1)[0])
        out.append(dict(path=f, name=name, load=float(d["load"]),
                        coords=d["coords"].astype(np.float32), tris=d["tris"].astype(np.int64),
                        truth=d["mises"].astype(np.float64)))
    return out


# ----------------------------------------------------------------- helpers

def fmt(v):
    return "{:,.0f}".format(v) if abs(v) >= 100 else "{:,.2f}".format(v)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def outline(tris):
    """Boundary edges of the mesh (plate outline + hole), as (M, 2) node pairs."""
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    return u[cnt == 1]


def nice_levels(lo, hi, n=100, symmetric=False):
    """SMOOTH style: n colour levels spanning exactly the data (a hair wider, so the extreme
    nodes are filled), and round-number colour-bar ticks that lie inside that range."""
    if symmetric:
        m = max(abs(lo), abs(hi), 1e-9)
        lo, hi = -m, m
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    pad = 1e-6 * (hi - lo)
    ticks = MaxNLocator(nbins=6, symmetric=symmetric).tick_values(lo, hi)
    ticks = ticks[(ticks >= lo - pad) & (ticks <= hi + pad)]
    return np.linspace(lo - pad, hi + pad, n + 1), ticks


def band_levels(lo, hi, nbands=20, symmetric=False):
    """BANDED style (as in the B1 plots): about nbands colour bands with round-number edges,
    e.g. 0, 20, 40 ... MPa, so every band is a clearly separate colour."""
    if symmetric:
        m = max(abs(lo), abs(hi), 1e-9)
        lo, hi = -m, m
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    return MaxNLocator(nbins=nbands, symmetric=symmetric).tick_values(lo, hi), None


# ----------------------------------------------------------------- contour plots

def _panel(ax, tri, vals, levels, cmap, segs, peak_xy=None, ms=8, lines=False):
    cs = ax.tricontourf(tri, vals, levels=levels, cmap=cmap, zorder=0)
    ax.set_rasterization_zorder(0.5)           # PDF: the colour fill is an image, lines and text stay vector
    if lines:                                   # thin iso-lines on every band edge
        ax.tricontour(tri, vals, levels=levels, colors=INK, linewidths=0.25, alpha=0.45, zorder=1)
    ax.add_collection(LineCollection(segs, colors=INK2, linewidths=0.6, zorder=2))
    if peak_xy is not None:
        ax.plot(peak_xy[0], peak_xy[1], "o", mfc="none", mec="#ff2d55", ms=ms, mew=1.8, zorder=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return cs


def _colorbar(fig, cs, cax, ticks, units, **kw):
    cb = fig.colorbar(cs, cax=cax, ticks=ticks, **kw)
    cb.ax.tick_params(labelsize=8.5)
    cb.set_label(units, fontsize=9)
    if cb.solids is not None:
        cb.solids.set_rasterized(True)
    return cb


def contour_figures(item, res, info, set_key, units, stem, formats, dpi, style=None):
    """Two figures per file, both in the colour style chosen by `style`:
         <stem>.png       the B1 layout - FE / MGN / error in rows, whole plate + zoom column
         <stem>_zoom.png  a closer square zoom on the hole, FE | MGN | error side by side
       style = dict(bands=20, smooth=False, lines=False)
         banded (default, as in B1): ~`bands` colour bands with round-number edges
         smooth: 100 levels (continuous look)        lines: thin iso-lines on the band edges"""
    style = dict(dict(bands=20, smooth=False, lines=False), **(style or {}))
    c, tris = item["coords"].astype(float), item["tris"]
    t, p, m = item["truth"], res["pred"], res["m"]
    err = p - t
    tri = mtri.Triangulation(c[:, 0], c[:, 1], tris)
    b = outline(tris)
    segs = np.stack([c[b[:, 0]], c[b[:, 1]]], axis=1)
    x0, x1 = float(c[:, 0].min()), float(c[:, 0].max())
    y0, y1 = float(c[:, 1].min()), float(c[:, 1].max())
    H = y1 - y0

    lo, hi = min(float(t.min()), float(p.min()), 0.0), max(float(t.max()), float(p.max()))
    el = max(float(np.abs(err).max()), 1e-9)
    if style["smooth"]:
        lv, lt = nice_levels(lo, hi)
        elv, elt = nice_levels(-el, el, symmetric=True)
    else:
        lv, lt = band_levels(lo, hi, style["bands"])               # FE and MGN share these levels
        elv, elt = band_levels(-el, el, style["bands"], symmetric=True)
    rows = [(t, lv, lt, "viridis", "Abaqus FE (truth)", "peak %s %s" % (fmt(m["peak_true"]), units),
             c[m["i_peak_true"]]),
            (p, lv, lt, "viridis", "MGN prediction", "peak %s %s  (%+.1f %%)" % (fmt(m["peak_pred"]), units,
                                                                                 m["peak_err_pct"]),
             c[m["i_peak_pred"]]),
            (err, elv, elt, "RdBu_r", "Error = MGN − FE", "RMSE %s %s,  max |error| %s %s"
             % (fmt(m["rmse"]), units, fmt(m["max_abs_err"]), units), None)]
    head = "%s   -   %s   -   load %g %s" % (label(item["name"], info), SET_TITLE[set_key], item["load"], units)
    nums = ("R² = %.4f     RMSE %s %s  (%.2f %% of load)     MAE %s %s     %d nodes"
            % (m["r2"], fmt(m["rmse"]), units, m["nrmse_load_pct"], fmt(m["mae"]), units, len(t)))

    # hole centre and size (hole nodes come from the graph's node types)
    nt = item.get("node_types")
    hole = c[np.asarray(nt) == HOLE_ID] if nt is not None else c[:0]
    if len(hole):
        cx, cy = float(hole[:, 0].mean()), float(hole[:, 1].mean())
        hx = float(np.abs(hole[:, 0] - cx).max())
        half = float(max(hx, np.abs(hole[:, 1] - cy).max()))
    else:
        cx, cy, hx, half = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.1 * H, 0.1 * H

    # ---- 1. the B1 layout: rows FE / MGN / error, columns whole plate | zoom | colour bar
    w = max(0.6 * H, 1.5 * hx)
    fig = plt.figure(figsize=(15.5, 7.4), facecolor="white")
    gs = fig.add_gridspec(3, 3, width_ratios=[6.0, 1.45, 0.07], wspace=0.06, hspace=0.42,
                          left=0.03, right=0.94, top=0.855, bottom=0.03)
    for r, (vals, levels, ticks, cmap, title, sub, pk) in enumerate(rows):
        for col, (xa, xb) in enumerate(((x0, x1), (cx - w, cx + w))):
            ax = fig.add_subplot(gs[r, col])
            cs = _panel(ax, tri, vals, levels, cmap, segs, pk, ms=11 if col else 8, lines=style["lines"])
            ax.set_xlim(xa, xb)
            ax.set_ylim(y0 - 0.02 * H, y1 + 0.02 * H)
            if col == 0:
                ax.set_title("%s    %s" % (title, sub), loc="left", fontsize=11.5)
            else:
                ax.set_title("zoom on the hole", fontsize=9.5, color=MUTED)
        _colorbar(fig, cs, fig.add_subplot(gs[r, 2]), ticks, units)
    fig.suptitle(head + "\n" + nums, fontsize=13, y=0.975)
    plot_style.save(fig, stem + ".png", formats, dpi)
    plt.close(fig)

    # ---- 2. closer zoom: three square panels side by side, same colours
    wz = min(max(1.9 * half, 0.12 * H), 0.5 * H)
    fig = plt.figure(figsize=(15.0, 5.6), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.045], wspace=0.12, hspace=0.12,
                          left=0.02, right=0.98, top=0.83, bottom=0.08)
    for k, (vals, levels, ticks, cmap, title, sub, pk) in enumerate(rows):
        ax = fig.add_subplot(gs[0, k])
        cs = _panel(ax, tri, vals, levels, cmap, segs, pk, ms=11, lines=style["lines"])
        ax.set_xlim(cx - wz, cx + wz)
        ax.set_ylim(max(cy - wz, y0 - 0.01 * H), min(cy + wz, y1 + 0.01 * H))
        ax.set_title("%s\n%s" % (title, sub), fontsize=11)
        _colorbar(fig, cs, fig.add_subplot(gs[1, k]), ticks, units, orientation="horizontal")
    fig.suptitle(head + "   -   zoom on the hole\n" + nums, fontsize=11.5, y=0.985, va="top")
    plot_style.save(fig, stem + "_zoom.png", formats, dpi)
    plt.close(fig)


# ----------------------------------------------------------------- scatter plots

def _parity(ax, t, p, colour, s=4, alpha=0.45, band=True, log=False):
    hi = 1.05 * max(float(t.max()), float(p.max()))
    lo = 0.0
    if log:
        pos = np.concatenate([t[t > 0], p[p > 0]])
        lo = float(np.percentile(pos, 0.5)) * 0.8 if len(pos) else 1e-3
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.scatter(t, p, s=s, alpha=alpha, color=colour, edgecolors="none", rasterized=True, zorder=3)
    xx = np.array([max(lo, 1e-12), hi])
    ax.plot(xx, xx, color=INK, ls="--", lw=1.0, zorder=4)
    if band:
        ax.plot(xx, 1.1 * xx, color=MUTED, ls=":", lw=0.8, zorder=4)
        ax.plot(xx, 0.9 * xx, color=MUTED, ls=":", lw=0.8, zorder=4)
    ax.set_xlim(max(lo, 1e-12) if log else lo, hi)
    ax.set_ylim(max(lo, 1e-12) if log else lo, hi)
    ax.set_aspect("equal")
    style(ax)


def scatter_single(k, info, units, path, formats, dpi):
    t, p, m = k["truth"], k["pred"], k["m"]
    fig, ax = plt.subplots(figsize=(4.4, 4.6))
    _parity(ax, t, p, SET_COLOUR[k["set"]], s=5)
    ax.text(0.04, 0.96, "R² = %.4f\nRMSE = %s %s (%.2f %% of load)\npeak error %+.1f %%\n"
            "dashed = perfect,  dotted = ±10 %%"
            % (m["r2"], fmt(m["rmse"]), units, m["nrmse_load_pct"], m["peak_err_pct"]),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8))
    ax.set_xlabel("Abaqus FE von Mises (%s)" % units, fontsize=9.5)
    ax.set_ylabel("MGN predicted (%s)" % units, fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.set_title("%s\nload %g %s  -  %s" % (label(k["name"], info), k["load"], units, SET_TITLE[k["set"]]),
                 fontsize=9.5)
    fig.tight_layout()
    plot_style.save(fig, path, formats, dpi)
    plt.close(fig)


def scatter_grid(items, order, info, units, load, title, path, formats, dpi):
    items = sorted(items, key=lambda k: ({"unseen_interp": 0, "unseen_extrap": 1}.get(k["set"], 0),
                                         order.get(k["name"], 999), k["name"]))
    if not items:
        return
    ncol = min(6, len(items))
    nrow = int(np.ceil(len(items) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.35 * nrow + 0.7))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, items):
        _parity(ax, k["truth"], k["pred"], SET_COLOUR[k["set"]], s=3, alpha=0.4, band=False)
        ax.set_title("%s\nR² = %.4f   NRMSE %.1f %%%s"
                     % (label(k["name"], info), k["m"]["r2"], k["m"]["nrmse_load_pct"],
                        "   EXTRAP" if k["set"] == "unseen_extrap" else ""), fontsize=8.5)
        ax.tick_params(labelsize=7)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.supxlabel("Abaqus FE von Mises (%s)" % units, fontsize=11)
    fig.supylabel("MGN predicted (%s)" % units, fontsize=11)
    fig.suptitle("%s at %g %s - predicted vs actual, one dot per node (dashed = perfect)" % (title, load, units),
                 fontsize=12.5)
    fig.tight_layout(rect=(0.01, 0.01, 1, 1 - 0.55 / fig.get_figheight()))
    plot_style.save(fig, path, formats, dpi)
    plt.close(fig)


def scatter_overlay(items, info, units, load, title, path, formats, dpi, log=False):
    if not items:
        return
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    t_all = np.concatenate([k["truth"] for k in items])
    p_all = np.concatenate([k["pred"] for k in items])
    used = []
    for pre, ftitle, col in FAMILIES + [("?", "other", GREEN)]:
        ks = [k for k in items if family_of(k["name"])[0] == pre]
        if not ks:
            continue
        t = np.concatenate([k["truth"] for k in ks])
        p = np.concatenate([k["pred"] for k in ks])
        ax.scatter(t, p, s=4, alpha=0.4, color=col, edgecolors="none", rasterized=True, zorder=3)
        used.append((col, "%s  (%s)" % (ftitle, ", ".join(short(k["name"]) for k in ks))))
    _parity(ax, t_all, p_all, "none", s=0, band=True, log=log)
    for col, lab in used:
        ax.scatter([], [], s=30, color=col, label=lab)
    ax.plot([], [], color=INK, ls="--", lw=1.0, label="perfect prediction")
    ax.plot([], [], color=MUTED, ls=":", lw=0.8, label="±10 %")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor=GRID, markerscale=1.2)
    r2s = [k["m"]["r2"] for k in items]
    ax.set_title("%s at %g %s - %d files\nR² all nodes pooled = %.4f     mean R² per file = %.4f"
                 % (title, load, units, len(items), r2_score(t_all, p_all), float(np.nanmean(r2s))), fontsize=10.5)
    ax.set_xlabel("Abaqus FE von Mises (%s)" % units)
    ax.set_ylabel("MGN predicted (%s)" % units)
    fig.tight_layout()
    plot_style.save(fig, path, formats, dpi)
    plt.close(fig)


# ----------------------------------------------------------------- summaries

def bands_from(loads):
    """Split the sorted training loads into contiguous bands."""
    if not loads:
        return []
    steps = np.diff(loads)
    if len(steps) == 0:
        return [(loads[0], loads[0])]
    cut = 1.5 * float(np.median(steps))
    bands, start = [], loads[0]
    for i, s in enumerate(steps):
        if s > cut:
            bands.append((start, loads[i]))
            start = loads[i + 1]
    bands.append((start, loads[-1]))
    return bands


def role(L, bands):
    if not bands:
        return ""
    if L < bands[0][0]:
        return "below trained range"
    if L > bands[-1][1]:
        return "above trained range"
    if any(lo <= L <= hi for lo, hi in bands):
        return "inside a trained band"
    return "in the gap between bands"


def summary_unseen_loads(rows, order, info, test_loads, bands, units, path, formats, dpi):
    rs = [r for r in rows if r["set"] == "unseen_loads"]
    if not rs:
        return
    geoms = sorted(set(r["geometry"] for r in rs), key=lambda g: (order.get(g, 999), g))
    ypos = dict((g, i) for i, g in enumerate(geoms))
    cols = [BLUE, ORANGE, AQUA, MAGENTA, GREEN]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, max(4.8, 0.33 * len(geoms) + 2.6)), sharey=True,
                                     gridspec_kw=dict(wspace=0.06))
    for k, L in enumerate(test_loads):
        sel = [r for r in rs if r["load"] == L]
        off = (k - (len(test_loads) - 1) / 2.0) * 0.22
        yy = [ypos[r["geometry"]] + off for r in sel]
        lab = "%g %s  (%s)" % (L, units, role(L, bands))
        r2v = np.array([r["r2"] for r in sel])
        a1.scatter(np.maximum(r2v, R2_FLOOR), yy, s=34, color=cols[k % 5], edgecolor="white", lw=0.8, label=lab,
                   zorder=3)
        low = r2v < R2_FLOOR
        if low.any():
            a1.scatter(np.full(low.sum(), R2_FLOOR), np.array(yy)[low], s=60, color=cols[k % 5], marker="<", zorder=4)
        a2.scatter([r["peak_err_pct"] for r in sel], yy, s=34, color=cols[k % 5], edgecolor="white", lw=0.8, zorder=3)
        a3.scatter([r["nrmse_load_pct"] for r in sel], yy, s=34, color=cols[k % 5], edgecolor="white", lw=0.8,
                   zorder=3)
    a1.set_yticks(range(len(geoms)))
    a1.set_yticklabels([label(g, info) for g in geoms], fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel("R²  (1 = perfect;  '<' = below %g)" % R2_FLOOR)
    lo_r2 = min([r["r2"] for r in rs if np.isfinite(r["r2"])] + [0.9])
    a1.set_xlim(max(lo_r2, R2_FLOOR) - 0.02, 1.005)
    a2.axvline(0, color=INK2, lw=1)
    a2.set_xlabel("peak stress error  (MGN peak − FE peak) / FE peak   [%]")
    a3.set_xlabel("NRMSE = RMSE / load   [%]")
    a3.set_xlim(left=0)
    for ax in (a1, a2, a3):
        style(ax)
    a1.set_title("R² per geometry", loc="left")
    a2.set_title("peak stress error", loc="left")
    a3.set_title("typical error / applied load", loc="left")
    h, lab = a1.get_legend_handles_labels()
    H = fig.get_figheight()
    fig.suptitle("Unseen LOADS - the %d training geometries at loads the model never saw" % len(geoms),
                 fontsize=13, y=1 - 0.12 / H, va="top")
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.45 / H), ncol=len(test_loads), frameon=False,
               fontsize=10)
    W = fig.get_figwidth()
    fig.subplots_adjust(left=2.3 / W, right=1 - 0.3 / W, bottom=0.8 / H, top=1 - 1.15 / H, wspace=0.06)
    plot_style.save(fig, path, formats, dpi)
    plt.close(fig)


def summary_r2_vs_load(rows, bands, units, path, formats, dpi):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for lo, hi in bands:
        ax.axvspan(lo, hi, color=BAND, alpha=0.08, lw=0)
    if bands:
        ax.text(bands[0][0], 1.0, " trained load bands", color=INK2, fontsize=9, va="top",
                transform=ax.get_xaxis_transform())
    for key in ("unseen_interp", "unseen_extrap"):
        rs = [r for r in rows if r["set"] == key]
        if not rs:
            continue
        for g in sorted(set(r["geometry"] for r in rs)):
            gr = sorted([r for r in rs if r["geometry"] == g], key=lambda r: r["load"])
            ax.plot([r["load"] for r in gr], [r["r2"] for r in gr], color=SET_COLOUR[key], lw=0.8, alpha=0.25)
        loads = sorted(set(r["load"] for r in rs))
        mean = [np.mean([r["r2"] for r in rs if r["load"] == L]) for L in loads]
        ax.plot(loads, mean, color=SET_COLOUR[key], lw=2.2, marker="o", ms=5,
                label="%s - mean of %d geometries" % (SET_TITLE[key], len(set(r["geometry"] for r in rs))))
    rs = [r for r in rows if r["set"] == "unseen_loads"]
    if rs:
        loads = sorted(set(r["load"] for r in rs))
        mean = [np.mean([r["r2"] for r in rs if r["load"] == L]) for L in loads]
        ax.plot(loads, mean, ls="", marker="D", ms=10, color=BLUE, mec="white", mew=1.5,
                label="%s - mean of %d geometries" % (SET_TITLE["unseen_loads"], len(set(r["geometry"] for r in rs))))
    style(ax)
    ax.set_xlabel("applied traction (%s)" % units)
    ax.set_ylabel("R²")
    allr2 = [r["r2"] for r in rows if np.isfinite(r["r2"])]
    if allr2:
        ax.set_ylim(max(min(allr2) - 0.05, -1.0), 1.01)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.set_title("R² against load - thin lines are single geometries, thick lines the mean", loc="left")
    fig.tight_layout()
    plot_style.save(fig, path, formats, dpi)
    plt.close(fig)


def summary_heatmap(rows, order, info, test_loads, units, path, formats, dpi):
    rs = [r for r in rows if r["set"] in ("unseen_interp", "unseen_extrap")]
    if not rs:
        return
    geoms = sorted(set(r["geometry"] for r in rs),
                   key=lambda g: (0 if any(r["set"] == "unseen_interp" for r in rs if r["geometry"] == g) else 1,
                                  order.get(g, 999), g))
    loads = sorted(set(r["load"] for r in rs))
    M = np.full((len(geoms), len(loads)), np.nan)
    setof = {}
    for r in rs:
        M[geoms.index(r["geometry"]), loads.index(r["load"])] = r["r2"]
        setof[r["geometry"]] = r["set"]
    vmin = min(0.5, np.floor(np.nanmin(M) * 10) / 10.0) if np.isfinite(np.nanmin(M)) else 0.5
    vmin = max(vmin, 0.0)
    fig, ax = plt.subplots(figsize=(max(12, 0.62 * len(loads) + 4), 0.42 * len(geoms) + 2.6))
    im = ax.imshow(np.clip(M, vmin, 1.0), cmap="Blues", vmin=vmin, vmax=1.0, aspect="auto")
    for i in range(len(geoms)):
        for j in range(len(loads)):
            v = M[i, j]
            if np.isfinite(v):
                frac = (min(max(v, vmin), 1.0) - vmin) / (1.0 - vmin + 1e-12)
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6.8,
                        color="white" if frac > 0.55 else INK)
    ax.set_xticks(range(len(loads)))
    ax.set_xticklabels(["%g%s" % (L, "\n(test)" if L in test_loads else "") for L in loads], fontsize=7.5)
    ax.set_yticks(range(len(geoms)))
    ax.set_yticklabels(["%s   [%s]" % (label(g, info), "interp" if setof[g] == "unseen_interp" else "EXTRAP")
                        for g in geoms], fontsize=9)
    n_int = sum(1 for g in geoms if setof[g] == "unseen_interp")
    if 0 < n_int < len(geoms):
        ax.axhline(n_int - 0.5, color=INK, lw=1.5)
    ax.set_xlabel("load (%s)" % units)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("R²  (below %.1f: lightest colour, number exact)" % vmin, fontsize=8.5)
    ax.set_title("Unseen GEOMETRIES - R² at every load (above the line: interpolation, below: extrapolation)",
                 loc="left")
    fig.tight_layout()
    plot_style.save(fig, path, formats, dpi)
    plt.close(fig)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    D = os.path.join("data", "si_cps4")
    ap.add_argument("--ckpt", default=os.path.join("results_si_cps4_b2", "mgn_best_val.pt"))
    ap.add_argument("--train-data", default=os.path.join(D, "dataset_si_cps4_aug20.pt"))
    ap.add_argument("--loads-dir", default=os.path.join(D, "unseen_loads"))
    ap.add_argument("--interp-dir", default=os.path.join(D, "unseen_interp"))
    ap.add_argument("--extrap-dir", default=os.path.join(D, "unseen_extrap"))
    ap.add_argument("--manifest", default=os.path.join(D, "si_manifest.csv"))
    ap.add_argument("--units", default="MPa")
    ap.add_argument("--out", default=None, help="default: <checkpoint folder>/unseen_eval_v2[_drawsK]")
    ap.add_argument("--draws", type=int, default=1,
                    help="average the prediction over this many random-edge draws (seeds aug_seed + 0..K-1)")
    ap.add_argument("--contour-loads", nargs="+", default=["test"],
                    help="'test' (default: the unseen test loads), 'all', or numbers, e.g. 60")
    ap.add_argument("--contour-sets", default="loads,interp,extrap",
                    help="which sets get contour plots, e.g. interp,extrap")
    ap.add_argument("--no-contours", action="store_true")
    ap.add_argument("--bands", type=int, default=20,
                    help="number of colour bands in the contour plots (default 20, as in B1)")
    ap.add_argument("--smooth", action="store_true", help="smooth contours (100 levels) instead of bands")
    ap.add_argument("--lines", action="store_true", help="thin iso-lines on every band edge")
    ap.add_argument("--scatter-loads", nargs="+", default=["test"],
                    help="'test' (default), 'all', or numbers - loads that get scatter plots")
    ap.add_argument("--no-scatter-files", action="store_true", help="skip the one-plot-per-file scatter plots")
    ap.add_argument("--overlay-log", action="store_true", help="log axes for the all-in-one scatter plot")
    ap.add_argument("--dpi", type=int, default=plot_style.DPI, help="PNG resolution (default 300)")
    ap.add_argument("--formats", default="png", help="png, pdf, svg - comma separated, e.g. png,pdf")
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--embedding", type=int, default=16)
    a = ap.parse_args()
    formats = tuple(f.strip().lower() for f in a.formats.split(",") if f.strip())

    if not os.path.isfile(a.ckpt):
        sys.exit("No checkpoint at %s" % a.ckpt)
    out = a.out or os.path.join(os.path.dirname(a.ckpt) or ".",
                                "unseen_eval_v2" + ("_draws%d" % a.draws if a.draws > 1 else ""))
    os.makedirs(out, exist_ok=True)

    sets = [("unseen_loads", a.loads_dir), ("unseen_interp", a.interp_dir), ("unseen_extrap", a.extrap_dir)]
    data = {}
    for key, folder in sets:
        data[key] = read_set(folder) if os.path.isdir(folder) else []
        print("%-14s %-34s %4d files" % (key, folder, len(data[key])))
    if not any(data.values()):
        sys.exit("No test files found - check --loads-dir / --interp-dir / --extrap-dir.")

    t0 = time.time()
    mgn, desc, train_loads = load_model(a)
    bands = bands_from(train_loads)
    info, order = read_manifest(a.manifest)
    test_loads = sorted(set(d["load"] for d in data["unseen_loads"])) or \
        sorted(set(d["load"] for k in data for d in data[k] if d["load"] not in train_loads))

    def pick(spec):
        if spec == ["all"]:
            return None
        if spec == ["test"]:
            return set(test_loads)
        return set(float(x) for x in spec)
    contour_loads, scatter_loads = pick(a.contour_loads), pick(a.scatter_loads)
    contour_sets = set("unseen_" + s.strip() for s in a.contour_sets.split(",") if s.strip())

    print("=" * 78)
    print("checkpoint   %s  ->  %s" % (a.ckpt, desc))
    print("device       %s    edge augmentation %.0f%% (seed %d)%s"
          % (mgn.device, 100 * mgn.aug_perc, mgn.aug_seed,
             "   prediction = mean of %d random-edge draws" % a.draws if a.draws > 1 else ""))
    print("trained on   %s %s" % (", ".join("%g-%g" % b for b in bands), a.units))
    print("test loads   %s %s" % (", ".join("%g" % L for L in test_loads), a.units))
    print("figures      font %s, %d dpi, %s" % (FONT, a.dpi, " + ".join(formats)))
    print("output       %s" % out)
    print("=" * 78)

    rows, keep, topo = [], [], {}
    n_total = sum(len(v) for v in data.values())
    done = 0
    for key, _ in sets:
        cdir = os.path.join(out, "contours", key)
        for item in data[key]:
            preds = []
            for k in range(max(1, a.draws)):
                tk = (item["name"], len(item["coords"]), k)
                if tk not in topo:
                    ei, nt, fl = build_topology(item["coords"], item["tris"], item["name"],
                                                aug_perc=mgn.aug_perc, aug_seed=mgn.aug_seed + k)[:3]
                    topo[tk] = (ei, nt, fl)
                ei, nt, fl = topo[tk]
                item["node_types"] = np.asarray(nt)
                case = Case(geometry=item["name"], coordinates=item["coords"], edge_index=ei, node_types=nt,
                            von_mises=np.zeros(len(item["coords"]), dtype=np.float32),
                            metadata={"load": item["load"]}, edge_flag=fl)
                with torch.no_grad():
                    preds.append(np.asarray(mgn.predict(case), dtype=np.float64).ravel())
            pred = np.mean(preds, axis=0)
            m = metrics(item["truth"], pred, item["load"])
            rows.append(dict(set=key, geometry=item["name"], load=item["load"], n_nodes=len(pred),
                             **dict((k, v) for k, v in m.items() if not k.startswith("i_"))))
            if scatter_loads is None or item["load"] in scatter_loads:
                keep.append(dict(set=key, name=item["name"], load=item["load"], truth=item["truth"], pred=pred, m=m))
            if not a.no_contours and key in contour_sets and (contour_loads is None or item["load"] in contour_loads):
                os.makedirs(cdir, exist_ok=True)
                contour_figures(item, dict(pred=pred, m=m), info, key, a.units,
                                os.path.join(cdir, "%s__%g" % (short(item["name"]), item["load"])), formats, a.dpi,
                                style=dict(bands=a.bands, smooth=a.smooth, lines=a.lines))
            done += 1
            if done % 25 == 0 or done == n_total:
                print("  %4d/%d files   %.0f s" % (done, n_total, time.time() - t0))

    # ---- table -------------------------------------------------------------
    csv_path = os.path.join(out, "metrics_all.csv")
    cols = ["r2", "rmse", "mae", "max_abs_err", "peak_true", "peak_pred", "peak_err_pct", "nrmse_load_pct"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "geometry", "load", "n_nodes", "r2", "rmse_" + a.units, "mae_" + a.units,
                    "max_abs_err_" + a.units, "peak_true_" + a.units, "peak_pred_" + a.units, "peak_err_pct",
                    "nrmse_load_pct"])
        for r in sorted(rows, key=lambda r: (r["set"], order.get(r["geometry"], 999), r["geometry"], r["load"])):
            w.writerow([r["set"], r["geometry"], "%g" % r["load"], r["n_nodes"]] + ["%.6f" % r[c] for c in cols])

    print("\n%-16s %6s %9s %9s %9s  %-26s %15s %12s" % ("test", "files", "mean R2", "median", "worst", "worst case",
                                                      "mean|peak err|", "mean NRMSE"))
    print("-" * 112)
    for key, _ in sets:
        rs = [r for r in rows if r["set"] == key]
        if not rs:
            continue
        r2s = np.array([r["r2"] for r in rs])
        wr = rs[int(np.nanargmin(r2s))]
        print("%-16s %6d %9.4f %9.4f %9.4f  %-26s %14.1f%% %11.2f%%"
              % (key, len(rs), np.nanmean(r2s), np.nanmedian(r2s), wr["r2"], "%s @ %g" % (short(wr["geometry"]),
                 wr["load"]), np.mean([abs(r["peak_err_pct"]) for r in rs]),
                 np.mean([r["nrmse_load_pct"] for r in rs])))
    for key, title in (("unseen_loads", "UNSEEN LOADS - R2 per geometry"),
                       ("unseen_interp", "UNSEEN GEOMETRIES (interpolation) - R2 at the test loads"),
                       ("unseen_extrap", "UNSEEN GEOMETRIES (extrapolation) - R2 at the test loads")):
        rs = [r for r in rows if r["set"] == key and r["load"] in test_loads]
        if not rs:
            continue
        print("\n" + title)
        print("  %-30s" % "geometry" + "".join("%12s" % ("%g %s" % (L, a.units)) for L in test_loads))
        for g in sorted(set(r["geometry"] for r in rs), key=lambda g: (order.get(g, 999), g)):
            vals = dict((r["load"], r["r2"]) for r in rs if r["geometry"] == g)
            print("  %-30s" % label(g, info)[:30] + "".join("%12.4f" % vals.get(L, np.nan) for L in test_loads))

    # ---- summary figures -------------------------------------------------
    F = (formats, a.dpi)
    summary_unseen_loads(rows, order, info, test_loads, bands, a.units, os.path.join(out, "summary_unseen_loads.png"), *F)
    summary_r2_vs_load(rows, bands, a.units, os.path.join(out, "summary_r2_vs_load.png"), *F)
    summary_heatmap(rows, order, info, test_loads, a.units, os.path.join(out, "summary_unseen_geometries_heatmap.png"), *F)

    # ---- scatter plots: separately and together ------------------------------
    print("\nscatter plots ...")
    if not a.no_scatter_files:
        for k in keep:
            sdir = os.path.join(out, "scatter", k["set"])
            os.makedirs(sdir, exist_ok=True)
            scatter_single(k, info, a.units, os.path.join(sdir, "%s__%g.png" % (short(k["name"]), k["load"])), *F)
    groups = [("unseen_geometries", "Unseen GEOMETRIES", ("unseen_interp", "unseen_extrap")),
              ("unseen_loads", "Unseen LOADS (training geometries)", ("unseen_loads",))]
    for L in sorted(set(k["load"] for k in keep)):
        for gname, gtitle, members in groups:
            its = [k for k in keep if k["load"] == L and k["set"] in members]
            scatter_grid(its, order, info, a.units, L, gtitle,
                         os.path.join(out, "scatter_grid_%s_%g.png" % (gname, L)), *F)
            scatter_overlay(its, info, a.units, L, gtitle,
                            os.path.join(out, "scatter_overlay_%s_%g.png" % (gname, L)), *F, log=a.overlay_log)

    n_ct = len(glob.glob(os.path.join(out, "contours", "*", "*.png")))
    n_sc = len(glob.glob(os.path.join(out, "scatter", "*", "*.png")))
    print("\nwrote        %s" % csv_path)
    print("             %s/summary_*.png, scatter_grid_*.png, scatter_overlay_*.png" % out)
    print("             %d contour plots in %s" % (n_ct, os.path.join(out, "contours")))
    print("             %d single scatter plots in %s" % (n_sc, os.path.join(out, "scatter")))
    print("time         %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
