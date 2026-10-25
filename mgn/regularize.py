"""
Regularisation for Run B2-R (B2 = Kt target) - against over-fitting on the SAME dataset.

WHY B2 OVER-FITS
    B2 learns Kt = stress / load. The plate is linear elastic, so Kt does not
    depend on the load: the 540 training graphs (27 shapes x 20 loads) carry
    only 27 different targets. After ~1000 epochs the network starts to
    memorise those 27 shapes - train R2 keeps rising, validation R2 falls
    (B2: best 0.9936 at epoch 1050, 0.973-0.982 later).

THE FOUR PARTS (each can be switched off for an ablation)
    1. FRESH RANDOM EDGES FOR EVERY GRAPH        GraphRefresher(per_copy=True)
       B1/B2 draw one set of random edges per SHAPE per epoch, shared by its 20
       load copies. Here every copy gets its own draw, so each epoch shows every
       shape with 20 different random-edge sets instead of 1. B1 showed that the
       model recognises a training shape by its random edges (R2 0.9995 -> 0.82
       when only they are re-drawn); 20x more draws make that much harder.
    2. MESH JITTER                               GraphRefresher(jitter=0.05)
       Every epoch each INTERIOR node is moved by a small random step, at most
       jitter x (its shortest mesh edge). Hole, plate-edge, load and support
       nodes never move, so the shape is exact. The target moves with the node
       (first-order Taylor): Kt(x + d) = Kt(x) + grad Kt . d, with grad Kt from
       a least-squares fit over the node's mesh neighbours. The exact edge
       vectors of each mesh - a fingerprint the network can memorise - change
       every epoch; the physics does not.
    3. WEIGHT AVERAGING (EMA)                     WeightEMA(decay=0.9995)
       A running average of the weights over the last ~2000 training steps
       (~4 epochs). The averaged weights are scored, saved and used for
       prediction. Averaging removes the step-to-step noise and settles in a
       flatter minimum, which generalises better (Polyak & Juditsky 1992).
    4. LEARNING RATE CUT ON A PLATEAU + EARLY STOP   (in train_b2r_val.py)
       When the validation R2 has not improved for 300 epochs, the learning rate
       is halved (down to 1e-6); after 1000 epochs without improvement training
       stops. The best-validation weights are kept.
    Plus, in train_b2r_val.py, the load is not an input (Kt is load-independent
    in linear elasticity, so the load can only add noise to a Kt prediction).

Pure numpy / torch. The model, the anchors and the paper's training step
(_train_epoch) are not changed: this module only rewrites, before every
epoch, the edge_index, edge_attr and y of the training graphs (fem_data).
"""

import math

import numpy as np
import torch

from .graph import NODE_TYPE_TO_ID, geometry_seed

INTERIOR = NODE_TYPE_TO_ID["interior"]


# ------------------------------------------------------------------ random edges

def mesh_codes(mesh_edge_index, n_nodes):
    """Sorted unique codes lo * n + hi of the undirected mesh edges."""
    e = np.asarray(mesh_edge_index, dtype=np.int64)
    lo, hi = np.minimum(e[0], e[1]), np.maximum(e[0], e[1])
    return np.unique(lo * n_nodes + hi)


def draw_random_edges(n_nodes, codes, n_new, rng):
    """n_new random node pairs, uniform over all pairs that are not a self-loop,
    not a mesh edge and not drawn twice (the EA-GNN rule of mgn.graph.augment_edges,
    vectorised: ~1 ms instead of ~15 ms per graph). Returns (2, 2 n_new) int64:
    [pairs forward | pairs reversed]."""
    out = np.empty(0, dtype=np.int64)
    while len(out) < n_new:
        m = int(1.3 * (n_new - len(out))) + 32
        a = rng.integers(0, n_nodes, m)
        b = rng.integers(0, n_nodes, m)
        keep = a != b
        lo, hi = np.minimum(a, b)[keep], np.maximum(a, b)[keep]
        c = lo * n_nodes + hi
        pos = np.minimum(np.searchsorted(codes, c), len(codes) - 1)
        c = c[codes[pos] != c]                                   # not a mesh edge
        out = np.concatenate([out, c])
        _, first = np.unique(out, return_index=True)             # not drawn twice (keep draw order)
        out = out[np.sort(first)]
    out = out[:n_new]
    lo, hi = out // n_nodes, out % n_nodes
    return np.concatenate([np.stack([lo, hi]), np.stack([hi, lo])], axis=1)


# ------------------------------------------------------------------ target gradient

def node_gradients(coords, mesh_edge_index, values):
    """Least-squares gradient of a nodal field at every node from its mesh neighbours:
    minimise sum_j (g . (x_j - x_i) - (v_j - v_i))^2  ->  (N, 2)."""
    x = np.asarray(coords, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64).ravel()
    src, dst = np.asarray(mesh_edge_index, dtype=np.int64)
    d = x[dst] - x[src]
    dv = v[dst] - v[src]
    n = len(x)
    A = np.zeros((n, 2, 2))
    b = np.zeros((n, 2))
    np.add.at(A, src, d[:, :, None] * d[:, None, :])
    np.add.at(b, src, d * dv[:, None])
    tr = A[:, 0, 0] + A[:, 1, 1]
    A[:, 0, 0] += 1e-9 * tr + 1e-30
    A[:, 1, 1] += 1e-9 * tr + 1e-30
    return np.linalg.solve(A, b[:, :, None])[:, :, 0]


# ------------------------------------------------------------------ graph refresher

class GraphRefresher:
    """Before every epoch: new random edges for every training graph and (optionally)
    jittered interior nodes with the target moved along. Writes fem_data[i]['edge_index'],
    ['edge_attr'] (normalised exactly like the model's own) and ['y'].

    Reproducible: graph i in epoch e always gets the same draw (seed from the shape
    name, aug_seed, e and the graph's copy number), so --resume continues identically."""

    def __init__(self, mgn, cases, fem_data, *, per_copy=True, jitter=0.05):
        if not mgn.augmented:
            raise ValueError("GraphRefresher needs an edge-augmented model (aug_perc > 0)")
        self.mgn, self.per_copy, self.jitter = mgn, bool(per_copy), float(jitter)
        self.items = []
        copies, shared = {}, {}
        for case, fd in zip(cases, fem_data):
            key = (case.geometry, len(case.coordinates))
            k = copies.get(key, 0)
            copies[key] = k + 1
            if key not in shared:
                x = np.asarray(case.coordinates, dtype=np.float64)
                mesh = mgn._mesh_edges(case)                            # real edges, both directions
                n = len(x)
                codes = mesh_codes(mesh, n)
                src, dst = mesh
                ln = np.linalg.norm(x[dst] - x[src], axis=1)
                hmin = np.full(n, np.inf)
                np.minimum.at(hmin, src, ln)
                movable = np.asarray(case.node_types).astype(np.int64) == INTERIOR
                shared[key] = dict(x=x, mesh=mesh, codes=codes, n_new=int(round(mgn.aug_perc * len(codes))),
                                   step=np.where(movable & np.isfinite(hmin), hmin, 0.0))
            sh = shared[key]
            y0 = fd["y"].detach().clone().float()
            grad = (node_gradients(sh["x"], sh["mesh"], y0.numpy()) if self.jitter > 0 else None)
            self.items.append(dict(fd=fd, key=key, name=case.geometry, copy=k, y0=y0, grad=grad, **sh))
        self.n_shapes = len(shared)

    def _seed(self, it, epoch):
        k = it["copy"] if self.per_copy else 0
        return [geometry_seed(it["name"], self.mgn.aug_seed), int(epoch), int(k)]

    def refresh(self, epoch):
        mgn = self.mgn
        for it in self.items:
            rng = np.random.default_rng(self._seed(it, epoch))
            aug = draw_random_edges(len(it["x"]), it["codes"], it["n_new"], rng)
            ei = np.concatenate([it["mesh"], aug], axis=1)
            flag = np.concatenate([np.zeros(it["mesh"].shape[1]), np.ones(aug.shape[1])])
            x, y = it["x"], it["y0"]
            if self.jitter > 0:
                jr = np.random.default_rng(self._seed(it, epoch) + [7])
                ang = jr.uniform(0.0, 2.0 * math.pi, len(x))
                rad = self.jitter * it["step"] * np.sqrt(jr.uniform(0.0, 1.0, len(x)))     # uniform in a disc
                dlt = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
                x = x + dlt
                y = y + torch.from_numpy((it["grad"] * dlt).sum(axis=1)).float()
            d = x[ei[1]] - x[ei[0]]
            raw = np.concatenate([d, np.linalg.norm(d, axis=1, keepdims=True), flag[:, None]], axis=1)
            it["fd"]["edge_index"] = torch.from_numpy(ei)
            it["fd"]["edge_attr"] = mgn._normalize_edges(torch.from_numpy(raw).float())
            it["fd"]["y"] = y


# ------------------------------------------------------------------ weight averaging

class WeightEMA:
    """Exponential moving average of the model weights, updated after every optimiser step:
        w_avg <- d w_avg + (1 - d) w,   d = min(decay, (1 + n) / (10 + n))
    (the warm-up term keeps the first steps from being dominated by the random start)."""

    def __init__(self, model, decay=0.9995):
        self.decay = float(decay)
        self.n = 0
        self.shadow = dict((k, v.detach().clone().float()) for k, v in model.state_dict().items())

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def attach(self, optimizer, model):
        """Update automatically after every optimizer.step() (also the paper's _train_epoch)."""
        return optimizer.register_step_post_hook(lambda opt, args, kwargs: self.update(model))

    def state_dict_for(self, model):
        """The averaged weights in the model's own dtypes (what is saved / evaluated)."""
        ref = model.state_dict()
        return dict((k, self.shadow[k].to(ref[k].dtype)) for k in ref)

    def swap_in(self, model):
        """Put the averaged weights into the model; returns the raw weights for swap_out."""
        raw = dict((k, v.detach().clone()) for k, v in model.state_dict().items())
        model.load_state_dict(self.state_dict_for(model))
        return raw

    @staticmethod
    def swap_out(model, raw):
        model.load_state_dict(raw)

    def state(self):
        return {"decay": self.decay, "n": self.n, "shadow": self.shadow}

    def load_state(self, st, device):
        self.decay, self.n = float(st["decay"]), int(st["n"])
        self.shadow = dict((k, v.to(device).float()) for k, v in st["shadow"].items())


def adamw_groups(model, weight_decay):
    """AdamW parameter groups: decay on the weight matrices only (not biases, not the
    node-type embedding). weight_decay = 0 gives exactly Adam."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.startswith("node_embedding") else decay).append(p)
    return [{"params": decay, "weight_decay": float(weight_decay)}, {"params": no_decay, "weight_decay": 0.0}]
