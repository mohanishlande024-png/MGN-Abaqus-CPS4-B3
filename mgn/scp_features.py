"""
Stress-concentration prior (SCP) features - Run B3.   Version 3: seven simple numbers, Kirsch fall-off.

IDEA
    Every node is told, with textbook formulas, how strong the stress concentration near it
    should be. The network learns the correction. The load always pulls along x.

    Kirsch (1898)  circular hole: edge stress 1 - 2 cos 2theta; the extra stress above the hole falls
                   off as (q^2 + 3 q^4) / 4 of its edge value, q = a / r
    Inglis (1913)  notch: peak Kt = 1 + 2 sqrt(a / rho)
    Peterson       finite width: Kt grows with hole width / plate width
                   (Pilkey & Pilkey, Peterson's Stress Concentration Factors, 2008)

AT EVERY HOLE-EDGE NODE b   (hole edges found from the mesh: an edge on exactly one element)
    rho   local radius = 1 / curvature (straight edge: infinite; never below half a segment)
    a     half-size of the hole ACROSS the load
    nu    direction the edge faces (normal pointing into the material)

THE SEVEN NUMBERS OF NODE i   (b = nearest hole-edge node, d = distance to it)
    1 sharpness    S = 1 + 2 sqrt(a / rho)              Inglis Kt of that edge point (1 if straight)
    2 facing       c = -cos 2 nu                        +1 faces across the load, -1 along it (Kirsch)
    3 edge_Kt      K = S        if facing across        estimated Kt ON the edge:
                     = -1       if facing along           K = 1 + (S - 1) max(c, 0) + 2 min(c, 0)
                     (blended in between; for a circle K = 1 - 2 cos 2theta = Kirsch exactly)
    4 near_hole    F(a / (a + d))                       1 on the edge -> 0 far away (hole-size scale)
    5 near_tip     F(r / (r + d)),  r = min(rho, a)     the same on the local-radius scale
    6 Kt_here      1 + (K - 1) * near_tip               estimated Kt AT this node
    7 width_ratio  2a / W                               hole width / plate width
    F(q) = (q^2 + 3 q^4) / 4 is Kirsch's fall-off: exact for a circular hole (1/4 of the extra stress
    fades as (a/r)^2, 3/4 as (a/r)^4). A plate without a hole gets [1, 0, 1, 0, 0, 1, 0].

Check against FE (49 local CPS4 geometries, before any training): edge_Kt at the FE peak follows
the FE peak Kt with r = 0.95 (log), median error 21 %.

Pure numpy.
"""

import math

import numpy as np

SCP_NAMES = ["scp_sharpness", "scp_facing", "scp_edge_Kt", "scp_near_hole", "scp_near_tip",
             "scp_Kt_here", "scp_width_ratio"]
SCP_DIM = len(SCP_NAMES)
VERSION = 3                 # 1 = first B3 package; 2 = simple numbers with (L/(L+d))^2; 3 = simple, Kirsch fall-off
SCP_PARAMS = dict(version=VERSION, names=SCP_NAMES, load_axis="x", rho_floor="0.5 x mean boundary segment",
                  smoothing="3-point", decay="Kirsch (q^2 + 3q^4)/4", severity="Inglis")
NO_HOLE = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])


# --------------------------------------------------------------- boundary loops (mesh topology only)

def _boundary_edges(mesh_edge_index, n_nodes):
    """Undirected boundary edges: an edge shared by exactly one element has exactly one
    common neighbour of its two end nodes."""
    e = np.asarray(mesh_edge_index, dtype=np.int64)
    und = np.unique(np.sort(e.T, axis=1), axis=0)
    adj = [set() for _ in range(n_nodes)]
    for a, b in und:
        adj[a].add(int(b))
        adj[b].add(int(a))
    keep = np.array([len(adj[a] & adj[b]) == 1 for a, b in und], dtype=bool)
    return und[keep]


def _loops(bedges):
    nbr = {}
    for a, b in bedges:
        nbr.setdefault(int(a), []).append(int(b))
        nbr.setdefault(int(b), []).append(int(a))
    seen, loops = set(), []
    for start in sorted(nbr):
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [v for v in nbr[cur] if v != prev]
            if not nxt:
                break
            nxt = nxt[0] if (prev is not None or len(nxt) == 1) else min(nxt)
            if nxt == start or nxt in seen:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
    return loops


def hole_boundaries(coords, mesh_edge_index):
    """Hole loops ordered counter-clockwise (material on the right). The outer plate edge is
    the loop with the largest bounding box and is dropped."""
    loops = _loops(_boundary_edges(mesh_edge_index, len(coords)))
    if not loops:
        return []
    box = [float(np.prod(np.ptp(coords[l], axis=0))) for l in loops]
    outer = int(np.argmax(box))
    out = []
    for i, l in enumerate(loops):
        if i == outer:
            continue
        p = coords[l]
        area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
        out.append(l if area > 0 else l[::-1])
    return out


# --------------------------------------------------------------- the prior

def kirsch_falloff(d, L):
    """F = (q^2 + 3 q^4) / 4, q = L / (L + d): Kirsch's fall-off of the extra stress, 1 on the edge.
    For a circular hole of radius L this is exact on the line across the load
    (sigma_xx / sigma = 1 + q^2 / 2 + 3 q^4 / 2 = 1 + 2 F)."""
    q = L / (L + d)
    return 0.25 * q ** 2 + 0.75 * q ** 4


kirsch_decay = kirsch_falloff        # old name


def boundary_table(coords, mesh_edge_index):
    """Per hole-edge node: position, edge_Kt K, facing c, sharpness S, a, r = min(rho, a), beta. None if no hole."""
    x = np.asarray(coords, dtype=np.float64)
    W = float(np.ptp(x[:, 1]))                                    # plate width across the x-load
    rows = []
    for l in hole_boundaries(x, mesh_edge_index):
        p = x[l]
        prv, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
        d1, d2 = p - prv, nxt - p
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        den = (np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) * np.linalg.norm(nxt - prv, axis=1))
        kap = 2.0 * cross / np.maximum(den, 1e-30)                 # 1 / circumradius, signed
        kap = (np.roll(kap, 1) + kap + np.roll(kap, -1)) / 3.0
        seg = np.linalg.norm(d2, axis=1)
        a = 0.5 * float(np.ptp(p[:, 1]))                           # half-extent across the load
        rho = np.where(kap > 0, 1.0 / np.maximum(kap, 1e-300), np.inf)
        rho = np.maximum(rho, 0.5 * float(seg.mean()))
        S = 1.0 + 2.0 * np.sqrt(a / rho)                           # Inglis (S = 1 where rho is infinite)
        t = nxt - prv
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-30)
        n = np.stack([t[:, 1], -t[:, 0]], axis=1)                   # right of the CCW direction = material
        c = -np.cos(2.0 * np.arctan2(n[:, 1], n[:, 0]))             # Kirsch angle factor
        K = 1.0 + (S - 1.0) * np.maximum(c, 0.0) + 2.0 * np.minimum(c, 0.0)
        rows.append(dict(xy=p, K=K, c=c, S=S, a=np.full(len(l), a), L=np.minimum(rho, a),
                         beta=np.full(len(l), 2.0 * a / W if W > 0 else 0.0)))
    if not rows:
        return None
    return dict((k, np.concatenate([r[k] for r in rows])) for k in rows[0])


def scp_features(coords, mesh_edge_index, return_info=False):
    """(N, 7) float32: sharpness, facing, edge_Kt, near_hole, near_tip, Kt_here, width_ratio.
    mesh_edge_index = the REAL mesh edges only."""
    x = np.asarray(coords, dtype=np.float64)
    T = boundary_table(x, mesh_edge_index)
    n = len(x)
    if T is None:
        F = np.tile(NO_HOLE, (n, 1))
    else:
        best_d = np.full(n, np.inf)
        best_j = np.zeros(n, dtype=np.int64)
        for s in range(0, len(T["xy"]), 512):                       # nearest hole-boundary node, chunked
            dd = np.sqrt(((x[:, None, :] - T["xy"][None, s:s + 512, :]) ** 2).sum(-1))
            j = dd.argmin(axis=1)
            v = dd[np.arange(n), j]
            m = v < best_d
            best_d[m], best_j[m] = v[m], j[m] + s
        j, d = best_j, best_d
        S, c, K, a, r, beta = T["S"][j], T["c"][j], T["K"][j], T["a"][j], T["L"][j], T["beta"][j]
        near_hole = kirsch_falloff(d, a)
        near_tip = kirsch_falloff(d, r)
        F = np.stack([S, c, K, near_hole, near_tip, 1.0 + (K - 1.0) * near_tip, beta], axis=1)
    F = F.astype(np.float32)
    if not return_info:
        return F
    info = dict(n_holes=0 if T is None else len(hole_boundaries(x, mesh_edge_index)),
                max_edge_Kt=float(T["K"].max()) if T is not None else 1.0,
                a=float(T["a"].max()) if T is not None else 0.0,
                beta=float(T["beta"].max()) if T is not None else 0.0)
    return F, info
