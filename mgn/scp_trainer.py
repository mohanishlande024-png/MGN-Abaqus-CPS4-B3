"""
ScpKtMGN - Run B3: the B2 model (Kt target) with the stress-concentration prior (SCP) features
of mgn/scp_features.py IN PLACE OF the 14 anchor features.

    global input per node:  [load, 14 anchor numbers]  ->  [load, 7 SCP numbers]

Everything else is B2: graph, random edges, model size, the paper's training step, Kt target,
prediction = load x Kt. The SCP numbers go through the global encoder into every one of the
20 node updates, exactly as before (only the width of the global input changes, 15 -> 8).

Checkpoints: multi_geometry_mgn.pt carries ck["b1"], ck["b2"] and ck["b3"] = {"features": "scp"};
mgn_best_val.pt carries config["features"] = "scp". scripts/eval_unseen_v2.py rebuilds the model
from either.
"""

import numpy as np
import torch

from .kt_trainer import KtMGN
from .scp_features import SCP_DIM, SCP_NAMES, SCP_PARAMS, VERSION, scp_features

FEATURES = "scp"


def check_version(params, where="checkpoint"):
    """Refuse a model trained with another version of the SCP numbers (the features would differ)."""
    v = (params or {}).get("version")
    if v != VERSION:
        raise ValueError("%s was trained with SCP features version %s, but mgn/scp_features.py is version %s. "
                         "Retrain B3 with the current code (delete the old results_si_cps4_b3 folder first)."
                         % (where, v, VERSION))


class ScpKtMGN(KtMGN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_anchors:                       # 'anchors' = "geometry features on" (here: SCP)
            self.anchor_dim = SCP_DIM

    def config(self):
        c = super().config()
        c.update(features=FEATURES, anchor_dim=self.anchor_dim,
                 anchor_params=SCP_PARAMS if self.use_anchors else None)
        return c

    def anchors_for(self, fem):
        """(N, 7) SCP features of one mesh, computed once per geometry (from the clean mesh)."""
        key = (getattr(fem, "geometry", ""), len(fem.coordinates))
        if key not in self._anchor_cache:
            self._anchor_cache[key] = scp_features(np.asarray(fem.coordinates), self._mesh_edges(fem))
        return self._anchor_cache[key]

    def save(self, filepath):
        super().save(filepath)
        ck = torch.load(filepath, map_location="cpu", weights_only=False)
        ck["anchor_names"] = SCP_NAMES if self.use_anchors else []
        ck["b3"] = dict(features=FEATURES)
        torch.save(ck, filepath)

    @classmethod
    def load(cls, filepath, device=None):
        ck = torch.load(filepath, map_location="cpu", weights_only=False)
        if "b3" not in ck:
            raise ValueError("%s is not a B3 (SCP features) checkpoint" % filepath)
        check_version(ck.get("b1", {}).get("anchor_params"), filepath)
        mgn = super(ScpKtMGN, cls).load(filepath, device)
        print("  B3: stress-concentration prior features (%d per node)" % SCP_DIM)
        return mgn
