#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dock one of the paper's three GPM molecules (1N2J / 1LRH / 1JD0) on the CIM:
quota mode, 8-bit (precision=8, truncated_precision=8, no spin expansion).

    python scripts/dock.py 1N2J            # decode the shipped real-CIM solutions
    python scripts/dock.py 1N2J --live     # resubmit to the hardware (needs a license)
    python scripts/dock.py all             # all three, comparison table

Credentials for --live come from the environment (KAIWU_USER_ID / KAIWU_SDK_CODE);
they are never written to disk. CIM traffic must not go through a proxy."""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qdock_kaiwu import backends, evaluate
from qdock_kaiwu.gpm import _matches_to_poses

PAPER = {"1N2J": 0.8, "1LRH": 1.4, "1JD0": 0.6}


def solve_live(Q, pdb):
    """One real-CIM run: quota mode + 8-bit PrecisionReducer (no expansion)."""
    import kaiwu as kw
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = "*"      # CIM off the proxy
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    backends.init_license()                                    # reads env credentials
    ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)
    opt = kw.cim.CIMOptimizer(task_name=f"qdock_{pdb}_GPM_quota_p8t8", wait=True,
                              interval=2, task_mode="quota", sample_number=300)
    red = kw.cim.PrecisionReducer(opt, precision=8, truncated_precision=8,
                                  only_feasible_solution=False)
    spins = np.asarray(red.solve(ising))
    return [backends._spins_to_binary(s, Q.shape[0]) for s in spins]


def dock(pdb, live=False):
    d = dict(np.load(os.path.join(ROOT, "results", f"{pdb}_cim.npz"), allow_pickle=True))
    Q = d["Q"].astype(float)
    raw = solve_live(Q, pdb) if live else list(d["solutions"])
    ranked = backends._rank_unique(raw, Q)
    lig = SimpleNamespace(coords=d["lig_coords"])
    poses, _ = _matches_to_poses(lig, d["variables"], d["box_coords"], ranked)
    r = evaluate.pose_rmsds(np.array(poses), d["lig_coords"], d["lig_elements"])
    return int(d["n_vars"]), len(poses), float(r.min())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdb", choices=["1N2J", "1LRH", "1JD0", "all"])
    ap.add_argument("--live", action="store_true", help="resubmit to the CIM (needs a license)")
    a = ap.parse_args()
    targets = ["1N2J", "1LRH", "1JD0"] if a.pdb == "all" else [a.pdb]
    print(f"{'PDB':<6}{'spins':>7}{'poses':>7}{'mRMSD/A':>9}{'paper/A':>9}")
    for p in targets:
        nv, npose, m = dock(p, live=a.live)
        print(f"{p:<6}{nv:>7}{npose:>7}{m:>9.2f}{PAPER[p]:>9.1f}")
