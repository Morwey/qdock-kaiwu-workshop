# -*- coding: utf-8 -*-
"""Run the two workshop demos on the CIM: build the QUBO, solve it through an
8-bit PrecisionReducer, and pool five runs (the CIM is sampling-limited, so any
one run lands anywhere in the spread). Reuses the cached runs in cim_cache/.

    export KAIWU_USER_ID=... KAIWU_SDK_CODE=...
    python run_demo.py            # both demos
    python run_demo.py gpm        # 3f3d only
    python run_demo.py fam        # 3d4z only
"""
import os, sys, numpy as np
import kaiwu as kw
from qdock_kaiwu import GPMDock, FAMDock, backends, scoring, evaluate, init_license
from qdock_kaiwu.qubo import build_gpm_qubo, build_fam_qubo
from qdock_kaiwu.gpm import _matches_to_poses
from qdock_kaiwu.params import GPM as GPM_P, FAM as FAM_P

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
init_license()
kw.common.CheckpointManager.save_dir = os.path.join(HERE, "cim_cache")


def cim_dock(Q, task_name, t=12, samples=300):
    ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)
    cim = kw.cim.CIMOptimizer(task_name=task_name, wait=True, interval=1,
                              task_mode="quota", sample_number=samples)
    reducer = kw.cim.PrecisionReducer(cim, precision=8, truncated_precision=t,
                                      only_feasible_solution=False)
    spins = np.asarray(reducer.solve(ising))
    return backends._rank_unique([backends._spins_to_binary(s, Q.shape[0]) for s in spins], Q)


def pool(name, d, lig, Q, variables, sites, tasks):
    per_run, best = [], 9.9
    for task in tasks:
        ranked = cim_dock(Q, task)
        P = np.array(_matches_to_poses(lig, np.array(variables), sites, ranked)[0])
        rmsd = evaluate.pose_rmsds(P, lig.coords, lig.elements)
        vina = scoring.score_poses(d.receptor_pdbqt, lig.lines, P, d.workdir + "/score")
        per_run.append(round(float(rmsd[int(np.nanargmin(vina))]), 2))
        best = min(best, float(rmsd.min()))
    print("\n%s — %d variables" % (name, Q.shape[0]))
    print("  per-run best RMSD:", per_run)
    print("  pooled best: %.2f A" % best)


def gpm():
    d = GPMDock(backend="cim", workdir="/tmp/run_gpm")
    d.make_receptor(f"{DATA}/3f3d_protein.mol2"); d.make_ligand([f"{DATA}/3f3d_ligand.mol2"])
    d.make_box_ligand(f"{DATA}/3f3d_ligand.mol2")
    lig = d.ligands[0]
    Q, v = build_gpm_qubo(lig.coords, lig.ad_types, d.grid_dict, d.box_coords,
                          GPM_P["edge_cutoff"], GPM_P["K_dist"], GPM_P["K_mono"])
    tasks = ["qdock_3f3d_GPM_2p0_p8t12"] + [f"repro3_3f3d_GPM_t12_r{i}" for i in range(4)]
    pool("GPM 3f3d", d, lig, np.asarray(Q, float), v, d.box_coords, tasks)


def fam():
    d = FAMDock(backend="cim", workdir="/tmp/run_fam", max_features=24)
    d.make_receptor(f"{DATA}/3d4z_protein.pdb"); d.make_ligand([f"{DATA}/3d4z_ligand.mol2"])
    d.make_box_ligand(f"{DATA}/3d4z_ligand.mol2")
    lig = d.ligands[0]
    Q, v = build_fam_qubo(lig.coords, lig.ad_types, d.feat_coords, d.feat_elements,
                          FAM_P["edge_cutoff"], FAM_P["K_dist"], FAM_P["K_mono"])
    tasks = ["qdock_3d4z_FAM_1p0_p8t12"] + [f"repro3_3d4z_FAM_t12_r{i}" for i in range(4)]
    pool("FAM 3d4z", d, lig, np.asarray(Q, float), v, d.feat_coords, tasks)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("gpm", "all"):
        gpm()
    if which in ("fam", "all"):
        fam()
