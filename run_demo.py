# -*- coding: utf-8 -*-
"""Run the two workshop demos on the CIM, end to end: build the QUBO, sweep
truncated_precision 8/10/12 through a PrecisionReducer, decode, Vina-rescore, and
print the best RMSD at each precision. Reuses the cached CIM runs in cim_cache/.

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


def cim_dock(Q, task_name, t, samples=300):
    ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)
    cim = kw.cim.CIMOptimizer(task_name=task_name, wait=True, interval=1,
                              task_mode="quota", sample_number=samples)
    reducer = kw.cim.PrecisionReducer(cim, precision=8, truncated_precision=t,
                                      only_feasible_solution=False)
    spins = np.asarray(reducer.solve(ising))
    return backends._rank_unique([backends._spins_to_binary(s, Q.shape[0]) for s in spins], Q)


def sweep(name, d, lig, Q, variables, sites, task):
    print("\n=== %s precision sweep (CIM) — %d variables ===" % (name, Q.shape[0]))
    for t in (8, 10, 12):
        ranked = cim_dock(Q, task % t, t)
        poses, _ = _matches_to_poses(lig, np.array(variables), sites, ranked)
        P = np.array(poses)
        rmsd = evaluate.pose_rmsds(P, lig.coords, lig.elements)
        vina = scoring.score_poses(d.receptor_pdbqt, lig.lines, P, d.workdir + "/score")
        bi = int(np.nanargmin(vina))
        print("  t=%2d  poses=%d  mRMSD=%.2f  vina-best RMSD=%.2f A" % (t, len(P), rmsd.min(), rmsd[bi]))


def gpm():
    d = GPMDock(backend="cim", workdir="/tmp/run_gpm")
    d.make_receptor(f"{DATA}/3f3d_protein.mol2"); d.make_ligand([f"{DATA}/3f3d_ligand.mol2"])
    d.make_box_ligand(f"{DATA}/3f3d_ligand.mol2")
    lig = d.ligands[0]
    Q, v = build_gpm_qubo(lig.coords, lig.ad_types, d.grid_dict, d.box_coords,
                          GPM_P["edge_cutoff"], GPM_P["K_dist"], GPM_P["K_mono"])
    sweep("GPM 3f3d", d, lig, Q, v, d.box_coords, "qdock_3f3d_GPM_2p0_p8t%d")


def fam():
    d = FAMDock(backend="cim", workdir="/tmp/run_fam", max_features=24)
    d.make_receptor(f"{DATA}/3d4z_protein.pdb"); d.make_ligand([f"{DATA}/3d4z_ligand.mol2"])
    d.make_box_ligand(f"{DATA}/3d4z_ligand.mol2")
    lig = d.ligands[0]
    Q, v = build_fam_qubo(lig.coords, lig.ad_types, d.feat_coords, d.feat_elements,
                          FAM_P["edge_cutoff"], FAM_P["K_dist"], FAM_P["K_mono"])
    sweep("FAM 3d4z", d, lig, Q, v, d.feat_coords, "qdock_3d4z_FAM_1p0_p8t%d")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("gpm", "all"):
        gpm()
    if which in ("fam", "all"):
        fam()
