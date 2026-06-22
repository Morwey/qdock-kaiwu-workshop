# -*- coding: utf-8 -*-
"""Redock 1y6r and tabulate GPM/FAM sampling power vs the crystal pose, with
Vina scores, at the default and finer grid resolutions.

    QDOCK_BACKEND=sa    conda run -n qdock python scripts/benchmark.py   # Kaiwu SA
    QDOCK_BACKEND=cim   ...                                              # real CIM

Needs the chemistry CLIs on PATH (PATH=~/miniforge3/envs/chem/bin:$PATH) and a
Kaiwu license (KAIWU_USER_ID / KAIWU_SDK_CODE).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qdock_kaiwu import GPMDock, FAMDock, scoring, evaluate, params, init_license

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BACKEND = os.environ.get("QDOCK_BACKEND", "sa")
CASE = "1y6r"

if BACKEND in ("sa", "cim"):
    init_license()


def gpm(grid_length):
    wd = "/tmp/bench_gpm_%s" % grid_length
    os.system("rm -rf " + wd)
    g = GPMDock(backend=BACKEND, workdir=wd)
    g.make_receptor("%s/%s_protein.pdb" % (DATA, CASE))
    g.make_ligand(["%s/%s_ligand.mol2" % (DATA, CASE)])
    L = g.ligands[0]
    g.make_box_ligand("%s/%s_ligand.mol2" % (DATA, CASE), grid_length=grid_length)
    P = g.dock(L)
    r = evaluate.pose_rmsds(P, L.coords, L.elements)
    best = int(r.argmin())
    vb = scoring.score_pose(g.receptor_pdbqt, L.lines, P[best], wd + "/s", "b")
    vc = scoring.score_pose(g.receptor_pdbqt, L.lines, L.coords, wd + "/s", "c")
    return dict(method="GPM", grid_length=grid_length, n_vars=int(g.last_qubo.shape[0]),
                n_poses=len(P), mRMSD=float(r.min()),
                disc_error=evaluate.discretization_error(L.coords, g.box_coords, L.elements),
                success_2A=bool(r.min() < 2.0), success_1p5A=bool(r.min() < 1.5),
                vina_best=_f(vb), vina_crystal=_f(vc))


def fam(grid_length, max_features):
    wd = "/tmp/bench_fam_%s" % grid_length
    os.system("rm -rf " + wd)
    f = FAMDock(backend=BACKEND, workdir=wd, max_features=max_features)
    f.make_receptor("%s/%s_protein.pdb" % (DATA, CASE))
    f.make_ligand(["%s/%s_ligand.mol2" % (DATA, CASE)])
    L = f.ligands[0]
    f.make_box_ligand("%s/%s_ligand.mol2" % (DATA, CASE), grid_length=grid_length)
    P = f.dock(L)
    r = evaluate.pose_rmsds(P, L.coords, L.elements)
    best = int(r.argmin())
    vb = scoring.score_pose(f.receptor_pdbqt, L.lines, P[best], wd + "/s", "b")
    return dict(method="FAM", grid_length=grid_length, n_features=len(f.feat_coords),
                n_vars=int(f.last_qubo.shape[0]), n_poses=len(P), mRMSD=float(r.min()),
                disc_error=evaluate.discretization_error(L.coords, f.feat_coords, L.elements),
                success_2A=bool(r.min() < 2.0), success_1p5A=bool(r.min() < 1.5),
                vina_best=_f(vb))


def _f(x):
    return None if x is None else float(x)


def fam_resolution():
    """Geometric resolution study: how the FAM discretization floor scales when
    the feature spacing is refined 1.0 -> 0.5 A (no docking; solver-independent)."""
    from qdock_kaiwu import FAMDock
    f = FAMDock(backend=BACKEND, workdir="/tmp/fam_res")
    f.make_receptor("%s/%s_protein.pdb" % (DATA, CASE))
    f.make_ligand(["%s/%s_ligand.mol2" % (DATA, CASE)])
    L = f.ligands[0]
    f.make_box_ligand("%s/%s_ligand.mol2" % (DATA, CASE))
    rows = []
    for gl in (params.FAM["grid_length"], params.FAM["grid_length_fine"]):
        pts = f.feature_coords_at(gl)
        rows.append(dict(grid_length=gl, n_feature_points=len(pts),
                         disc_error=evaluate.discretization_error(L.coords, pts, L.elements)))
    return rows


def main():
    t0 = time.time()
    rows = []
    print("GPM 2.0", flush=True); rows.append(gpm(params.GPM["grid_length"]))
    print("GPM 1.5", flush=True); rows.append(gpm(params.GPM["grid_length_fine"]))
    print("FAM 1.0", flush=True); rows.append(fam(params.FAM["grid_length"], 60))
    print("FAM resolution (geometric)", flush=True)
    fam_res = fam_resolution()
    out = dict(backend=BACKEND, case=CASE, wall_seconds=round(time.time() - t0, 1),
               rows=rows, fam_resolution=fam_res)
    with open("/tmp/bench.json", "w") as fh:
        json.dump(out, fh, indent=2)
    for r in rows + fam_res:
        print(r, flush=True)
    print("wrote /tmp/bench.json in", out["wall_seconds"], "s")


if __name__ == "__main__":
    main()
