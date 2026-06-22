# -*- coding: utf-8 -*-
"""Generate notebooks/qdock_kaiwu_workshop.ipynb."""
import json, os

NB = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "Python 3 (qdock)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4, "nbformat_minor": 5}


def md(*l): NB["cells"].append({"cell_type": "markdown", "metadata": {}, "source": _s(l)})
def code(*l): NB["cells"].append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": _s(l)})
def _s(lines):
    t = "\n".join(lines).split("\n")
    return [x + "\n" for x in t[:-1]] + [t[-1]]


md("# QDock-Kaiwu — docking pose sampling on the Kaiwu SDK",
   "",
   "Molecular docking splits into **pose sampling** (NP-hard) and **scoring**.",
   "QDock encodes pose sampling as a **QUBO**; we solve it on Kaiwu — the classical",
   "**simulated-annealing** optimizer and the real **Coherent Ising Machine** — and",
   "score the poses with **AutoDock Vina**.",
   "",
   "Outline: QUBO basics → solve a QUBO on Kaiwu **SA** and **CIM** → **GPM** and",
   "**FAM** redocking of 1y6r + Vina → grid-resolution study.")

md("## 0. Setup")
code("import os, sys, numpy as np",
     "os.environ['PATH'] = os.path.expanduser('~/miniforge3/envs/chem/bin') + os.pathsep + os.environ['PATH']",
     "sys.path.insert(0, os.path.abspath('..'))",
     "import kaiwu as kw",
     "from qdock_kaiwu import GPMDock, FAMDock, scoring, evaluate, backends, qubo, params, init_license",
     "",
     "# License: set KAIWU_USER_ID / KAIWU_SDK_CODE in your environment, then:",
     "init_license()",
     "# ...or pass them directly:  init_license(user_id='YOUR_ID', sdk_code='YOUR_CODE')",
     "DATA = os.path.abspath('../data')")

md("## 1. A QUBO, solved on Kaiwu SA and CIM",
   "",
   "Kaiwu minimizes an **Ising** Hamiltonian and returns **spin** (±1) solutions.",
   "We convert QUBO↔Ising with Kaiwu's routine and map the ancilla-gauged spins",
   "back to binary `x = (1 + s·s_anc)/2`. Here is a 4-variable QUBO.")

code("Q = np.array([[-3.,  2.,  0., -1.],",
     "              [ 0., -2.,  1.,  0.],",
     "              [ 0.,  0., -3.,  2.],",
     "              [ 0.,  0.,  0., -1.]])",
     "import itertools",
     "brute = min(backends.qubo_energy(Q, np.array(b)) for b in itertools.product([0,1], repeat=4))",
     "print('brute-force minimum :', brute)")

md("### Solve it with the **SA** optimizer (`kaiwu.classical.SimulatedAnnealingOptimizer`)")
code("ising, offset = kw.conversion.qubo_matrix_to_ising_matrix(Q)",
     "sa = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=5, alpha=0.99,",
     "         cutoff_temperature=0.01, iterations_per_t=200, size_limit=50, rand_seed=42)",
     "spins = sa.solve(ising)                         # (<=50, n+1) spins in {-1,+1}",
     "ranked = backends._rank_unique([backends._spins_to_binary(s, 4) for s in spins], Q)",
     "print('SA best energy     :', ranked[0][0], ' solution:', ranked[0][1])",
     "print('matches brute force:', abs(ranked[0][0]-brute) < 1e-9)")

md("### Solve the same QUBO on the real **CIM** (`kaiwu.cim.CIMOptimizer`)",
   "",
   "This submits the Ising matrix to the photonic machine and polls the result",
   "(~1 minute, uses quota). The CIM is suited to small QUBOs like this one.")
code("kw.common.CheckpointManager.save_dir = '/tmp/kaiwu_cim'",
     "os.makedirs('/tmp/kaiwu_cim', exist_ok=True)",
     "cim = kw.cim.CIMOptimizer(task_name='qdock_workshop_demo', wait=True,",
     "                          interval=1, task_mode='quota', sample_number=10)",
     "spins_cim = cim.solve(ising)",
     "ranked_cim = backends._rank_unique([backends._spins_to_binary(s, 4) for s in np.asarray(spins_cim)], Q)",
     "print('CIM best energy    :', ranked_cim[0][0], ' solution:', ranked_cim[0][1])",
     "print('matches brute force:', abs(ranked_cim[0][0]-brute) < 1e-9)")

md("## 2. Grid Point Matching — redock 1y6r",
   "",
   "Fill the pocket with a 2.0 Å grid; AutoGrid gives each point a van der Waals",
   "energy (the QUBO reward). One variable per (ligand atom, grid point).")
code("g = GPMDock(backend='sa', workdir='nb_gpm')      # backend='cim' for the machine",
     "g.make_receptor(f'{DATA}/1y6r_protein.pdb')",
     "g.make_ligand([f'{DATA}/1y6r_ligand.mol2'])",
     "g.make_box_ligand(f'{DATA}/1y6r_ligand.mol2')    # grid 2.0 Å (GPM default)",
     "lig = g.ligands[0]",
     "poses = g.dock(lig)",
     "r = evaluate.pose_rmsds(poses, lig.coords, lig.elements)",
     "print(f'QUBO variables : {g.last_qubo.shape[0]}   reads (poses): {len(poses)}')",
     "print(f'mRMSD          : {r.min():.2f} Å   (<2.0 Å: {r.min()<2.0},  <1.5 Å: {r.min()<1.5})')")

md("### Score the poses with AutoDock Vina")
code("best = int(r.argmin())",
     "vb = scoring.score_pose(g.receptor_pdbqt, lig.lines, poses[best],  'nb_gpm/score', 'best')",
     "vc = scoring.score_pose(g.receptor_pdbqt, lig.lines, lig.coords,    'nb_gpm/score', 'cryst')",
     "print(f'Vina, best-RMSD pose : {vb:.2f} kcal/mol')",
     "print(f'Vina, crystal pose   : {vc:.2f} kcal/mol')")

md("## 3. Feature Atom Matching — the low-qubit encoding",
   "",
   "FAM matches ligand atoms to a few pocket feature atoms, so the QUBO is far",
   "smaller. Same target, for a direct comparison.")
code("f = FAMDock(backend='sa', workdir='nb_fam')",
     "f.make_receptor(f'{DATA}/1y6r_protein.pdb')",
     "f.make_ligand([f'{DATA}/1y6r_ligand.mol2'])",
     "f.make_box_ligand(f'{DATA}/1y6r_ligand.mol2')    # 1.0 Å (FAM default)",
     "ligf = f.ligands[0]",
     "posesf = f.dock(ligf)",
     "rf = evaluate.pose_rmsds(posesf, ligf.coords, ligf.elements)",
     "print(f'feature atoms = {len(f.feat_coords)}')",
     "print(f'FAM mRMSD = {rf.min():.2f} Å  (<2.0 Å: {rf.min()<2.0},  <1.5 Å: {rf.min()<1.5})')",
     "print(f'\\nSame target:  GPM {r.min():.2f} Å @ {g.last_qubo.shape[0]} qubits  vs  '",
     "      f'FAM {rf.min():.2f} Å @ {f.last_qubo.shape[0]} qubits')")

md("## 4. Grid resolution — each encoding on its own scale",
   "",
   "A ligand atom cannot match closer than the nearest site, so grid spacing sets",
   "a floor on accuracy (the *discretization error*). Each encoding is refined on",
   "its own scale: **GPM 2.0 → 1.5 Å** and **FAM 1.0 → 0.5 Å**.")
md("**GPM** — full redock at each spacing:")
code("def gpm_res(gl):",
     "    gg = GPMDock(backend='sa', workdir=f'nb_gpm_{gl}')",
     "    gg.make_receptor(f'{DATA}/1y6r_protein.pdb'); gg.make_ligand([f'{DATA}/1y6r_ligand.mol2'])",
     "    gg.make_box_ligand(f'{DATA}/1y6r_ligand.mol2', grid_length=gl)",
     "    L = gg.ligands[0]; P = gg.dock(L, save_pose=False)",
     "    rr = evaluate.pose_rmsds(P, L.coords, L.elements)",
     "    de = evaluate.discretization_error(L.coords, gg.box_coords, L.elements)",
     "    return gg.last_qubo.shape[0], de, float(rr.min())",
     "for gl in (params.GPM['grid_length'], params.GPM['grid_length_fine']):",
     "    V, de, m = gpm_res(gl)",
     "    print(f'GPM grid {gl} Å : qubits={V:5d}  discretization_error={de:.2f} Å  mRMSD={m:.2f} Å')")
md("**FAM** — the discretization floor as the feature spacing is halved",
   "(geometric; the floor is solver-independent):")
code("for gl in (params.FAM['grid_length'], params.FAM['grid_length_fine']):",
     "    pts = f.feature_coords_at(gl)",
     "    de = evaluate.discretization_error(ligf.coords, pts, ligf.elements)",
     "    print(f'FAM feature spacing {gl} Å : {len(pts):5d} feature points  discretization_error={de:.2f} Å')",
     "print('\\nFiner grid lowers the discretization floor; realizing it in mRMSD needs',",
     "      'a solver budget (reads / iterations) or a larger machine that keeps pace',",
     "      'with the finer grid\\'s larger QUBO.')")

md("## 5. Where QDock stands",
   "",
   "Published sampling power (CASF-2016, 257 cases, success = mRMSD < 2 Å):",
   "",
   "| method | success @ 2 Å | avg. mRMSD |",
   "|---|---|---|",
   "| Glide SP | 93.4 % | 1.0 Å |",
   "| **GPM** | 87.5 % | 1.1 Å |",
   "| **FAM** | 67.3 % | 1.8 Å |",
   "",
   "**Takeaways**",
   "- Pose sampling → **QUBO** → solved on Kaiwu **SA** (CPU) or the real **CIM**.",
   "- **GPM** is accurate (≈ Glide SP); **FAM** trades accuracy for far fewer qubits.",
   "- The SA initial temperature must match the QUBO energy scale (T₀ ≈ 10 here).",
   "- Accuracy is set by grid resolution (discretization), refined per encoding.")

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "qdock_kaiwu_workshop.ipynb"), "w") as fh:
    json.dump(NB, fh, indent=1)
print("wrote notebook with", len(NB["cells"]), "cells")
