# -*- coding: utf-8 -*-
"""Generate notebooks/qdock_kaiwu_colab.ipynb — a Colab-runnable companion that
reproduces the two CIM docking results (poses, RMSD, H-bonds, 3D figure) from the
shipped poses, with only numpy + matplotlib. The Kaiwu SDK (macOS/Windows only)
is shown for reference; the local notebook runs it live."""
import json, os

NB = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}


def md(*l): NB["cells"].append({"cell_type": "markdown", "metadata": {}, "source": _s(l)})
def code(*l): NB["cells"].append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": _s(l)})
def _s(lines):
    t = "\n".join(lines).split("\n")
    return [x + "\n" for x in t[:-1]] + [t[-1]]


md("# QDock-Kaiwu on Colab",
   "",
   "Docking pose sampling encoded as a QUBO and solved on the **Kaiwu Coherent Ising",
   "Machine**. This Colab reproduces the workshop's two results — the CIM-sampled",
   "poses, their RMSD to the crystal, the hydrogen bonds, and the 3D overlay — using",
   "only NumPy + Matplotlib. The Kaiwu solve that produced the poses is shown at the",
   "end; it runs in `qdock_kaiwu_workshop.ipynb` with the Kaiwu SDK (macOS/Windows).",
   "",
   "| demo | encoding | ligand | CIM best RMSD (t8 → t12) |",
   "|---|---|---|---|",
   "| 3f3d | GPM (vdW grid) | fragment | 4.19 → 4.15 → **1.25 Å** |",
   "| 3d4z | FAM (features) | gluco-imidazole | 3.48 → **1.65** → 1.84 Å |")

code("!git clone -q https://github.com/Morwey/qdock-kaiwu-workshop.git",
     "%cd qdock-kaiwu-workshop",
     "import sys, numpy as np, matplotlib.pyplot as plt",
     "sys.path.insert(0, '.')",
     "from qdock_kaiwu import evaluate          # pure-NumPy metrics (no Kaiwu needed)",
     "D = np.load('data/demo_poses.npz')")

md("## 1. The CIM poses vs the crystal",
   "",
   "Each pose is the best of the CIM's reads at `truncated_precision=12`. RMSD is the",
   "heavy-atom distance to the crystal ligand (the redocking reference).")
code("for tag in ('gpm', 'fam'):",
     "    rmsd = evaluate.pose_rmsds([D[f'{tag}_docked']], D[f'{tag}_crystal'], D[f'{tag}_elements'])[0]",
     "    print(f'{tag.upper()}: CIM-docked RMSD = {rmsd:.2f} A')")

md("## 2. 3D overlay — CIM pose (colored sticks) vs crystal (blue dashed)")
code("EL = {'C':'#444','N':'#2c5fa8','O':'#cc3333','S':'#d4a000','P':'#d4a000'}",
     "def bonds(xyz, el):",
     "    keep = np.array([e != 'H' for e in el]); idx = np.where(keep)[0]; P = xyz[keep]",
     "    return [(idx[i], idx[j]) for i in range(len(P)) for j in range(i+1, len(P))",
     "            if np.linalg.norm(P[i]-P[j]) < 1.85]",
     "",
     "def draw(ax, crystal, docked, el, title, hb=None):",
     "    el = [str(e).strip().upper()[:1] for e in el]; keep = [e != 'H' for e in el]",
     "    for a, b in bonds(crystal, el): ax.plot(*zip(crystal[a], crystal[b]), color='#9ec3e6', lw=2, ls=(0,(4,2)))",
     "    for a, b in bonds(docked, el):  ax.plot(*zip(docked[a], docked[b]), color='#666', lw=3)",
     "    C, Dk = crystal[keep], docked[keep]; ek = [e for e,k in zip(el,keep) if k]",
     "    ax.scatter(*C.T, c='#9ec3e6', s=45, alpha=.6, label='crystal')",
     "    ax.scatter(*Dk.T, c=[EL.get(e,'#888') for e in ek], s=130, edgecolor='k', lw=.5, label='CIM pose')",
     "    pts = [C, Dk]",
     "    if hb is not None:",
     "        for a, r in hb: ax.plot(*zip(a, r), color='#2ca02c', lw=2, ls=':')",
     "        if len(hb): ax.scatter(*np.array([r for _,r in hb]).T, c='#2ca02c', s=70, marker='^'); pts.append(np.array([r for _,r in hb]))",
     "        ax.plot([], [], color='#2ca02c', ls=':', marker='^', label='H-bond → receptor')",
     "    P = np.vstack(pts); c0 = P.mean(0); rad = np.abs(P-c0).max()+1",
     "    ax.set_xlim(c0[0]-rad,c0[0]+rad); ax.set_ylim(c0[1]-rad,c0[1]+rad); ax.set_zlim(c0[2]-rad,c0[2]+rad)",
     "    ax.set_title(title); ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])",
     "",
     "hb = list(zip(D['fam_hb_lig'], D['fam_hb_rec']))",
     "fig = plt.figure(figsize=(12, 5.5))",
     "draw(fig.add_subplot(121, projection='3d'), D['gpm_crystal'], D['gpm_docked'], D['gpm_elements'],",
     "     f\"GPM 3f3d  —  {float(D['gpm_rmsd']):.2f} A\")",
     "draw(fig.add_subplot(122, projection='3d'), D['fam_crystal'], D['fam_docked'], D['fam_elements'],",
     "     f\"FAM 3d4z  —  {float(D['fam_rmsd']):.2f} A, {len(hb)} H-bonds\", hb=hb)",
     "plt.tight_layout(); plt.show()")

md("## 3. FAM reads out as hydrogen bonds",
   "",
   "FAM rewards matching each ligand atom to a pocket feature of similar",
   "electronegativity — polar to polar. The docked gluco-imidazole places its N/O",
   "atoms onto receptor polar atoms, recovering the crystal's hydrogen-bond network:")
code("for a, r in zip(D['fam_hb_lig'], D['fam_hb_rec']):",
     "    print(f'ligand polar atom → receptor polar atom : {np.linalg.norm(a-r):.2f} A')")

md("## 4. The Kaiwu solve (reference)",
   "",
   "The poses above came from this — `qubo_matrix_to_ising_matrix` → an 8-bit",
   "`PrecisionReducer` around the `CIMOptimizer` → decode. Run it in",
   "`qdock_kaiwu_workshop.ipynb` (the Kaiwu SDK is macOS/Windows; export",
   "`KAIWU_USER_ID` / `KAIWU_SDK_CODE` first).",
   "",
   "```python",
   "import kaiwu as kw",
   "kw.license.init(user_id=..., sdk_code=...)",
   "ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)        # docking QUBO → Ising",
   "cim = kw.cim.CIMOptimizer(task_name='qdock_3f3d_GPM_2p0_p8t12', wait=True,",
   "                          interval=1, task_mode='quota', sample_number=300)",
   "reducer = kw.cim.PrecisionReducer(cim, precision=8, truncated_precision=12,",
   "                                  only_feasible_solution=False) # quantize to 8 bits",
   "spins = reducer.solve(ising)                                   # submit → poll → decode",
   "```")

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "notebooks", "qdock_kaiwu_colab.ipynb")
with open(path, "w") as fh:
    json.dump(NB, fh, indent=1)
print("wrote", path, "with", len(NB["cells"]), "cells")
