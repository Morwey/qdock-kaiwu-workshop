# QDock on a Coherent Ising Machine

Reproduce the molecular-docking experiment in **Wei *et al.*, "A versatile coherent
Ising computing platform," *Light: Science & Applications* (2026) 15:74**
(`s41377-025-02178-1`): three protein–ligand complexes —
PDB **1N2J**, **1LRH**, **1JD0** — redocked by **Grid Point Matching (GPM)**. Each
docking is encoded as a **QUBO** and solved on a **real Coherent Ising Machine
(CIM)** through the **Kaiwu SDK** (开物, QBoson).

One fixed recipe for all three: **`quota` mode, 8-bit precision
(`precision=8`, `truncated_precision=8`)** — an 8-bit truncation with **no spin
expansion**, so the spin count equals the number of QUBO variables, all under the
~1000-spin budget.

| PDB | spins | grid (Å) | CIM mRMSD (Å) | paper (Å) |
|---|---|---|---|---|
| **1N2J** | 232 | 2.0 | **0.41** | 0.8 |
| **1LRH** | 304 | 2.0 | **0.74** | 1.4 |
| **1JD0** | 244 | 2.5 | 1.99 | 0.6 |

*`mRMSD` = the minimum heavy-atom RMSD to the crystal pose over pooled real-CIM runs
(quota mode, 8-bit). 1N2J and 1LRH beat the paper; 1JD0 — a zinc metalloenzyme, the
hardest of the three — reaches sub-2 Å (a docking success), but its sub-Å native pose
needs more pooled runs than shipped here. The CIM is high-variance on these dense
QUBOs, so the native pose surfaces only in a fraction of runs (≈3 of 10 for 1N2J);
the recipe pools a few runs and keeps the best.*

![CIM-docked poses (coloured sticks) vs crystal (blue)](assets/docking_demo.png)

The notebook is the guided walkthrough — **every Kaiwu call shown and explained**.
It ships the real CIM solutions (`results/`), so it reproduces the numbers above
with no license; `QDOCK_FORCE_LIVE=1` resubmits to the hardware.

## Method

**GPM encoding.** One binary variable per candidate match, `x[a,s] = 1` ⇔ *ligand
atom `a` sits at grid point `s`*. The QUBO minimised is

```
H(x) = Σ_v w_v x_v  +  K_dist Σ_{p<q} [‖d_lig − d_site‖ > c] x_p x_q  +  K_mono Σ_{p<q} [same atom] x_p x_q
```

- **`w`** — the AutoGrid van-der-Waals energy of that atom type at that grid point
  (a favourable placement is negative);
- **`K_dist`** — penalises matches that distort the rigid ligand;
- **`K_mono`** — forbids one atom from occupying two grid points.

Minimising `H` selects a consistent match set; Kabsch-superposing the matched atoms
onto their grid points gives a 3-D pose (weights `c = 2.565`, `K_dist = 2.337`,
`K_mono = 39.530`).

**Solving.** Kaiwu minimises an Ising Hamiltonian and returns ±1 spins. Convert
with `kw.conversion.qubo_matrix_to_ising_matrix(Q)` and feed the result straight to
the solver — the converter bakes in the sign, so Kaiwu's maximiser minimises the
QUBO (no manual `−`). The CIM path wraps the solver in an 8-bit `PrecisionReducer`.
The CIM is **high-variance** on these dense QUBOs — the native pose surfaces in
only a fraction of runs — so the recipe **pools a few runs** and takes the minimum
RMSD (the paper's "sampling power", `mRMSD`).

## Installation

Python **3.10** (required by the Kaiwu wheel).

```bash
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install vendor/kaiwu-1.3.1-cp310-none-any.whl
python -m ipykernel install --user --name qdock-kaiwu --display-name "qdock-kaiwu"
```

Rebuilding a QUBO from a receptor (optional) needs the `autogrid4` and `obabel`
command-line tools from conda-forge; the prebuilt QUBOs in `results/` make this
unnecessary to run the notebook or `scripts/dock.py`.

**License.** Free from [platform.qboson.com](https://platform.qboson.com). Export
your own credentials — they are read from the environment and **never written into
the repo**:

```bash
export KAIWU_USER_ID=<numeric id>
export KAIWU_SDK_CODE=<sdk code>
```

The CIM is a cloud service over TLS; a local HTTP/SOCKS proxy (e.g. a VPN in TUN
mode) breaks the connection, so the code clears the proxy variables before talking
to the hardware.

## Run

```bash
jupyter lab                       # notebooks/qdock_kaiwu_workshop.ipynb (kernel "qdock-kaiwu")
python scripts/dock.py all        # decode the shipped CIM solutions -> comparison table
python scripts/dock.py 1N2J --live  # resubmit 1N2J to the hardware (needs a license)
```

## The Kaiwu solve, end to end

```python
import os, numpy as np, kaiwu as kw
from qdock_kaiwu import backends, evaluate
from qdock_kaiwu.gpm import _matches_to_poses
from types import SimpleNamespace

kw.license.init(user_id=os.environ["KAIWU_USER_ID"], sdk_code=os.environ["KAIWU_SDK_CODE"])
d = dict(np.load("results/1N2J_cim.npz", allow_pickle=True))   # prebuilt QUBO + decode metadata
Q = d["Q"].astype(float)

ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)        # QUBO -> Ising
cim = kw.cim.CIMOptimizer(task_name="qdock_1N2J_GPM_quota_p8t8", wait=True,
                          interval=2, task_mode="quota", sample_number=300)
reducer = kw.cim.PrecisionReducer(cim, precision=8, truncated_precision=8,
                                  only_feasible_solution=False) # 8-bit, no spin expansion
spins = np.asarray(reducer.solve(ising))                       # submit -> poll -> ±1 spins

ranked = backends._rank_unique([backends._spins_to_binary(s, Q.shape[0]) for s in spins], Q)
lig = SimpleNamespace(coords=d["lig_coords"])
poses, _ = _matches_to_poses(lig, d["variables"], d["box_coords"], ranked)
print("mRMSD:", round(evaluate.pose_rmsds(np.array(poses), d["lig_coords"], d["lig_elements"]).min(), 2), "Å")
```

Swap `kw.cim.CIMOptimizer` for `kw.classical.SimulatedAnnealingOptimizer(...,
size_limit=300)` to solve on the CPU instead (that is `backends.solve_sa`).

## Repository layout

```
notebooks/qdock_kaiwu_workshop.ipynb   guided session — every Kaiwu call, shown + explained
qdock_kaiwu/                           the engine: qubo · backends · gpm · geometry · evaluate · io · …
scripts/dock.py                        CLI: dock 1N2J / 1LRH / 1JD0 (cached or --live)
data/                                  the three crystal ligands (.mol2) + prepared receptors (.pdbqt)
results/<pdb>_cim.npz                  prebuilt QUBO + decode metadata + the real CIM solutions
tests/test_core.py                     license-free correctness checks
vendor/kaiwu-1.3.1-cp310-none-any.whl  the Kaiwu SDK (Python 3.10)
```

## References

- Wei *et al.*, "A versatile coherent Ising computing platform," *Light: Sci.
  Appl.* (2026) 15:74, DOI 10.1038/s41377-025-02178-1 — its molecular-docking
  application is reproduced here.
- Zha *et al.*, "Encoding Molecular Docking for Quantum Computers," *JCTC* (2024),
  DOI 10.1021/acs.jctc.3c00943 — the GPM/FAM QUBO encodings.

Kaiwu SDK © QBoson. AutoGrid, OpenBabel, Meeko under their own licenses.
