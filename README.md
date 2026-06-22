# QDock-Kaiwu

Molecular-docking **pose sampling** encoded as a **QUBO** and solved on the
**Kaiwu SDK** (开物, Bose Quantum): its classical simulated-annealing optimizer
and its real Coherent Ising Machine. Two encodings from Zha *et al.*, *J. Chem.
Theory Comput.* 2024 — Grid Point Matching and Feature Atom Matching — with
poses scored by **AutoDock Vina**.

Worked example — 1y6r (renin–inhibitor redocking), Kaiwu classical SA, seed 42:

| Encoding | Variables (qubits) | mRMSD vs crystal | < 2.0 Å | < 1.5 Å | Vina (best pose / crystal) |
|---|---|---|---|---|---|
| **GPM** | 1128 | **0.15 Å** | ✓ | ✓ | −11.35 / −11.80 kcal·mol⁻¹ |
| **FAM** | 1620 | **1.46 Å** | ✓ | ✓ | — |

---

## 1. Pose sampling as a QUBO

Docking splits into **pose sampling** (search the binding geometry — NP-hard) and
**scoring** (rank poses — cheap). QDock casts sampling as a quadratic
unconstrained binary optimization,

```
minimize  H(x) = Σ_v w_v x_v  +  Σ_{p<q} Q_pq x_p x_q ,     x ∈ {0,1},
```

the native problem of a quantum annealer / Coherent Ising Machine. One binary
variable per candidate match:

```
x_(a,s) = 1   ⇔   ligand atom a is placed at site point s.
```

`H` has three parts:

| term | role | parameter |
|---|---|---|
| linear `w_v` | reward for placing atom *a* at site *s* | data (below) |
| distance penalty | suppress matches that distort the rigid ligand: `|d(a,b) − d(s,t)| > c` | `edge_cutoff` = c, `K_dist` |
| monogamy penalty | one atom occupies one site | `K_mono` |

Minimizing `H` selects a consistent set of matches; superposing the matched
ligand atoms onto their sites (Kabsch) turns the selection into a 3-D pose.
Sampling many low-energy solutions yields many candidate poses.

### Two encodings

- **GPM — Grid Point Matching.** Sites are points of a grid filling the pocket
  (default spacing **2.0 Å**); the reward `w` is the **van der Waals energy** at
  that grid point, from **AutoGrid**. High resolution, many qubits, accurate.
- **FAM — Feature Atom Matching.** Sites are a few **pocket feature atoms**
  (default spacing **1.0 Å**, typed C / N-donor / O-acceptor); the reward is the
  electronegativity mismatch `w = |EN(a) − EN(s)| − 0.5`. Few qubits, cheaper.

Recommended parameters (`qdock_kaiwu/params.py`; multipliers fitted on the Astex
Diversity Set in the paper):

| | grid (default → fine) | `edge_cutoff` | `K_dist` | `K_mono` |
|---|---|---|---|---|
| GPM | 2.0 → 1.5 Å | 2.162 | 0.405 | 25.090 |
| FAM | 1.0 → 0.5 Å | 1.870 | 2.261 | 11.479 |

Each encoding is refined on its own scale for the resolution study (§5): GPM
2.0→1.5, FAM 1.0→0.5 — never on a common grid, since their defaults differ by 2×.

---

## 2. Solving the QUBO on Kaiwu

Kaiwu minimizes an **Ising** Hamiltonian and returns **spin** (±1) solutions. We
convert with Kaiwu's own routine and map the ancilla-gauged spins back to binary:

```python
ising, offset = kw.conversion.qubo_matrix_to_ising_matrix(Q)
spins         = optimizer.solve(ising)          # ±1, shape (reads, n+1)
x_i           = (1 + s_i * s_ancilla) // 2      # last spin is the ancilla gauge
```

The mapping is exact — `kw.common.hamiltonian(ising, s) + offset == xᵀQx` for the
mapped binary, for every spin configuration (verified in `tests/test_core.py`).

### How to call the **SA** simulated-annealing optimizer

```python
import kaiwu as kw
kw.license.init(user_id="...", sdk_code="...")          # once per environment

ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)
sa = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=10.0,    # matched to the QUBO energy scale (see below)
        alpha=0.999,                 # cooling factor
        cutoff_temperature=0.01,
        iterations_per_t=2000,
        size_limit=300,              # number of reads
        rand_seed=42)
spins = sa.solve(ising)              # (<=300, n+1) spin solutions
```

In this package that is `backend="sa"` (the default): `GPMDock(backend="sa")`,
or `qdock_kaiwu.solve_qubo(Q, backend="sa")`.

### How to call the **CIM** real machine

```python
import kaiwu as kw
kw.license.init(user_id="...", sdk_code="...")
kw.common.CheckpointManager.save_dir = "/tmp/kaiwu_cim"   # cache submitted tasks

ising, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q)
cim = kw.cim.CIMOptimizer(task_name="qdock_demo",
                          wait=True,            # block until the machine returns
                          interval=1,           # poll every minute
                          task_mode="quota",
                          sample_number=10)     # number of samples
spins = cim.solve(ising)            # submitted to the photonic CIM, polled back
```

In this package that is `backend="cim"`: `GPMDock(backend="cim")`, or
`qdock_kaiwu.solve_qubo(Q, backend="cim", task_name="...")`. The CIM is a real
machine — a submission takes ~1 minute and consumes quota; it suits small QUBOs
(the docking-as-QUBO demo in the notebook). A full docking QUBO has ~10³ spins
and needs a correspondingly large hardware quota.

### Setting the SA initial temperature

The SA optimizer exposes the full schedule, so set the **initial temperature to
the QUBO's energy scale**. For these docking QUBOs that scale is small
(T₀ ≈ 10), set by the rewards and the K_dist terms — not by K_mono. At T₀ = 100
the optimizer accepts a monogamy-violating move (ΔE ≈ K_mono = 25) with
probability `exp(−25/100) = 0.78`, i.e. it ignores the constraints and
random-walks; at T₀ ≈ 10 (with slow cooling, α = 0.999) the search stays in the
feasible region. Measured on GPM 1y6r:

| T₀ | best E | mRMSD |
|---|---|---|
| 100 | −14.1 | 2.9 Å |
| **10** | **−16.5** | **0.15 Å** |

So a higher T₀ helps only with a cooling schedule slow enough to spend the budget
in the productive temperature window; with a fixed budget, an over-high T₀ wastes
it on a high-temperature random walk. The per-encoding T₀ is in `params.py`. The
dense docking QUBO has high read-to-read variance, so the worked numbers use a
fixed `seed=42` and many reads.

---

## 3. Install

The Kaiwu wheel is CPython-3.10 and pins `numpy==2.2.6`, so it lives in its own
env; the chemistry CLIs (grids, prep, scoring) go in a second env.

```bash
# env A: Kaiwu + Python
conda create -y -n qdock python=3.10
conda run -n qdock pip install /path/to/kaiwu-1.3.1-cp310-none-any.whl numpy==2.2.6 prody scipy jupyter

# env B: chemistry CLIs (no ADFR needed for GPM)
conda create -y -n chem -c conda-forge python=3.11 autogrid vina openbabel
```

Kaiwu license (free, <https://platform.qboson.com>): each participant uses their
own `user_id` / `sdk_code`.

```bash
export KAIWU_USER_ID=...      # qdock_kaiwu.init_license() reads these,
export KAIWU_SDK_CODE=...      # or pass them to init_license(...)
```

`autosite` (FAM's pocket finder) ships only with the ADFR suite; without it FAM
uses the built-in `pocket_points` substitute. macOS arm64: if a `.so` fails to
load, run `xattr -dr com.apple.quarantine <kaiwu dir>` once.

---

## 4. Run

```python
import os
os.environ["PATH"] = os.path.expanduser("~/miniforge3/envs/chem/bin") + ":" + os.environ["PATH"]
from qdock_kaiwu import GPMDock, evaluate, init_license

init_license(user_id="...", sdk_code="...")          # or set the env vars
g = GPMDock(backend="sa", workdir="run_1y6r")         # backend="cim" for the machine
g.make_receptor("data/1y6r_protein.pdb")
g.make_ligand(["data/1y6r_ligand.mol2"])
g.make_box_ligand("data/1y6r_ligand.mol2")           # grid 2.0 Å (GPM default)
lig = g.ligands[0]
poses = g.dock(lig)                                  # (n_poses, n_atoms, 3)
print("mRMSD:", evaluate.mrmsd(poses, lig.coords, lig.elements), "Å")
```

- `notebooks/qdock_kaiwu_workshop.ipynb` — the guided session (QUBO → Kaiwu SA &
  CIM on a small instance → GPM/FAM redock + Vina → grid resolution).
- `scripts/benchmark.py` — redocks 1y6r at both resolutions (`QDOCK_BACKEND=sa`).
- `tests/test_core.py` — license-free correctness checks.

---

## 5. Grid resolution

A ligand atom cannot match closer than the nearest site, so grid spacing sets a
floor on accuracy — the *discretization error*, which the paper ties to mRMSD
(R² ≈ 0.93). Refining the grid lowers this floor at the cost of more qubits. See
`BENCHMARK.md` for the measured GPM 2.0→1.5 Å and FAM 1.0→0.5 Å numbers: the
discretization error drops monotonically, while realizing it in mRMSD needs a
solver budget that keeps pace with the larger QUBO.

---

## 6. Comparison with classical docking

Published sampling power (CASF-2016, 257 redocking cases, 30 poses, success =
mRMSD < 2 Å):

| method | success @ 2 Å | avg. mRMSD |
|---|---|---|
| Glide SP (classical reference) | 93.4 % | 1.0 Å |
| **GPM** | 87.5 % | 1.1 Å |
| **FAM** | 67.3 % | 1.8 Å |

The Vina scores in this repo are an independent scoring yardstick on QDock's
sampled poses.

---

## 7. Layout

```
qdock_kaiwu/   params · qubo · backends · gpm · fam · geometry · io · scoring · evaluate · tools
notebooks/     qdock_kaiwu_workshop.ipynb
scripts/       benchmark.py
tests/         test_core.py
data/          1y6r_protein.pdb · 1y6r_ligand.mol2
```

## 8. References

Zha J. *et al.* "Encoding Molecular Docking for Quantum Computers."
*J. Chem. Theory Comput.* 2024. DOI: 10.1021/acs.jctc.3c00943.
Kaiwu SDK © QBoson. AutoGrid, AutoDock Vina, OpenBabel under their own licenses.
